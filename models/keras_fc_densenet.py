"""
Build a FC-DenseNet model as described in https://arxiv.org/abs/1611.09326.
"""
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Dense, Reshape, Flatten, Embedding, Dropout, Input
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
from keras.layers import Conv2DTranspose, Conv2D, concatenate, Activation
from keras.layers import UpSampling2D
from keras.models import Model
from keras.activations import relu
from keras import regularizers
from keras.regularizers import l2
from keras.layers import Lambda


def _channel_dim(data_format):
    return 1 if data_format == 'channels_first' else -1

def _conv_block(x, nb_filter, bn_momentum, dropout_rate=None, block_prefix='ConvBlock', 
               data_format='channels_last'):
    """
    Adds a single layer (conv block) of a dense block. It is composed of a 
    batch normalization, a relu, a convolution and a dropout layer.
    
    Args
        x: input tensor
        nb_filter: number of convolution filters, this is also the number 
            of feature maps returned by the block
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers
        dropout_rate: dropout rate
        block_prefix: prefix for naming
        data_format: 'channels_first' or 'channels_last'
        
    Return:
        output tensor
    """
    with tf.name_scope(block_prefix):
        concat_axis = _channel_dim(data_format)
        x = BatchNormalization(momentum=bn_momentum, axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

        # FC-DenseNet paper does not say anything about stride in the conv block, assume default (1,1)
        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,
                   data_format=data_format)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x



def _dense_block(x, nb_layers, nb_filter, growth_rate, bn_momentum, dropout_rate=None, grow_nb_filters=True, 
                  return_concat_list=False, block_prefix='DenseBlock', data_format='channels_last'):
    """
    Adds a dense block. The input and output of each conv block is 
    concatenated and used as input for the next conv block. The result
    is the concatenated outputs of all conv blocks. In addition
    the first element of the result is the input tensor `x`, this 
    works as a shortcut connection.
    
    The block leaves height and width of the input unchanged and
    adds `nb_layers` * `growth_rate` feature maps.
    
    Args:
        x: input tensor
        nb_layers: the number of conv_blocks in the dense block
        nb_filter: filter count that will be incremented for each conv block            
        growth_rate: growth rate of the dense block, this is the number
            of filters in each conv block
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers    
        dropout_rate: dropout rate
        grow_nb_filters: flag if nb_filters should be updated
        block_prefix: prefix for naming
        data_format: 'channels_first' or 'channels_last'
    
    Return:
        x: tensor concatenated from [x, cb_1_out, ..., cb_n_out]
        nb_filter: updated nb_filters
        x_list: list [x, cb_1_out, ..., cb_n_out]
    """
    with tf.name_scope(block_prefix):
        concat_axis = _channel_dim(data_format)
        x_list = [x]
        for i in range(nb_layers):
            cb = _conv_block(x, growth_rate, bn_momentum, dropout_rate, data_format=data_format,
                            block_prefix='ConvBlock_%i' % i)
            x_list.append(cb)
            x = concatenate([x, cb], axis=concat_axis)
            if grow_nb_filters:
                nb_filter += growth_rate

        return x, nb_filter, x_list


def _transition_down_block(x, nb_filter, bn_momentum, weight_decay=1e-4, transition_pooling='max', 
                          block_prefix='TransitionDown', data_format='channels_last'):
    """
    Adds a pointwise convolution layer (with batch normalization and relu),
    and a pooling layer. 
    
    The block cuts height and width of the input in half.
    
    Args:
        x: input tensor
        nb_filter: number of convolution filters, this is also the number 
            of feature maps returned by the block
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers    
        weight_decay: weight decay factor
        transition_pooling: aggregation type for pooling layer
        block_prefix: prefix for naming
        data_format: 'channels_first' or 'channels_last'

    Return:
        output tensor
    """
    with tf.name_scope(block_prefix):
        concat_axis = _channel_dim(data_format)
        x = BatchNormalization(momentum=bn_momentum, axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False, 
                   kernel_regularizer=l2(weight_decay), data_format=data_format)(x)
        if transition_pooling == 'avg':
            x = AveragePooling2D((2, 2), strides=(2, 2), data_format=data_format)(x)
        elif transition_pooling == 'max':
            x = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format)(x)
        return x        

def _transition_up_block(x, nb_filters, type='deconv', weight_decay=1e-4, 
                          block_prefix='TransitionUp', data_format='channels_last'):
    """
    Adds an upsampling block. 
    
    The block doubles height and width of the input.
    
    Args:
        x: input tensor
        nb_filter: number of convolution filters, this is also the number 
            of feature maps returned by the block
        type: type of upsampling operation: 'upsampling' or 'deconv'
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
        data_format: 'channels_first' or 'channels_last'
    Returns:
        output tensor
    """
    with tf.name_scope(block_prefix):
        if type == 'upsampling':
            return UpSampling2D(data_format=data_format)(x)
        else:
            return Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same', strides=(2, 2),
                                kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
                                data_format=data_format)(x)
        

def _create_fc_dense_net(img_input, 
                           nb_classes,
                           nb_dense_block=3,
                           nb_layers_per_block=4,
                           init_conv_filters=48, 
                           growth_rate=12, 
                           initial_kernel_size=(3, 3), 
                           transition_pooling='max',
                           upsampling_type='deconv',
                           bn_momentum=0.9,
                           weight_decay=1e-4, 
                           dropout_rate=0.2,  
                           final_softmax=False,
                           name_scope='DenseNetFCN',
                           data_format='channels_last'):
    """
    Create a fully convolutional DenseNet.
    
    Args:
        img_input: tuple of shape (batch_size, channels, height, width) or (batch_size, height, width, channels)
            depending on data_format
        nb_classes: number of classes
        nb_dense_block: number of dense blocks on the downsampling path, without the bottleneck dense block
        nb_layers_per_block: number of layers in dense blocks, can be an int if all dense blocks have the
            same number of layers or a list of ints with the number of layers in the dense block on the
            downsampling path and the bottleneck dense block
        init_conv_filters: number of filters in the initial concolution
        growth_rate: number of filters in each conv block
        dropout_rate: dropout rate
        initial_kernel_size: the kernel of the first convolution might vary in size based 
            on the application
        transition_pooling: aggregation type of pooling in the downsampling layers: 'avg' or 'max'            
        upsampling_type: type of upsampling operation used: 'upsampling' or 'deconv'
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers
        weight_decay: weight decay
        dropout_rate: dropout rate
        final_softmax: if True a final softmax activation layers is added, otherwise the network 
            returns unnormalized log probabilities  
        data_format: 'channels_first' or 'channels_last'

    Returns:
        Tensor with shape: (height * width, nb_classes): class probabilities if final_softmax==True, 
        otherwiese the unnormalized output of the last layer
    """
    with tf.name_scope(name_scope):

        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError('Invalid data_format: %s. Must be one of [channels_first, channels_last]' % data_format)
        
        if data_format == 'channels_first':
            concat_axis = 1
            _, channel, row, col = img_input.shape
        else:
            concat_axis = -1
            _, row, col, channel = img_input.shape
        
        if channel not in [1,3]:
            raise ValueError('Invalid number of channels: %d. Must be one of [1,3]' % channel)

        upsampling_type = upsampling_type.lower()
        if upsampling_type not in ['upsampling', 'deconv']:
            raise ValueError('"upsampling_type" must be one of [upsampling, deconv]')

        # `nb_layers` is a list with the number of layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block + 1):
                raise ValueError('If `nb_layers_per_block` is a list, its length must be '
                                 '(`nb_dense_block` + 1)')

            bottleneck_nb_layers = nb_layers[-1]
            rev_layers = nb_layers[::-1]
            nb_layers.extend(rev_layers[1:])
        else:
            bottleneck_nb_layers = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

        tf.logging.info('Layers in each dense block: %s' % nb_layers)

        # make sure we can concatenate the skip connections with the upsampled
        # images on the upsampling path
        img_downsize_factor = 2**nb_dense_block
        if row % img_downsize_factor != 0:
            raise ValueError('Invalid image height %d. Image height must be a multiple of '
                             '2^nb_dense_block=%d' % (row, img_downsize_factor))
        if col % img_downsize_factor != 0:
            raise ValueError('Invalid image width %d. Image width must be a multiple of '
                             '2^nb_dense_block=%d' % (col, img_downsize_factor))

        # Initial convolution
        with tf.name_scope('Initial'):
            x = Conv2D(init_conv_filters, initial_kernel_size, kernel_initializer='he_normal', padding='same', 
                       use_bias=False, kernel_regularizer=l2(weight_decay), data_format=data_format)(img_input)
            x = BatchNormalization(momentum=bn_momentum, axis=concat_axis, epsilon=1.1e-5)(x)
            x = Activation('relu')(x)

        # keeps track of the current number of feature maps
        nb_filter = init_conv_filters
        
        # collect skip connections on the downsampling path so that
        # they can be concatenated with outputs on the upsampling path
        skip_list = []
                            
        # Build the downsampling path by adding dense blocks and transition down blocks
        for block_idx in range(nb_dense_block):
            x, nb_filter, _ = _dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bn_momentum=bn_momentum,
                                           dropout_rate=dropout_rate, data_format=data_format, 
                                           block_prefix='DenseBlock_%i' % block_idx)

            skip_list.append(x)
            x = _transition_down_block(x, nb_filter, weight_decay=weight_decay, bn_momentum=bn_momentum,
                                       transition_pooling=transition_pooling, data_format=data_format,
                                       block_prefix='TransitionDown_%i' % block_idx)

        # Add the bottleneck dense block.
        _, nb_filter, concat_list = _dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate, bn_momentum=bn_momentum, 
                                                 dropout_rate=dropout_rate, data_format=data_format,
                                                 block_prefix='Bottleneck_DenseBlock_%i' % nb_dense_block)

        tf.logging.info('Number of skip connections: %d' %len(skip_list))

        # reverse the list of skip connections
        skip_list = skip_list[::-1]  
        
        # Build the upsampling path by adding dense blocks and transition up blocks
        for block_idx in range(nb_dense_block):
            n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]
                    
            # upsampling block must upsample only the feature maps (concat_list[1:]),
            # not the concatenation of the input with the feature maps
            l = concatenate(concat_list[1:], axis=concat_axis, name='Concat_DenseBlock_out_%d' % block_idx)
            
            t = _transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay,
                                      data_format=data_format, block_prefix='TransitionUp_%i' % block_idx)

            # concatenate the skip connection with the transition block output
            x = concatenate([t, skip_list[block_idx]], axis=concat_axis, name='Concat_SkipCon_%d' % block_idx)
        
            # Dont allow the feature map size to grow in upsampling dense blocks
            x_up, nb_filter, concat_list = _dense_block(x, nb_layers[nb_dense_block + block_idx + 1], nb_filter=growth_rate, 
                                                        growth_rate=growth_rate, bn_momentum=bn_momentum, dropout_rate=dropout_rate, 
                                                        grow_nb_filters=False, data_format=data_format,
                                                        block_prefix='DenseBlock_%d' % (nb_dense_block + 1 + block_idx))            
        
        # final convolution
        with tf.name_scope('Final'):
            l = concatenate(concat_list[1:], axis=concat_axis)
            x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', use_bias=False, data_format=data_format)(l)
            print(x)
            # x = Reshape((row * col, nb_classes), name='logit')(x)
            print(x)
                
            if final_softmax:
                x = Activation('softmax', name='softmax')(x)

        return x


def build_FC_DenseNet56(nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    """Build FC-DenseNet56"""
    inputs = Input(shape=input_shape)
    logits = _create_fc_dense_net(inputs,
                               nb_classes=nb_classes,
                               nb_dense_block=5,
                               nb_layers_per_block=4,
                               growth_rate=12,
                               init_conv_filters=48, 
                               dropout_rate=dropout_rate,
                               final_softmax=final_softmax,
                               name_scope='FCDenseNet56',
                               data_format=data_format)
    print(logits)
    out_1 = Lambda(lambda x: x, name='out_1')(logits)
    out_2 = Lambda(lambda x: x, name='out_2')(logits)
    out_3 = Lambda(lambda x: x, name='out_3')(logits)
    out_4 = Lambda(lambda x: x, name='out_4')(logits)
    out_5 = Lambda(lambda x: x, name='out_5')(logits)
    return Model(inputs=inputs,
                 outputs=[out_1,
                          out_2,
                          out_3,
                          out_4,
                          out_5])
    # return Model(inputs=inputs, outputs=logits)


def build_FC_DenseNet67(nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    """Build FC-DenseNet67"""
    inputs = Input(shape=input_shape)
    logits = _create_fc_dense_net(inputs,
                               nb_classes=nb_classes,
                               nb_dense_block=5,
                               nb_layers_per_block=5,
                               growth_rate=16,
                               init_conv_filters=48, 
                               dropout_rate=dropout_rate,
                               final_softmax=final_softmax,
                               name_scope='FCDenseNet67',
                               data_format=data_format)
    return Model(inputs=inputs, outputs=logits)


def build_FC_DenseNet103(nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    """Build FC-DenseNet103"""
    inputs = Input(shape=input_shape)
    logits = _create_fc_dense_net(inputs,
                               nb_classes=nb_classes,
                               nb_dense_block=5,
                               nb_layers_per_block=[4,5,7,10,12,15],
                               growth_rate=16,
                               init_conv_filters=48, 
                               dropout_rate=dropout_rate,
                               final_softmax=final_softmax,
                               name_scope='FCDenseNet103',
                               data_format=data_format)
    return Model(inputs=inputs, outputs=logits)


def build_FC_DenseNet(model_version, nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    if model_version == 'fcdn56':
        return build_FC_DenseNet56(nb_classes, final_softmax, input_shape, dropout_rate, data_format)
    elif model_version == 'fcdn67':
        return build_FC_DenseNet67(nb_classes, final_softmax, input_shape, dropout_rate, data_format)
    elif model_version == 'fcdn103':
        return build_FC_DenseNet103(nb_classes, final_softmax, input_shape, dropout_rate, data_format)
    else:
        raise ValueError('Invalid model_version: %s' % model_version)

