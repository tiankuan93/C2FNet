# C2FNet

This repository is code for papaer "Weakly-Supervised Nucleus Segmentation Based on Point Annotations: A Coarse-to-Fine Self-Stimulated Learning Strategy", which has been accepted by MICCAI 2020.


## Dependencies
Tensorflow 1.15.0

Keras 2.3.0

## Usage
### Train
1. split data to train and val set, each set has img and mask folders.
    ```
    example: 
    ./data/monuseg/train_val/
    |-- train
    |   |-- img
    |   |-- mask
    |-- val
    |   |-- img
    |   |-- mask
    ```

2. run train_one_fold.py to train segmentation model.
    * set in_dataset_fold=`train_val`, in_dataset_name=`monuseg`, 
    save_checkpoint_path=`./checkpoints/monuseg_ln`
    * set train_full_mask_flag=`True` to train fully supervised model
    * set itr_sum=`4`, which indicate that the model will train in 4 iteration, 
    0,1,2 are in first stage, 3 is in the second stage

### Test
1. run test_edge_point.py to predict result with trained model.
    * set fold=`train_val`,
    * set model_name=`LinkNet.nuclei.train_val.512_loss_0.01_0.01_0.01_0.01_1.0_train_val_r3_resume_point_edge_fake_sobel.last.h5`
    * set val_dir=`data/monuseg/train_val/val/img/`
    * set save_dir=`data/monuseg/train_val/val/result_r3/`
    * model_name and save_dir are corresponding, including r0, r1, r2, r3

### Evaluation Metrics
1. cd experiments, and run compute_metrics.py to compute evaluation metrics.
    * set base_dir=`../data/monuseg/train_val/val/`
    * set pred_sub_dirs=`['result_r0', 'result_r1', 'result_r2', 'result_r3']`
    
## Citation
If you find this code helpful, please cite our work:

Tian K, Zhang J, Shen H, et al. 
Weakly-Supervised Nucleus Segmentation Based on Point Annotations: A Coarse-to-Fine Self-Stimulated Learning Strategy[C]
//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020: 299-308.
