# Fisher Deep Domain Adaptation

This repository contains the code of our SDM20 paper "Fisher Deep Domain Adaptation". 

## Usage
### Dependencies
- PyTorch >= 0.4.0
- Python3

### Prepare Datasets
Two domain adaptation benchmarks, [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/) and [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/), are used. 

Each benchmark has several domains. For each domain, please create a txt file that lists the image file names and their labels. Each line in the txt file is structured as "image_file_path label". 

An example of the Office-31 dataset is provided in the `data/office` folder. You may replace the `[local_path]` in the txt files with the path where you save the dataset. 

### Training
Three transfer criterion, Maximum Mean Discrepancy (MMD), deep CORrelation ALignment (CORAL), and domain adversarial loss, are used in the experiment. Use the `train_ada.py` script if you use the domain adversarial loss. Use the `train_feature.py` if you use the MMD or CORAL loss. 

An example of training with the MMD loss on the Amazon -> DSLR task is given in the following. 

```
python train_feature.py --gpu_id <gpu_id> \
                        --net ResNet50 \
                        --dset office \
                        --test_interval 500 \
                        --snapshot_interval 10000 \
                        --ly_type cosine \
                        --loss_type mmd \
                        --fisher_loss_type tr \
                        --output_dir <output_path> \
                        --s_dset_path ../data/office31/amazon_31_list.txt \
                        --t_dset_path ../data/office31/dslr_31_list.txt \
                        --em_loss_coef 0.1 \
                        --inter_loss_coef 1. \
                        --intra_loss_coef 1. \
                        --trade_off 1.
```

The weight of the transfer criteria is controlled by the `trade_off` argument. If you do not want to use a transfer criteria, please set the `trade_off` argument to 0. 

The Fisher loss has two instantiations, Trace Ratio and Trace Difference. The instantiation is determined by the `fisher_loss_type` argument. The within-class and between-class penalty are controlled by the `intra_loss_coef` and `inter_loss_coef`, respectively. 

### Evaluation
To evaluate the model, the following command can be used. 

```
python eval_da.py --gpu_id <gpu_id> \
                  --net ResNet50 \
                  --dset office \
                  --ly_type cosine \
                  --ckpt_path <path of the saved model> \
                  --s_dset_path ../data/office31/amazon_31_list.txt \
                  --t_dset_path ../data/office31/dslr_31_list.txt
```

## Citation
If you use this code, please consider citing our paper: 

```
Yinghua Zhang, Yu Zhang, Ying Wei, Kun Bai, Yangqiu Song, Qiang Yang. Fisher Deep Domain Adaptation. In: Proceedings of SIAM International Conference on Data Mining (SDM), Cincinnati, Ohio, USA, 2020. 
```

## Contact
yzhangdx@cse.ust.hk
