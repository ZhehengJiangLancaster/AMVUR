# AMVUR
This is a PyTorch implementation of our paper: [A Probabilistic Attention Model with Occlusion-aware Texture Regression for 3D Hand Reconstruction from a Single RGB Image(CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_A_Probabilistic_Attention_Model_With_Occlusion-Aware_Texture_Regression_for_3D_CVPR_2023_paper.pdf)
## Requirements
* **Python** (>=3.7)
* **PyTorch** (>=1.7.1)
* **torchvision** (>=0.8.2)
* **cuda** (>=11.0)
* **PyTorch3D** (>=0.3.0)
## Data
* Download the FreiHAND dataset from the [website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html).
* Download the HO3D dataset from the [website](https://www.tugraz.at/index.php?id=40231).
  You need to put them to `${REPO_DIR}/data` file
## Pytorch MANO layer
* Download [manopth](https://github.com/hassony2/manopth), and put the file to `${REPO_DIR}/manopth`.
* Download `MANO_RIGHT.pkl` from [here](https://mano.is.tue.mpg.de/), and put the file to `${REPO_DIR}/AMVUR/modeling/data`.
## Backbone Download
Download the `cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml` and `hrnetv2_w64_imagenet_pretrained.pth` from [HRNet models](https://github.com/HRNet/HRNet-Image-Classification), and put them to `${REPO_DIR}/models/hrnet`.


## Experiment
* Supervised Experiment
  
  Evaluation: Our pre-trained model can be downloaded from [here](https://livelancsac-my.sharepoint.com/:u:/g/personal/jiangz13_lancaster_ac_uk/Eb4oS3v4MbJFtfYGLrlbmigBO0JJFh5As05v7JvMzaTvGg?e=8O9AU3), and put the file to `${REPO_DIR}/pre_trained`.
  
  Run:
  
```bash
  python -m torch.distributed.launch --nproc_per_node=4 \
       experiments/supervised_HO3D_v2.py \
       --config_json ./experiments/config/test.json
```

  It will generate a prediction file called `pred.zip`. Afte that, please submit the prediction file to [codalab challenge](https://competitions.codalab.org/competitions/22485) and see the results.

  Training:
  
```bash
  python -m torch.distributed.launch --nproc_per_node=4 \
       experiments/supervised_HO3D_v2.py \
       --config_json ./experiments/config/train.json
```

* Weakly Supervised Experiment
  
  Evaluation:
  ```bash
  python -m torch.distributed.launch --nproc_per_node=4 \
       experiments/weakly_supervised_HO3D_v2.py \
       --config_json ./experiments/config/test.json
  ```
  
  Training:
  
```bash
  python -m torch.distributed.launch --nproc_per_node=4 \
       experiments/weakly_supervised_HO3D_v2.py \
       --config_json ./experiments/config/train.json
```

## Citation
```text
@inproceedings{jiang2023probabilistic,
  title={A Probabilistic Attention Model with Occlusion-aware Texture Regression for 3D Hand Reconstruction from a Single RGB Image},
  author={Jiang, Zheheng and Rahmani, Hossein and Black, Sue and Williams, Bryan M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={758--767},
  year={2023}
}
