# FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer

#### [website](https://flatformer.mit.edu/) | [paper](https://arxiv.org/abs/2301.08739)

## Results

### 3D Object Detection (on Waymo validation)

| Model                                                        | #Sweeps | mAP/H_L1 | mAP/H_L2 | Veh_L1     | Veh_L2    | Ped_L1    | Ped_L2    | Cyc_L1    | Cyc_L2    |
| ------------------------------------------------------------ | ------- | -------- | -------- | ---------- | --------- | --------- | --------- | --------- | --------- |
| [FlatFormer](https://github.com/mit-han-lab/flatformer-dev/blob/main/configs/flatformer/flatformer_waymo_D1_2x_3class.py) | 1       | 76.1/73.4 | 69.7/67.2 | 77.5/77.1 | 69.0/68.6 | 79.6/73.0 | 71.5/65.3 | 71.3/70.1 | 68.6/67.5 |
| [FlatFormer](https://github.com/mit-han-lab/flatformer-dev/blob/main/configs/flatformer/flatformer_waymo_D1_2x_3class_2f.py) | 2       | 78.9/77.3 | 72.7/71.2 | 79.1/78.6 | 70.8/70.3 | 81.6/78.2 | 73.8/70.5 | 76.1/75.1 | 73.6/72.6 |
| [FlatFormer](https://github.com/mit-han-lab/flatformer-dev/blob/main/configs/flatformer/flatformer_waymo_D1_2x_3class_3f.py) | 3       | 79.6/78.0 | 73.5/72.0 | 79.7/79.2 | 71.4/71.0 | 82.0/78.7 | 74.5/71.3 | 77.2/76.1 | 74.7/73.7 |
| FlatFormer with 1/34 trainSet & 1/8 testSet                  | 1       | 9.32/9.03 | 8.33/8.08 | 10.30/10.22 | 8.83/8.75 | 6.24/5.68 | 5.17/4.71 | 11.43/11.20 | 10.99/10.77 |
| FlatFormer with 1/34 trainSet & 1/8 testSet                  | 3       | 9.92/9.72 | 8.86/8.69 | 11.04/10.96 | 9.48/9.41 | 6.82/6.52 | 5.66/5.42 | 11.89/11.67 | 11.45/11.23 |

## Usage

### Prerequisites

#### conda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p ./miniconda3
./miniconda3/bin/conda init bash
./miniconda3/bin/conda init zsh

conda create -n flatformer python=3.6
```

#### libraries

The code is built with following libraries:

* Python >= 3.6, <3.8
* [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, <= 1.10.2
* [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
* [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.14.0

```
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install opencv-python==4.6.0.66
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install numba==0.48.0
pip install trimesh==2.35.39
pip install future tensorboard
pip install way-open-dataset-tf-2.1.0==1.2.0
pip install yapf==0.40.1
pip install setuptools==59.5.0
pip install nuscenes-devkit
pip install scikit-image
```

#### flash-attention & flatformer

After installing these dependencies, please run this command to install the codebase:

```
git clone https://github.com/Dao-AILab/flash-attention.git
cd ./flash-attention
git checkout f515c77f2528b5062ebcc6c905c8817ca0ac0ad1
python setup.py develop

git clone https://github.com/mit-han-lab/flatformer
cd ./flatformer
python setup.py develop
```

### Dataset Preparation

Please follow the instructions from [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/waymo_det.md)  to download and preprocess the Waymo Open Dataset v1.2.0.

#### waymo dataset

Please download the v1.2.0 datasets from [waymo](https://waymo.com/open/) and its data split from [mmdet3d](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o), you can choose the tar version or the individual files.

The folder structure should be organized as follows after the downloading.
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets
```

You can use the gloud tools to download the datasets. Please keep at least 3T memory.
```
curl https://sdk.cloud.google.com | bash
gcloud auth login
gcloud auth application-default login

<path-to-cloud-sdk>/google-cloud-sdk/bin/gsutil -m cp -r "gs://waymo_open_dataset_v_1_2_0/" .
# or
<path-to-cloud-sdk>/google-cloud-sdk/bin/gsutil -m cp -r "gs://waymo_open_dataset_v_1_2_0_individual_files/" .
```

#### mmdet3d

```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d/
git checkout v0.15.0
python -m pip install -v -e .
```

#### make the datasets
```
cd mmdetection3d/
cd data
ln -s <path-to-waymo>/waymo waymo
python tools/create_data.py waymo --root-path ./data/waymo --out-dir ./data/waymo --workers 128 --extra-tag waymo --version v1.4
```

If you use the subset of waymo datasets, please remake the data split files according to files in training, testing or waymo_gt_database of `path-to-waymo/waymo/kitti_format/ImageSets`. If you only need to unzip the first zip of train, test and val, you can refer to `ImageSets/`. In the meantime I implemented the ability to skip non-existing files.

After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── waymo
│   │   ├── waymo_format
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
│   │   │   ├── gt.bin
│   │   ├── kitti_format
│   │   │   ├── ImageSets
│   │   │   ├── training
│   │   │   ├── testing
│   │   │   ├── waymo_gt_database
│   │   │   ├── waymo_infos_trainval.pkl
│   │   │   ├── waymo_infos_train.pkl
│   │   │   ├── waymo_infos_val.pkl
│   │   │   ├── waymo_infos_test.pkl
│   │   │   ├── waymo_dbinfos_train.pkl
```

#### evaluation waymo

```
wget https://github.com/bazelbuild/bazel/releases/download/0.28.0/bazel-0.28.0-installer-linux-x86_64.sh
bash bazel-0.28.0-installer-linux-x86_64.sh
 
git clone https://github.com/waymo-research/waymo-open-dataset.git
cd waymo-open-dataset/
git checkout r1.3
./configure.sh
bazel clean
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cp bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main /<path-to-flatformer>/flatformer/mmdet3d/core/evaluation/waymo_utils/
```

### Training

```
# multi-gpu training
bash tools/dist_train.sh configs/flatformer/$CONFIG.py 8 --work-dir $CONFIG/ --cfg-options evaluation.pklfile_prefix=./work_dirs/$CONFIG/results evaluation.metric=waymo
```

Such as:
```
bash tools/dist_train.sh configs/flatformer/flatformer_waymo_D1_2x_3class_3f.py 4 --work-dir waymo_D1_2x_3class_3f/ --cfg-options evaluation.pklfile_prefix=./waymo_D1_2x_3class_3f/results evaluation.metric=waymo
```

### Evaluation

```
# multi-gpu testing
bash tools/dist_test.sh configs/flatformer/$CONFIG.py /work_dirs/$CONFIG/latest.pth 8 --eval waymo
```

Such as:
```
bash tools/dist_test.sh configs/flatformer/flatformer_waymo_D1_2x_3class.py waymo_D1_2x_3class/epoch_23.pth 4 --eval waymo
```

## Citation

If FlatFormer is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```
@inproceedings{liu2023flatformer,
  title={FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer},
  author={Liu, Zhijian and Yang, Xinyu and Tang, Haotian and Yang, Shang and Han, Song},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
