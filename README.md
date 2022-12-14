#  Dual-Semantic Consistency Learning for Visible-Infrared Person Re-Identification (TIFS 2022)

<p align="left">
  <br>
    <a href='https://ieeexplore.ieee.org/document/9963944'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
</p>
This repository is an official implementation of DSCNet, a strong baseline for VI-ReID
<div align="left">
  <img src="assets\teaser.png" width="60%" height="60%" />
</div><br/>

## 📈 News 
**2022.10.24** DSCNet has been formally accepted by IEEE Transactions on Information Forensics & Security.  
**2022.10.15** Code Release.  

## 🚀 Highlight 
1. **Insights**: This paper derives the modality discrepancy from the channel-level semantic inconsistency. It is the **FIRST** method to address the limitations on the channel-level representation.
2. **A strong baseline**: Faster Convergence and Outstanding Performance for VI-ReID.

| Model  | Training Epochs | Rank-1 (%) | mAP(%)    | Training Time |
| ------ | --------------- | ---------- | --------- | ------------- |
| MCLNet | 200             | 65.30      | 61.59     | 24 hours      |
| DSCNet | **50**          | **73.89**  | **69.47** | **5 hours**   |

## ⚙️ Setup environment
* Clone this repo: 
```shell
git clone https://github.com/bitreidgroup/DSCNet.git && cd DSCNet
```
* Create a conda environment and activate the environment.
```shell
conda env create -f environment.yaml &&  conda activate dsc
```

*We recommend Python = 3.6, CUDA = 10.0, Cudnn = 7.6.5, Pytorch = 1.2, and CudaToolkit = 10.0.130 for the environment.* 

## 🔧 Preparing dataset

- **SYSU-MM01 Dataset** :  The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm). 

- We preprocess the SYSU-MM01 dataset to speed up the training process. The identities of cameras will be also stored in ".npy" format.
```shell
python utils/pre_process_sysu.py
```
- **RegDB Dataset** :  The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

  (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website).

## ⏳  Training

*You may need manually define the data path first. More details are in the config files.*

* SYSU-MM01 Dataset (all-search)

```shell
bash scripts/train_sysu_all.sh
```

* SYSU-MM01 Dataset (indoor-search)

```shell
bash scripts/train_sysu_indoor.sh
```

## 💽 Testing

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python scripts/test.py --ckpt [CKPT_PATH] --config [CONFIT] 
```

For example :  You can test the checkpoints by running the commands below.

```shell
bash scripts/eval_sysu.sh
```



 ## ⏰ Reproduce our experimental results

DSCNet: We provide some experimental  results on the **SYSU-MM01** datasets with pretrained models. These model are trained on 1x 2080ti 

| config            | Rank-1(%) | Rank-10(%) |mAP(%)    | Training Log | Pretrained |
|:--------:|:----------:|:---------:|:--------:|:--------:|:-------------:|
| SYSU-MM01(all-search) | 73.89 | 96.27 | 69.47 | log(TBA) |   [Weights](https://drive.google.com/uc?id=1nm_plUl4HjikpL1vwfHGkVBnIA02kMls)     |
| SYSU-MM01(indoor-search) | 79.35 | 95.74 | 82.68 | log(TBA) | Weights(TBA) |

Before running the commands below, please update the config files on the setting of  `resume`.

```shell
python scripts/reproduce.sh
```


### 💾GPUs

All our experiments were performed on a single NVIDIA GeForce 2080 Ti GPU

| Training Datasets | Approximate GPU memory | Approximate training time |
| ----------------- | ---------------------- | ------------------------- |
| SYSU-MM01         | 9GB                    | 5 hours                   |
| RegDB             | 6GB                    | 3 hours                   |


### Citation

If this repository helps your research, please cite :

```
@article{zhang2022dual,
  title={Dual-Semantic Consistency Learning for Visible-Infrared Person Re-Identification},
  author={Zhang, Yiyuan and Kang, Yuhao and Zhao, Sanyuan and Shen, Jianbing},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2022},
  publisher={IEEE}
}
```

###  📄 References.

1. Y. Zhang, Y. Kang, S. Zhao, and J. Shen. Dual-Semantic Consistency Learning for Visible-Infrared Person Re-Identification. IEEE Transactions on Information Forensics & Security, 2022.
2. M. Ye, W. Ruan, B. Du, and M. Shou. Channel Augmented Joint Learning for Visible-Infrared Recognition. IEEE International Conference on Computer Vision (ICCV), 2021.

###  ✉️ Contact.

If you have some questions, feel free to contact me.  yiyuanzhang.ai@gmail.com
