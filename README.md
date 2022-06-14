# DDR
DDR: Deep-Discrete spherical Registration

This repository contains the code for performing spherical (cortical) surface registration using deep learning. This is the official PyTorch implementation of the paper [A Deep-Discrete Learning Framework for Spherical Surface Registration](https://arxiv.org/abs/2203.12999), accepted at the MICCAI 2022 conference.  


![DDR](https://github.com/mohamedasuliman/DDR/blob/main/doc/model.jpg)



## Data 

The data used for the image registration task comes from the [HCP dataset](https://www.humanconnectome.org/). 

## Code Usage

## Training

To train the affine registration network, run:
```
DDR_affine_train.py
```

To train the first stage of the non-linear registration network, run:

```
DDR_coarse_train.py
```

Note that you have to manually set the directories where you want to save your corresponding models and results at the beginning of these codes. 

**More Coming soon**

## Installation

**Coming soon**

## Citation

Please cite this work if you found it useful:

[A Deep-Discrete Learning Framework for Spherical Surface Registration](https://arxiv.org/abs/2203.12999)

```
@article{suliman2022deep,
  title={A Deep-Discrete Learning Framework for Spherical Surface Registration},
  author={Suliman, Mohamed A and Williams, Logan ZJ and Fawaz, Abdulah and Robinson, Emma C},
  journal={arXiv preprint arXiv:2203.12999},
  year={2022}
}
