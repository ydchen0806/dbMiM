# Self-supervised Neuron Segmentation with Multi-agent Reinforcement Learning (IJCAI23)

This repository contains the official implementation of the paper **Self-supervised Neuron Segmentation with Multi-agent Reinforcement Learning**, presented at IJCAI 2023. You can find the paper [here](https://www.ijcai.org/proceedings/2023/0068.pdf).

## Environment Setup

To streamline the setup process, we provide a Docker image that can be used to set up the environment with a single command. The Docker image is available at:

```sh
docker pull registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26
```
## Dataset Download

The datasets required for pre-training and segmentation are as follows:

| Dataset Type          | Dataset Name | Description                              |
|-----------------------|--------------|------------------------------------------|
| Pre-training Dataset  | Region of FAFB Dataset | Fly brain dataset for pre-training       |
| Segmentation Dataset  | CREMI Dataset| Challenge on circuit reconstruction datasets|

### Pre-training Dataset: FAFB

The FAFB dataset is used for pre-training. Please follow the instructions provided in the paper to acquire and preprocess this dataset.

### Segmentation Dataset: CREMI

The CREMI dataset is used for the segmentation tasks. Detailed instructions for downloading and preprocessing can be found on the [CREMI Challenge website](https://cremi.org/).
