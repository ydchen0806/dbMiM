# Self-supervised Neuron Segmentation with Multi-agent Reinforcement Learning (IJCAI 2023)

This repository contains the official implementation of the paper **Self-supervised Neuron Segmentation with Multi-agent Reinforcement Learning**, presented at IJCAI 2023. You can find the paper [here](https://www.ijcai.org/proceedings/2023/0068.pdf).

<div style="text-align: center;">
  <img src="framework.png" alt="The pipeline of our proposed methods" width="100%" />
  <p><b>Figure 1:</b> The pipeline of our proposed methods</p>
</div>

<div style="text-align: center;">
  <img src="decision_module.png" alt="The framework of our proposed decision module" width="50%" />
  <p><b>Figure 2:</b> The framework of our proposed decision module</p>
</div>

## Environment Setup

To streamline the setup process, we provide a Docker image that can be used to set up the environment with a single command. The Docker image is available at:

```sh
docker pull registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26
```
## Dataset Download

The datasets required for pre-training and segmentation are as follows:

| Dataset Type          | Dataset Name           | Description                              | URL                                           |
|-----------------------|------------------------|------------------------------------------|-----------------------------------------------|
| Pre-training Dataset  | Region of FAFB Dataset | Fly brain dataset for pre-training       | [EM Pretrain Dataset](https://huggingface.co/datasets/cyd0806/EM_pretrain_data)  |
| Segmentation Dataset  | CREMI Dataset          | Challenge on circuit reconstruction datasets| [CREMI Dataset](https://cremi.org/)           |

### Pre-training Dataset: Region of FAFB

The FAFB region dataset is used for pre-training. Please follow the instructions provided in the paper to acquire and preprocess this dataset. You can download it from the Hugging Face [EM Pretrain Dataset](https://huggingface.co/datasets/cyd0806/EM_pretrain_data). Use the subfolder `FAFB_hdf` to match the paper's settings, or use additional relevant data to achieve better results.

To use this dataset, please refer to the license provided [here](#license-important-).

### Segmentation Dataset: CREMI

The CREMI dataset is used for the segmentation tasks. Detailed instructions for downloading and preprocessing can be found on the [CREMI Challenge website](https://cremi.org/).

## Usage Guide

### 1. Pretraining
```
python pretrain.py -c pretraining_all -m train
```
### 2. Finetuning
```
python finetune.py -c seg_3d -m train -w [your pretrained path]
```

# License (Important !!!)

<details>
<summary>Usage Notes</summary>

### **Before the public release of the data, the following usage restrictions must be met:**

1. **Non-commercial Use:** Users do not have the rights to copy, distribute, publish, or use the data for commercial purposes or develop and produce products. Any format or copy of the data is considered the same as the original data. Users may modify the content and convert the data format as needed but are not allowed to publish or provide services using the modified or converted data without permission.
   
2. **Research Purposes Only:** Users guarantee that the authorized data will only be used for their own research and will not share the data with third parties in any form.

3. **Citation Requirements:** Research results based on the authorized data, including books, articles, conference papers, theses, policy reports, and other publications, must cite the data source according to citation norms, including the authors and the publisher of the data.

4. **Prohibition of Profit-making Activities:** Users are not allowed to use the authorized data for any profit-making activities.

5. **Termination of Data Use:** Users must terminate all use of the data and destroy the data (e.g., completely delete from computer hard drives and storage devices/spaces) upon leaving their team or organization or when the authorization is revoked by the copyright holder.

### **Data Information**

- **Sample Source:** Mouse MEC MultiBeam-SEM, Intelligent Institute Brain Imaging Platform (Wafer 4 at layer VI, wafer 25, wafer 26, and wafer 36 at layer II/III)
- **Resolution:** 8nm x 8nm x 35nm
- **Volume Size:** 1250 x 1250 x 125
- **Annotation Completion Dates:** 2023.12.11 (w4), 2024.04.12 (w36)
- **Authors:** Shi Te, Guo Jun, Yin Chunying, Zhang Ruobing
- **Copyright Holder:** Institute of Artificial Intelligence, Hefei Comprehensive National Science Center

### **Acknowledgment Norms**

- **Chinese Name:** 合肥综合性国家科学中心人工智能研究院
- **English Name:** Institute of Artificial Intelligence, Hefei Comprehensive National Science Center

</details>


## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{chen2023self,
  title={Self-supervised neuron segmentation with multi-agent reinforcement learning},
  author={Chen, Yinda and Huang, Wei and Zhou, Shenglong and Chen, Qi and Xiong, Zhiwei},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={609--617},
  year={2023}
}
```
# Contact 
If you need any help or are looking for cooperation feel free to contact us. cyd0806@mail.ustc.edu.cn
