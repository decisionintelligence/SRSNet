

# Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective 

**This code is the official PyTorch implementation of our NIPS'25 paper: [Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective](https://arxiv.org/pdf/2510.14510).**

[![NeurIPS](https://img.shields.io/badge/NeurIPS'25-SRSNet-orange)](https://arxiv.org/pdf/2510.14510)  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)  ![Stars](https://img.shields.io/github/stars/decisionintelligence/SRSNet) 

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!

🚩 News (2025.9) Our paper has been accepted as a **Spotlight** poster in NeurIPS 2025.

🚩 News (2025.11) Our paper is unfairly desk rejected by the SPC of NeurIPS 2025 without any evidence provided. We sincerely remind all contributors to carefully check the **"Dual Submission"** policies of top AI conferences and pay attention to such subjective descriptions as **"substantial similarity"** and **"thinly slicing"**. Under such subjective criterions, if your papers (shared authors) focus on the same topic and are parallelly submitted, they are at risk of being judged as dual submissions by some Program Chairs not majoring in your domains.

🚩 News (2025.12) 🎉🎉 We successfully appealed to the NIPS board's Ethics & Grievances Committee, which rejected the decision of the SPC and restored our article to the accepted status.

## Introduction

In this paper, we pioneer the exploration of constructing a selective representation space to flexibly include the information beneficial for forecasting. Specifically, we propose the **Selective Representation Space (SRS)** module, which utilizes the learnable Selective Patching and Dynamic Reassembly techniques to adaptively select and shuffle the patches from the contextual time series, aiming at fully exploiting the information of contextual time series to enhance the forecasting performance of patch-based models. To demonstrate the effectiveness of SRS module, we propose a simple yet effective **SRSNet** consisting of SRS and an MLP head, which achieves state-of-the-art performance on real-world datasets from multiple domains. 

<div align="center">
<img alt="Logo" src="figures/overview.png" width="100%"/>
</div>

The important components of the **SRS** Module: (1) Selective Patching; (2) Dynamic Reassembly ; (3) Adaptive Fusion
<div align="center">
<img alt="Logo" src="figures/architecture.png" width="100%"/>
</div>


## Quickstart

> [!IMPORTANT]
> this project is fully tested under python 3.8, it is recommended that you set the Python version to 3.8.
1. Requirements

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

2. Data preparation

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link). Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model

- To see the model structure of **SRSNet**,  [click here](./ts_benchmark/baselines/srsnet/models/srsnet_model.py).
- We provide all the experiment scripts for SRSNet and other baselines under the folder `./scripts/multivariate_forecast`.  For example you can reproduce all the experiment results as the following script:

```shell
sh ./scripts/multivariate_forecast/ETTh1_script/SRSNet.sh
```



## Results
Extensive experiments on  8 real-world datasets demonstrate that SRSNet achieves state-of-the-art~(SOTA) performance.

<div align="center">
<img alt="Logo" src="figures/exp.png" width="100%"/>
</div>


## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2025srsnet,
  title     = {Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective},
  author    = {Wu, Xingjian and Qiu, Xiangfei and Cheng, Hanyin and Li, Zhengyu and Hu, Jilin and Guo, Chenjuan and Yang, Bin},
  booktitle = {NeurIPS},
  year      = {2025}
}
```



## Contact

If you have any questions or suggestions, feel free to contact:

- [Xingjian Wu](https://ccloud0525.github.io/) ([xjwu@stu.ecnu.edu.cn](mailto:xjwu@stu.ecnu.edu.cn))
- [Xiangfei Qiu](https://qiu69.github.io/) ([xfqiu@stu.ecnu.edu.cn](mailto:xfqiu@stu.ecnu.edu.cn))

Or describe it in Issues.
