## Overview

Incorporated LaCLIP model into the multi-view architecture of the paper ["MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System"](https://arxiv.org/pdf/2307.07135). The objective
is to leverage enhanced ability of LaCLIP as a result of language augmented pre-training to improve Multi-modal Sarcasm Detection.

## Dataset

Download the image data from [1](https://drive.google.com/file/d/1mK0Nf-jv_h2bgHUCRM4_EsdTiiitZ_Uj/view?usp=sharing_eil&ts=5d480e04) [2](https://drive.google.com/file/d/1AOWzlOz5hmdO39dEmzhQ4z_nabgzi7Tu/view?usp=sharing_eil&ts=5d480e04) [3](https://drive.google.com/file/d/1dJERrVlp7DlNSXk-uvbbG6Rv7uvqTOKd/view?usp=sharing_eil&ts=5d480e04) 
[4](https://drive.google.com/file/d/1pODuKC4gP6-QDQonG8XTqI8w8ds68mE3/view?usp=sharing_eil&ts=5d480e04), 
and unzip them into this folder (`data/dataset_image`)

MMSD 2.0 dataset is in the folder `data/text_json_final`.

## Installation
```
conda create -n env_name python=3.9  
source activate env_name     
pip install -r requirements.txt
```

## Train

### MMSD2.0 dataset on LACLIP 

```
python3 main.py --model LACLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > LA_CLIP_MMSD2.log 2>&1 &
```
Note: LaCLIP model is pretrained on LAION dataset.

## Acknowledgement

This code is adapted from [this](https://github.com/JoeYing1019/MMSD2.0) codebase.



