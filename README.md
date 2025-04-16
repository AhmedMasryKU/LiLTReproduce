# LiLT Reproducibility Study
This repository contains the codebase for our reproducibility study of LiLT (Language-independent Layout Transformer), originally proposed in the paper ["LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding"](https://arxiv.org/abs/2202.13669).

**Project Members:** Ahmed Masry, Yahia Ahmed

## Overview

LiLT introduces a two-encoder architecture for structured document understanding by **decoupling layout and text modalities** during pretraining. This enables seamless **multilingual transfer** by reusing the layout encoder weights for different language text encoders.

Our study investigates two core claims:
1. **BiACM (Bi-directional Attention Complementation Mechanism)** improves text-layout integration.
2. **DETACH** operation boosts multilingual generalization by preventing layout gradients from influencing the text encoder during pretraining.

## Repository Structure
```
LiLTReproduce/
├── configs/
│   ├── lilt_custom_config/
│   ├── two_tower_config/
│   └── two_tower_multilingual_config/
├── create_multilingual_checkpoint.py
├── data/
│   ├── __init__.py
│   ├── preprocessing/
│   └── pretraining/
├── fine-tuning_ablation_study.ipynb
├── fine-tuning_hugging_face.ipynb
├── model_checkpoints_scripts/
│   ├── create_lilt_custom.py
│   └── create_two_tower_lilt.py
├── models/
│   ├── lilt_model.py
│   ├── lilt_model_detach.py
│   ├── lilt_without_biacm.py
│   ├── lilt_for_masked_lm.py
│   ├── lilt_for_token_classification.py
│   ├── lilt_two_tower_model.py
│   ├── lilt_two_tower_for_masked_lm.py
│   └── lilt_two_tower_for_token_classification.py
├── pretrain_lilt_mlm.py
├── pretrain_lilt_detach_mlm.py
├── pretrain_lilt_two_tower_mlm.py
├── push_model_to_hf.py
├── scripts/
│   ├── pretrain_lilt_mlm.sh
│   ├── pretrain_lilt_detach_mlm.sh
│   ├── pretrain_lilt_mlm_two_tower.sh
│   └── process_idl_data.sh
```

# Pretraining 
Check out our pretraining scripts in the "scripts" folder. In order to run them, you will need access to Compute Canada resources. Howver, you can also easily adapt them to your computing cluster!
We have added a requirements.txt file to install the required dependencies in your conda or virtual environment. 

# Finetuning 
You can run onr finetuning notebooks: fine-tuning_hugging_face and fine-tuning_ablation_study on colab. 
But you also need access to the datasets which we shae in this google drive folder: https://drive.google.com/drive/folders/1nbHe-kmNmh7gSfTaLLqx7AzzyNNFxu0U
