# finetuning_esm2_public
Setting up generalizable repository to finetune esm2 models
---
#### Files

finetuning_esm2.py --> Python script to finetune esm2

ESM2_w_regression_MLP_heads.py --> dataloader and model architexture loaded in finetuning_esm2.py

functions.py --> contains loaded functions important for finetuning esm2

datasets --> csv and pkl files without split in the filename

pkl files with splits in the filename --> these are fixed dataplits for datasets to avoid test leakage
