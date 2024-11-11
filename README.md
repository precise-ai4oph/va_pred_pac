# Fundus Image-based Visual Acuity Assessment with PAC-Guarantees

## Description
This repository contains the code for the paper, "Fundus Image-based Visual Acuity Assessment with PAC-Guarantees"

## How to Run
The code consists of two part: training and exporting

### Training
The training is done by train_fundus_pd.py

```
python train_fundus_pd.py --model ressnet18-pd
```

The major command line arguments are as follows.

|Name|Description|
|----|-----------|
|--dataset-root|Root Directory of dataset|
|--model|Model Name|
|--split-fn|Dataset Split file name|
|--split-idx|Specifies which split is used|
|--train-batch-size|Training batch size|
|--test-batch-size|Test batch size|
|--resampling|Whether resampling is used based on class frequency|
|--pretrain|Type of pretrained weights|
|--pretrain-model-fn|Pretrained weights file name|
|--train-last-layer-only|Train last layer only or all layers|
|--output-dir|Output file location|

### Export
The example command is as follows.
```
python export_pd_output.py --name efficientnetv2s --fn outpus/fundus_model_pd_best_seed100.pt --store-result
```

The command line arguments are as follows.

|Name|Description|
|----|-----------|
|--name|Name for this experiment|
|--fn|Stored model file|
|--blurred|Whether blurring is used for test data or not|
|--kernel-size|Kernel size for blur|
|--store-result|Whether the output is stored or not|
|--output-dir|Output data location

