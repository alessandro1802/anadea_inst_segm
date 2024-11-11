# Anadea home task: Instance segmentation

## Set-up
### Virtual environment
```shell
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
# Data obtainment
The [LIVECell dataset](https://sartorius-research.github.io/LIVECell/) contains a high-quality, manually annotated and expert-validated cell images that is the largest of its kind to date, consisting of over 1.6 million cells from a diverse set of cell morphologies and culture densities.
```shell
mkdir -p data/LIVECell_dataset_2021/annotations
aws s3 sync s3://livecell-dataset/LIVECell_dataset_2021/annotations/ ./data/LIVECell_dataset_2021
aws s3api get-object --bucket livecell-dataset  --key LIVECell_dataset_2021/images.zip ./data/LIVECell_dataset_2021/images.zip
unzip data/LIVECell_dataset_2021/images.zip
```

## File-structure
**Note**: Run EDA before training to merge train and val annotations.

```shell        
.
├── data
│   ├── LIVECell_dataset_2021/       # The dataset
├── EDA.ipynb                        # Exploratory data analysis
├── train.ipynb                      # Training and evaluation pipe-lines
├── model.py                         # Model definition
├── inference.ipynb                  # Inference example notebook
├── lightning_logs/                  # Training logs
├── models                           #
│   ├── MobileNetV2.pkl              # Serialized model
│   ├── transform.pkl                # Serialized pre-processor
│   └── checkpoints/                 # Model weights
├── requirements-macos.txt           #
├── requirements.txt                 #
└── README.md                        #
```
