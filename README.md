# anlp_legalNER

The repository contains the code for team VASPOM for ANLP final project @ Universität Potsdam. It contains the working code for the task of SemEval 2023; task 6 Subtask B: LegalNER. Given the two dataset type under the LegalNER task, which includes, i) Judgement, and ii) Preamble, we propose different startegies to compile the given data and train domain-specific NER models that can perform the sequence tagging procedure optimally. The various model configurations are stated under the table below. More information related to the task can be found [here](https://sites.google.com/view/legaleval/home)

# Task Data 

The below link provide direct access to the task the task dataset. Note. due to unavailibility of ground-truth labels for test-data, we're not using it for experimentation and result analysis.

- [Train Dataset](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_TRAIN.zip)
- [Validation Dataset](https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_TRAIN.zip)

# Training and Testing Setup(Linux SetUp)

1. Clone the repository

```git clone [git clone https URL]```

2. Create a Python virtual environment

```
# Update and upgrade
sudo apt update
sudo apt -y upgrade

# check for python version "ideal: 3.8.2"
python3 -V

# install python3-pip
sudo apt install -y python3-pip

# install-venv
sudo apt install -y python3-venv

# Create virtual environment
python3 -m venv my_env


# Activate virtual environment
source my_env/bin/activate
```

3. Install project dependent files

```
pip install requirements.txt
```

4. Install spacy's `en_core_web_sm` model (Note, used for pre-processing purpose)

```
python3 -m spacy download en_core_web_lg
```

5. Run main.py

```
python3 main.py
```

# Config.yaml

### Hyperparameters and settings for training a Named Entity Recognition (NER) model on InLegal dataset

### Training settings
batch_size: positive integer, e.g. `2`
max_epochs: positive integer, e.g. `20`
max_sequence_len: positive integer, e.g. `256`
learning_rate: positive float, e.g. `1e-4`
weight_decay: positive float, e.g. `1e-6`
early_stopping_threshold: positive integer, e.g. `5`

### Data settings
- source_column: `tokens`
- target_column: `BIO_tags` or `BIOES_tags`
- extract_form: `judgement`, `preamble` or `combined`
- path_to_data_dir: path to directory containing input data files
- path_to_class_labels: path to file containing class labels used for training the model

### Model settings
- model_checkpoint: `distilbert-base-uncased`, `law-ai/InLegalBERT`, `xlm-roberta-base`, or `roberta-base`
- version: `v1` to `v8`
- use_crf: True or False
- use_ensemble: True or False
- ensemble_tokenizer: `distilbert-base-uncased`, `law-ai/InLegalBERT`, `xlm-roberta-base`, or `roberta-base`
- ensemble_models:
    * base-model 1
        * type: `XLMRobertaforTokenClassification` or `XLMRobertaCRFforTokenClassification`
        * version: model version
        * source_column: `tokens`
        * target_column: `BIO_tags` or `BIOES_tags`
        * use_crf: True or False
        * num_labels: number of labels
        * path_to_model_file: path to saved model file
    * base-model 2
        * type: `XLMRobertaforTokenClassification` or `XLMRobertaCRFforTokenClassification`
        * version: model version
        * source_column: `tokens`
        * target_column: `BIO_tags` or `BIOES_tags`
        * use_crf: True or False
        * num_labels: number of labels
        * path_to_model_file: path to saved model file
- ensmbling_type: `max` or `soft`
- path_to_result_output_dir: path to directory where the results of the training and evaluation will be saved
- path_to_model_output_dir: path to directory where the trained models will be saved

# Project Directory Tree

```
├── data/
│   ├── NER_DEV/
│   │   ├── NER_DEV_JUDGEMENT.json
│   │   ├── NER_DEV_PREAMBLE.json
│   │   └── NER_DEV_COMBINED.json
│   ├── NER_TRAIN/
│   │   ├── NER_TRAIN_JUDGEMENT.json
│   │   ├── NER_TRAIN_PREAMBLE.json
│   │   └── NER_TRAIN_COMBINED.json
│   ├── combined_class_labels.pkl
│   ├── judgement_class_labels.pkl
│   └── preamble_class_labels.pkl
├── models/
│   ├── CRF.py
│   ├── ensemble.py
│   ├── modeling_bert.py
│   └── modeling_xlm_roberta.py
├── model_checkpoints/
├── results/
│   ├── val
│   ├── confusion_matrix
│   └── final_results_dev
├── config.yaml
├── helpers.py
├── data_utils.py
├── model_utils.py
├── main.py
└── .gitignore
```


# Model Configurations

## Trainable models

| Version | Encoder Type | Extract Form | Tagging Type | Training Method | Model Checkpoints |
|---------|--------------|--------------|--------------|----------------|--------------------|
| v1      | InLegalBERT  | Judgement    | BIO          | normal CrossEntropy | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v2      | InLegalBERT  | Judgement    | BIOES        | weighed CrossEntropy | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v3      | InLegalBERT  | Judgement    | BIOES        | normal CrossEntropy | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v4      | InLegalBERT  | Preamble     | BIOES        | normal CrossEntropy | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v1      | XLM-RoBERTa  | Judgement    | BIO          | normal CrossEntropy | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v2      | XLM-RoBERTa  | Judgement    | BIOES        | weighted CrossEntropy | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v3      | XLM-RoBERTa  | Judgement    | BIOES        | normal CrossEntropy | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v6      | XLM-RoBERTa  | Judgement    | BIO          | linear-chain Conditional Random Field (CRF) | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v7      | XLM-RoBERTa  | Judgement    | BIOES        | linear-chain Conditional Random Field (CRF) | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |
| v8      | XLM-RoBERTa  | Judgement + Preamble | BIOES | linear-chain Conditional Random Field (CRF) | [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing) |


## Ensemble models
| Voting Strategy | Model 1 Encoder Type | Model 1 Extract Form | Model 1 Tagging Type | Model 1 Training Method | Model 2 Encoder Type | Model 2 Extract Form | Model 2 Tagging Type | Model 2 Training Method |
|-----------------|---------------------|----------------------|----------------------|------------------------|---------------------|----------------------|----------------------|------------------------|
| soft            | XLM-R               | Judgement            | BIOES               | normal CrossEntropy loss   | XLM-R               | Judgement            | BIOES               | linear chain Conditional Random Field (CRF)           |
| max             | XLM-R               | Judgement            | BIOES               | normal CrossEntropy loss   | XLM-R               | Judgement            | BIOES               | linear chain Conditional Random Field (CRF)           |



We provide access to our already trained/finetuned models [here](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing). Every folder includes model checkpoints that correspond to a particular model setup and should be utilized with the relevant target labels

# Acknowledgement

We used code for the CRF part of our project that was based on a medium article available [here](https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea). The article explains how to implement a linear-chain Conditional Random Field using PyTorch.

We also Inherit the trasnformer model's tokenclassification class attributes for singular processing and save and load functionality. This includes the following classes found [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py) and [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py).

