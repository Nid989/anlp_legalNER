# anlp_legalNER
This repository contains the code for team VASPOM for ANLP final project @ Universität Potsdam

----

`v1` - Regular Finetunining with BIO tagging on judgement data (XLMR, InLegalBERT)

`v2` - Regular Finetuning with weighted CrossEntropyLoss on judgement data (XLMR, InLegalBERT)

`v3` - Regular Finetuning with BIOES tagging on judgement data (XLMR, InLegalBERT)

`v4` - Regualr Finetuning on preamble data (InLegalBERT)

`v5` - Dual Finetuning, preamble and judgement data NOT MENTIONING

`v6` - Regular Finetuning extended with CRF (Conditional Random Fields) with BIO tagging (XLMR)

`v7` - Regular Finetuning extended with CRF (Conditional Random Fields) with BIOES tagging (XLMR)

`v8` - Regular Finetuning extended with CRF (Conditional Random Fields) with BIOES tagging on Combined dataset (XLMR)

`v3-v7` - Ensemble of v3 and v7 using max-voting (XLMR)

----

[Link to project directory](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing)

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
