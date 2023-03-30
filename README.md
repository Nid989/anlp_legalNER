# anlp_legalNER
This repository contains the code for team VASPOM for ANLP final project @ Universität Potsdam

----

`v1` - Regular Finetunining with BIO tagging on judgement data

`v2` - Regular Finetuning with weighted CrossEntropyLoss on judgement data

`v3` - Regular Finetuning with BIOES tagging on judgement data

`v4` - Regualr Finetuning on preamble data

`v5` - Dual Finetuning, preamble and judgement data

`v6` - Regular Finetuning extended with CRF (Conditional Random Fields) with BIO tagging

`v7` - Regular Finetuning extended with CRF (Conditional Random Fields) with BIOES tagging

`v8` - Regular Finetuning extended with CRF (Conditional Random Fields) with BIOES tagging on Combined dataset

`v3-v7` - Ensemble of v3 and v7 using max-voting
----

[Link to project directory](https://drive.google.com/drive/folders/1EPzM7d3qtORmIZubnmnn8o_BP5ceDCWW?usp=sharing)


# Docker setup

1. To build an image and install dependencies, run command `docker build -t legalner .` in terminal.
2. To run an image as a container, run command `docker run legalner`
