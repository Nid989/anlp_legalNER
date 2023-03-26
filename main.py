import os
import gc
from pathlib import Path
import torch
from models.modeling_xlm_roberta import XLMRobertaCRFforTokenClassification, XLMRobertaforTokenClassification
from models.modeling_bert import InLegalBERTForTokenClassification
from models.ensemble import Ensembler
from data_utils import InLegalNERDataset

from helpers import config_data, save_results
from model_utils import train, generate_results
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    if not config_data["USE_ENSEMBLE"]:
        # version 1: train model using judgement data and standard CrossEntropy loss and BIO tagging
        # version 2: train model using judgement data and weighted CrossEntropy loss and BIO tagging 
        # version 3: train model using judgement data and standard CrossEntropy loss and BIOES tagging
        # version 4: train model using preamble data and standard CrossEntropy loss
        # version 5: # PENDING mention or not
        # version 6: train model using judgement data and Conditional Random Fields and BIO tagging
        # version 7: train model using judgement data and Conditional Random Fields and BIOES tagging

        model_checkpoint = config_data["MODEL_CHECKPOINT"]
        if "xlm-roberta" in model_checkpoint:
            TOKENIZER = AutoTokenizer.from_pretrained(model_checkpoint,
                                                    add_prefix_space=True) # specific to XLM-Roberta
            dataset = InLegalNERDataset(tokenizer=TOKENIZER) # initialize InLegalNERDataset
            print("Loading model...")
            if config_data["USE_CRF"]:
                MODEL = XLMRobertaCRFforTokenClassification.from_pretrained(model_checkpoint, num_labels=dataset.num_labels)
            else:
                MODEL = XLMRobertaforTokenClassification.from_pretrained(model_checkpoint, num_labels=dataset.num_labels)        
        elif "InLegalBERT" in model_checkpoint:
            TOKENIZER = AutoTokenizer.from_pretrained(model_checkpoint)
            dataset = InLegalNERDataset(tokenizer=TOKENIZER)
            print("Loading model...")
            MODEL = InLegalBERTForTokenClassification.from_pretrained(model_checkpoint, num_labels=dataset.num_labels)
        else: 
            raise ValueError(f"Define appropriate model checkpoint; found {model_checkpoint}; \
                            should be either [`xlm-roberta`, `InLegalBERT`] or its a finetuned version with definitive path!")
        
        MODEL.to(DEVICE)

        gc.collect()
        
        pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
        print(f"Total parameters: {pytorch_total_params}")
        pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {pytorch_total_train_params}")

        # --------------------------- TRAINING SETUP --------------------------- # 
        train(model=MODEL,
            dataset=dataset)
        
        # --------------------------- RESULTS --------------------------- # 
        results = generate_results(model=MODEL, dataset=dataset)
        # print(results)

        path_to_file=os.path.join(str(Path(config_data['PATH_TO_RESULT_OUTPUT_DIR']).parent), 
                                config_data['MODEL_CHECKPOINT'].split("/")[-1] + "-finetuned-for-token-classification-" + config_data['VERSION'] + ".csv")
        save_results(path_to_file=path_to_file, results=results)
        print("saved results at {}".format(path_to_file))

    else: # (limited usuage) using ensemble for xlm-roberta and its derivatives only
        TOKENIZER = AutoTokenizer.from_pretrained(config_data['ENSEMBLE_TOKENIZER'],
                                                add_prefix_space=True) # specific to XLM-Roberta
        dataset = InLegalNERDataset(tokenizer=TOKENIZER) # initialize InLegalNERDataset
        print("Loading model...")

        # generate results by using a ensembling procedure on already trained models
        MODEL = Ensembler(models_dict=config_data['ENSEMBLE_MODELS'], 
                        ens_type=config_data['ENSEMBLING_TYPE'], 
                        start_idx=0, end_idx=2, pad_idx=1)
        
        # --------------------------- RESULTS --------------------------- # 
        results = generate_results(model=MODEL, dataset=dataset)
        print(results)

        path_to_file=os.path.join(str(Path(config_data['PATH_TO_RESULT_OUTPUT_DIR']).parent), 
                                "xlm-roberta-base-finetuned-for-token-classification-ensemble-v3-v7.csv")
        save_results(path_to_file=path_to_file, results=results)
        print("saved results at {}".format(path_to_file))
        
