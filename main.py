import gc
import torch
from models.modeling_xlm_roberta import XLMRobertaCRFforTokenClassification, XLMRobertaForTokenClassification
from models.modeling_bert import InLegalBERTForTokenClassification
from data_utils import InLegalNERDataset

from helpers import config_data
from model_utils import train
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

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
            MODEL = XLMRobertaForTokenClassification.from_pretrained(model_checkpoint, num_labels=dataset.num_labels)
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

    # train(model=MODEL,
    #       tokenizer=TOKENIZER,
    #       dataset=dataset)