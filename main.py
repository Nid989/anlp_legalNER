import gc
import torch
from models.modeling_xlm_roberta import XLMRobertaCRFforTokenClassification, XLMRobertaForTokenClassification
from data_utils import InLegalNERDataset

from helpers import config_data
from model_utils import train
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    model_checkpoint = config_data["MODEL_CHECKPOINT"]
    TOKENIZER = AutoTokenizer.from_pretrained(model_checkpoint,
                                              add_prefix_space=True) # specific to XLM-Roberta
    dataset = InLegalNERDataset(device=DEVICE, tokenizer=TOKENIZER) # Initialize InLegalNERDataset

    print("Loading model...")
    MODEL = XLMRobertaCRFforTokenClassification.from_pretrained(model_checkpoint, num_labels=dataset.num_labels)
    MODEL.to(DEVICE)

    gc.collect()
    
    pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    print(f"Total parameters: {pytorch_total_params}")
    pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {pytorch_total_train_params}")

    # --------------------------- TRAINING SETUP --------------------------- # 

    train(model=MODEL,
          tokenizer=TOKENIZER,
          dataset=dataset)