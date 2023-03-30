# NOTE; this class should not be trained and is used only for prediction purpose or 
# to derive results via accumulated ensembling approach 

import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from models.modeling_xlm_roberta import XLMRobertaCRFforTokenClassification, XLMRobertaforTokenClassification

# usually defined via a dictionary, consisting of systematic declaration of 
# attributes including `model_type`, `model_version`(indirect), `model_path`,
# `if model use_crf?`, and `num_labels` (model specific). 

class Ensembler(nn.Module):
    # should not be trained
    def __init__(self,
                 models_dict: dict,
                 ens_type: str="soft",
                 **kwargs):
        super(Ensembler, self).__init__()

        self.models_dict = models_dict # just for safety
        self.ens_type = ens_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.models, self.crf_flags = self.load_models()

        if kwargs:
            self.start_idx = kwargs["start_idx"]
            self.end_idx = kwargs["end_idx"]
            self.pad_idx = kwargs["pad_idx"]

    def load_models(self):
        models_list = list(self.models_dict.keys())

        assert len(set([self.models_dict[model_name]['TARGET_COLUMN'] for model_name in models_list])) == 1; "TARGET_COLUMN mismatch, should be equal"

        models_type = [self.models_dict[model_name]['TYPE'] for model_name in models_list]
        models_path = [self.models_dict[model_name]['PATH'] for model_name in models_list]
        models_num_labels = [self.models_dict[model_name]['NUM_LABELS'] for model_name in models_list]
        crf_flags = [self.models_dict[model_name]['USE_CRF'] for model_name in models_list]

        models = []
        for model_type, model_path, num_labels in zip(models_type, models_path, models_num_labels):
            if model_type == "XLMRobertaforTokenClassification":
                model = XLMRobertaforTokenClassification.from_pretrained(model_path, num_labels=num_labels)
                model.to(self.device)
                models.append(model)
            elif model_type == "XLMRobertaCRFforTokenClassification":
                model = XLMRobertaCRFforTokenClassification.from_pretrained(model_path, num_labels=num_labels)
                model.to(self.device)
                models.append(model)
            else:
                raise ValueError("define appropriate model_type's; found {model_type}; \
                should be either [`XLMRobertaforTokenClassification`, `XLMRobertaCRFforTokenClassification`]")
        
        return models, crf_flags

    def align_crf_logits(self, 
                         logits: torch.Tensor):
        # transform xlmrcrf logits to match with xlmr logits [`depricate <s> </s> <pad> label values`]
        deprication_indices = [self.start_idx, self.end_idx, self.pad_idx]
        keep_mask = torch.ones(logits.shape[2], dtype=torch.bool).to(self.device)
        keep_mask[deprication_indices] = 0
        logits = torch.index_select(logits, 2, keep_mask.nonzero().squeeze())
        return logits

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        ensemble_logits = []
        for model, crf_flag in zip(self.models, self.crf_flags):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            ensemble_logits.append(self.align_crf_logits(outputs.logits) if crf_flag else outputs.logits)
        assert len(set([logits.shape[2] for logits in ensemble_logits])) == 1; "dimensionality mismatch!"

        if self.ens_type == "soft":
            ensemble_logits = torch.sum(torch.stack(ensemble_logits, dim=-1), dim=-1) / len(self.models)
        elif self.ens_type == "max": # max-voting
            ensemble_logits, _ = torch.mode(torch.stack(ensemble_logits, dim=-1), dim=-1)
        else:
            raise ValueError(f"define appropriate ens_type; found {self.ens_type}; should be either [`sum`, `vote`]")
        
        return TokenClassifierOutput(
            loss=None,
            logits=ensemble_logits,
            hidden_states=None,
            attentions=None
        )