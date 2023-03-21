import os
import gc
import pandas as pd
import itertools
from tqdm.auto import tqdm
import spacy
from datetime import datetime
from helpers import read_data, load_from_pickle
from helpers import extract_forms_, data_types_
from helpers import config_data
import torch
from torch.utils.data import DataLoader, TensorDataset

nlp = spacy.load('./en_core_web_sm') # load en_core_web_sm model 
tqdm.pandas()

class InLegalNERDataset:
    def __init__(self, device, tokenizer):
        
        self.batch_size = config_data["BATCH_SIZE"]
        self.max_seq_len = config_data["MAX_SEQUENCE_LEN"]
        self.device = device
        self.tokenizer = tokenizer
        self.source_column = config_data["SOURCE_COLUMN"]
        self.target_column = config_data["TARGET_COLUMN"]
        self.data_dir = config_data["PATH_TO_DATA_DIR"]
        self.extract_form = config_data["EXTRACT_FORM"]
        self.class_labels = load_from_pickle(path_to_file=config_data["PATH_TO_CLASS_LABELS"])
        self.use_crf = config_data["USE_CRF"]
        self.ner_label_encodings = self.prepare_NER_label_mapping()
        self.ner_labels_list = list(self.ner_label_encodings)
        self.num_labels = len(self.ner_labels_list)

    def prepare_NER_label_mapping(self):
        """
        prepare NER_label_encoding_dict, using original class-labels
        """
        if self.target_column == "BIOES_tags":
            NER_labels = list(itertools.chain(*[[f"{tag_id}-{classname}" for tag_id in ["B", "I", "E", "S"]] for classname in self.class_labels]))
            NER_labels.append("O")
        elif self.target_column == "BIO_tags":
            NER_labels = list(itertools.chain(*[[f"{tag_id}-{classname}" for tag_id in ["B", "I"]] for classname in self.class_labels]))
            NER_labels.append("O")
        else:
            raise ValueError("define appropriate target column type; either [`BIOES_tag`, `BIO_tags`]")
        NER_labels = ['<s>', '<pad>', '</s>'] + NER_labels if self.use_crf else NER_labels 
        NER_label_encodings = {ner_label: index for index, ner_label in enumerate(NER_labels)}
        return NER_label_encodings

    def prepare_dataset(self, 
                        data_type: data_types_="train", 
                        **kwargs):
        def convert_char_based_to_token_based(char_based_data):
            # Split token into tokens and create list of token dictionaries
            token_based_data = []
            doc = nlp(char_based_data["data"]["text"])
            for token in doc:
                token_based_data.append({
                    "start": token.idx,
                    "end": token.idx + len(token),
                    "text": token.text,
                    "labels": []
                })
            
            for entity in char_based_data["annotations"][0]["result"]:
                # Assign BIO labels to tokens
                label = entity["value"]
                start_idx = label["start"]
                end_index = label["end"]
                label_text = label["text"]
                label_type = label["labels"][0]

                for i, token in enumerate(token_based_data):
                    if token["start"] <= start_idx < token["end"]:
                        # Token start within the label
                        if token["end"] >= end_index:
                            # Token also end within the label
                            token_based_data[i]["labels"].append("B-" + label_type)
                        else:
                            # Token continues beyond label
                            token_based_data[i]["labels"].append("B-" + label_type)
                        found_match = True
                    elif start_idx <= token["start"] < end_index:
                        # Token starts after the label starts but before the label ends
                        if end_index  > token["end"]:
                            # Token continues beyond the labek
                            token_based_data[i]["labels"].append("I-" + label_type)
                        else:
                            # Token ends within label
                            token_based_data[i]["labels"].append("I-" + label_type)
                        found_match = True
                    elif token["start"] >= end_index:
                        # Token starts after label ends
                        found_match = True
                        break
                if not found_match:
                    # No matching token found 
                    raise ValueError("Label '{}', with indices {}-{} has no matching token".format(label_text, start_idx, end_index))

            for token in token_based_data:
                if not token["labels"]:
                    token["labels"].append("O")

            return token_based_data

        def convert_BIO_to_BIOES_tags(example):
            BIOES_tag_list = []
            previous_tag = None
            previous_entity_type = None
            for tag in example:
                if tag == "O":
                    if previous_tag is not None:
                        if previous_tag == "B":
                            BIOES_tag_list.pop()
                            BIOES_tag_list.append("S" + "-" + previous_entity_type)
                            BIOES_tag_list.append(tag)
                        elif previous_tag == "I":
                            BIOES_tag_list.pop()
                            BIOES_tag_list.append("E" + "-" + previous_entity_type)
                            BIOES_tag_list.append(tag)
                        else:
                            BIOES_tag_list.append(tag)
                    else:
                        BIOES_tag_list.append(tag)
                else:
                    tag, current_entity_type = tag.split("-")
                    if tag == "B":
                        if previous_tag == "B":
                            BIOES_tag_list.pop()
                            BIOES_tag_list.append("S" + "-" + previous_entity_type)
                            BIOES_tag_list.append(tag + "-" + current_entity_type)
                        else:
                            BIOES_tag_list.append(tag + "-" + current_entity_type)
                    elif tag == "I":
                        BIOES_tag_list.append(tag + "-" + current_entity_type)
                    previous_entity_type = current_entity_type
                previous_tag = tag
            return BIOES_tag_list

        def tabularize_data(token_based_data):
            tokens = []
            labels = []
            for token in token_based_data:
                tokens.append(token["text"])
                labels.append(token["labels"][0])
            return tokens, labels

        def apply_data_reformation(data):
            tokens, ner_tags = [], []
            for example in tqdm(data, total=len(data)):
                tokens_, ner_tags_ = tabularize_data(convert_char_based_to_token_based(example))
                tokens.append(tokens_)
                ner_tags.append(ner_tags_)

            df = pd.DataFrame({"tokens": tokens, "BIO_tags": ner_tags})
            return df

        df = apply_data_reformation(read_data(self.data_dir,
                                              data_type=data_type,
                                              extract_form=self.extract_form))
        
        # apply BIOES tagging
        df["BIOES_tags"] = df.progress_apply(lambda row: convert_BIO_to_BIOES_tags(row["BIO_tags"]), axis=1)

        del data_type
        gc.collect()

        return df.sample(n=48)

    def preprocess_dataset(self,
                           dataset: pd.DataFrame):
    
        # is_split_into_words; specifies whether the input sentence tobe tokenized is provides as single string or else as list of tokens/words!
        source = [s for s in dataset[self.source_column].values.tolist()]
        model_inputs = self.tokenizer(source,
                                max_length=self.max_seq_len,
                                padding="max_length",
                                truncation=True,
                                is_split_into_words=True)
        
        
        def synchronize_labels(dataset, model_inputs):
            """
            synchronize labels w.r.t tokenized model inputs
            """
            label_all_tokens = True
            NER_labels = []
            NER_labels_mask = []
            for index, label in tqdm(enumerate(dataset[self.target_column].values.tolist()), total=dataset.shape[0]):
                word_ids = model_inputs.word_ids(batch_index=index)
                previous_word_idx = None
                label_ids = []
                label_masks = []
                for index2, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        if self.use_crf and (index2==0 or word_ids[index2-1] is not None):
                            label_ids.append(self.ner_label_encodings['<s>'] if index2==0 else self.ner_label_encodings['</s>'])
                            label_masks.append(1)
                        else:
                            label_id = self.ner_label_encodings['<pad>'] if self.use_crf else -100
                            label_ids.append(label_id)
                            label_masks.append(0)
                    elif label[word_idx] == 0:
                        label_ids.append(0)
                        label_masks.append(1)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.ner_label_encodings[label[word_idx]])
                        label_masks.append(1)
                    else:
                        if self.use_crf:
                            label_ids.append(self.ner_label_encodings[label[word_idx]] if label_all_tokens else self.ner_label_encodings['<pad>'])
                            label_masks.append(1 if label_all_tokens else 0)
                        else:
                            label_ids.append(self.ner_label_encodings[label[word_idx]] if label_all_tokens else -100)
                            label_masks.append(1 if label_all_tokens else 0)
                
                    previous_word_idx = word_idx
                NER_labels.append(label_ids)
                NER_labels_mask.append(label_masks)

            return NER_labels, NER_labels_mask
        
        labels, labels_mask = synchronize_labels(dataset, model_inputs)
        model_inputs["labels"] = labels
        model_inputs["mask"] = labels_mask
        model_inputs["input_ids"] = torch.tensor([i for i in model_inputs["input_ids"]], dtype=torch.long, device=self.device) 
        model_inputs["attention_mask"] = torch.tensor([i for i in model_inputs["attention_mask"]], dtype=torch.long, device=self.device)
        model_inputs["labels"] = torch.tensor([i for i in model_inputs["labels"]], dtype=torch.long, device=self.device)
        model_inputs["mask"] = torch.tensor([i for i in model_inputs["mask"]], dtype=torch.long, device=self.device)
        
        del dataset
        del source
        del labels
        del labels_mask
        gc.collect()
        return model_inputs
    
    def set_up_data_loader(self, data_type: data_types_="train"):
        dataset = self.prepare_dataset(data_type) 
        dataset = self.preprocess_dataset(dataset=dataset)
        dataset = TensorDataset(dataset["input_ids"],
                                dataset["attention_mask"],
                                dataset["labels"],
                                dataset["mask"])
                                
        gc.collect()
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
    