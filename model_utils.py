import os
import gc
import shutil
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import torch
from helpers import save_results
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.modeling_outputs import TokenClassifierOutput
from helpers import config_data, check_and_create_directory
from evaluate import load
seqeval = load("seqeval")

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import confusion_matrix

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

def get_scores(p, ner_labels_encoding, ner_labels_list, use_crf: bool=False, full_rep: bool=False):
    """
    Args:
        p: (tuple) includes lists of predictions [0] and lists of gold-labels [1]
        ner_labels_encodings: (dict) ner_labels to unique index mapping
        ner_labels_list: (list) unique ner_labels in the give dataset
        use_crf: (bool) specifies whether the `model` extends to 
            Conditional Random Fiels
        full_rep: specifies whether to return overall results including class-wise
            scores or not
    Returns:
        results: (dict); encompassing class-wise and overall scores for the following 
             `f1-score`, `precision`, `recall`, and `accuracy`
    """
    predictions, labels = p
    
    if use_crf:
        ignore_idx_list = [0, 2, 1] # <s>, </s>, <pad>
    else:
        ignore_idx_list = [-100] # pad token

    true_predictions = [
        [ner_labels_list[p] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [ner_labels_list[l] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=False)
    
    if full_rep:
        return results
    else:    
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


# ------------------------------------- TRAINING UTILS ------------------------------------- # 

def train_epoch(model,
                data_loader,
                optimizer,
                use_crf:bool=False,
                device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Args:
        model: PyTorch model, with a transformers based text-encoder
        data_loader: PyTorch DataLoader; with specific batch_size; usually
            depicting orchestration of training data
        optimizer: PyTorch optimizer; used to tune model weights during 
            back-propogation
        use_crf: (bool) specifies whether the `model` extends to 
            Conditional Random Fiels
    Returns:
        Average loss over training epoch 
    """
    model.train()
    epoch_train_loss = 0.0
    pbar = tqdm(data_loader, desc="Training Iteration")
    for step, batch in enumerate(pbar):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels, mask = batch
        optimizer.zero_grad()
        if use_crf:
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            mask=mask,
                            reduction='mean')
        else:
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        loss = outputs.loss
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        pbar.set_description('train_loss={0:.3f}'.format(loss.item()))

    del batch
    del input_ids
    del attention_mask
    del labels
    del mask 
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_train_loss / step

def val_epoch(model,
              data_loader,
              use_crf: bool=False,
              device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Args:
        model: PyTorch model, with a transformers based text-encoder
        data_loader: PyTorch DataLoader; with specific batch_size; usually
            depicting orchestration of validation data
        use_crf: (bool) specifies whether the `model` extends to 
            Conditional Random Fields
    Returns:
        Average loss over validation epoch 
    """
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Validation Loss Iteration")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, mask = batch
            if use_crf:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                mask=mask,
                                reduction='mean')
            else:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
            loss = outputs.loss
            epoch_val_loss += loss.item()

            pbar.set_description('val_loss={0:.3f}'.format(loss.item()))
    
    del batch
    del input_ids
    del attention_mask
    del labels
    del mask 
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_val_loss / step

def test_epoch(model,
               data_loader,
               desc,
               use_crf: bool=False,
               device = "cuda" if torch.cuda.is_available() else "cpu",
               **gen_kwargs):
    """
    Args:
        model: PyTorch model, with a transformers based text-encoder
        data_loader: PyTorch DataLoader; with specific batch_size; usually
            depicting orchestration of test data
        desc: whether applying testing procedure over validation or test dataset
        use_crf: (bool) specifies whether the `model` extends to 
            Conditional Random Fields
    Returns:
        Predictions over provided dataset orchestrated under the `data_loader`
            alongwith gold-standard labels
    """
    model.eval()
    out_predictions = []
    gold = []
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, mask = batch
            if use_crf:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_kwargs)
                predictions = outputs.predictions

                out_predictions.extend(predictions.cpu().tolist())
                gold.extend(labels.cpu().tolist())
            else:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_kwargs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                predictions = probabilities.argmax(dim=-1).cpu().tolist()

                out_predictions.extend(predictions)
                gold.extend(labels.cpu().tolist())
    
    del batch
    del input_ids
    del attention_mask
    del labels
    del mask
    del outputs
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    return out_predictions, gold

def get_val_scores(model,
                   data_loader,
                   desc,
                   epoch,
                   ner_labels_encoding:dict,
                   ner_labels_list: list,
                   use_crf: bool=False,
                   **gen_kwargs):
    """
    Args:
        model: PyTorch model, with a transformers based text-encoder
        data_loader: PyTorch DataLoader; with specific batch_size; usually
            depicting orchestration of test data
        desc: whether applying testing procedure over validation or test dataset
        epoch: Epoch status at which the results are computed
        ner_labels_encodings: (dict) ner_labels to unique index mapping
        ner_labels_list: (list) unique ner_labels in the give dataset
        use_crf: (bool) specifies whether the `model` extends to 
            Conditional Random Fields
    Returns:
        results: (dict); encompassing class-wise and overall scores for the following 
             `f1-score`, `precision`, `recall`, and `accuracy`
    """
    predictions, gold = test_epoch(model=model,
                                   data_loader=data_loader,
                                   desc=desc,
                                   use_crf=use_crf,
                                   **gen_kwargs)
    result = get_scores(p=(predictions, gold),
                        ner_labels_encoding=ner_labels_encoding,
                        ner_labels_list=ner_labels_list,
                        use_crf=use_crf) 
    
    model_checkpoint = config_data["MODEL_CHECKPOINT"]
    version = config_data["VERSION"]
    if "Validation" in desc:
        val_df = pd.DataFrame(list(zip(gold, predictions)), columns=["ground_truth", "prediction"])
        file_name = check_and_create_directory(config_data["PATH_TO_RESULT_OUTPUT_DIR"] + "val/") + f"./{model_checkpoint.split('/')[-1]}_{version}_epoch_" + str(epoch+1) + "_val_results.csv"
        val_df.to_csv(file_name, index=False)
        print("Validation File Saved")
    elif "Test" in desc:
        test_df = pd.DataFrame(list(zip(gold, predictions)), columns=["ground_truth", "prediction"])
        file_name = check_and_create_directory(config_data["PATH_TO_RESULT_OUTPUT_DIR"] + "test/") + f"./{model_checkpoint.split('/')[-1]}_{version}_epoch_" + str(epoch+1) + "_test_results.csv"
        test_df.to_csv(file_name, index=False)
        print("Test File Saved")
    
    del predictions
    del gold
    gc.collect()
    torch.cuda.empty_cache()

    return result

def _save(model,
          output_dir: str,
          tokenizer=None,
          state_dict=None):
    # If we are executing this function, we are the process zero, so we don't check for that.
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model checkpoint to {output_dir}")
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    if not isinstance(model, PreTrainedModel):
        if isinstance(unwrap_model(model), PreTrainedModel):
            if state_dict is None:
                state_dict = model.state_dict()
            unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
        else:
            print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if state_dict is None:
                state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "WEIGHTS_NAME"))
    else:
        model.save_pretrained(output_dir, state_dict=state_dict)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

def save_model(model,
               output_dir: str,
               tokenizer=None,
               state_dict=None):
    """
    Will save the model, so you can reload it using :obj:`from_pretrained()`.
    Will only save from the main process.
    """
    _save(model, output_dir, tokenizer=tokenizer, state_dict=state_dict)

def train(model,
          dataset,
          **gen_kwargs):
    """
    Args:
        model: PyTorch model, with a transformers based text-encoder
        dataset: object of type InLegalNERDataset
    Desc: Performs overall training and validation over provided 
        data and model type
    """
    optimizer = AdamW(model.parameters(),
                      lr=config_data["LEARNING_RATE"],
                      weight_decay=config_data["WEIGHT_DECAY"])

    train_losses = []
    val_losses = []
    val_f1 = []
    patience = 1

    MODEL_CHECKPOINT = config_data["MODEL_CHECKPOINT"]
    VERSION = config_data["VERSION"]

    # ------------------------------ READ DATASET ------------------------------ #  
    train_data_loader = dataset.set_up_data_loader("train")
    val_data_loader = dataset.set_up_data_loader("dev")
    print("\nTraining and Validation Data Loaded...")
    # -------------------------------------------------------------------------- # 

    for epoch in range(config_data["MAX_EPOCHS"]):
        train_loss = train_epoch(model=model,
                                 data_loader=train_data_loader,
                                 optimizer=optimizer,
                                 use_crf=dataset.use_crf)
        train_losses.append(train_loss)

        val_loss = val_epoch(model=model,
                             data_loader=val_data_loader,
                             use_crf=dataset.use_crf)
        val_losses.append(val_loss)

        val_results = get_val_scores(model=model,
                                     data_loader=val_data_loader,
                                     desc="Validation Generation Iteration",
                                     epoch=epoch,
                                     use_crf=dataset.use_crf,
                                     ner_labels_encoding=dataset.ner_label_encodings,
                                     ner_labels_list=dataset.ner_labels_list,
                                     **gen_kwargs)
        val_f1.append(val_results["f1"])

        print("Epoch: {:0.2f}\ttrain_loss: {:0.2f}\tval_loss: {:0.2f}\tmin_validation_loss: {:0.2f}".format(
            epoch+1, train_loss, val_loss, min(val_losses)))

        print("val_precision: {:0.2f}\tval_recall: {:0.2f}\tval_f1: {:0.2f}\tval_accuracy: {:0.2f}".format(
            val_results["precision"], val_results["recall"], val_results["f1"], val_results["accuracy"]))

        path = config_data["PATH_TO_MODEL_OUTPUT_DIR"] + f"{MODEL_CHECKPOINT.split('/')[-1]}_{VERSION}_epoch_" + str(epoch+1) + "_" + datetime.now().strftime("%d-%m-%Y-%H:%M")
        
        save_model(model,
                   path,
                   dataset.tokenizer)
        print("Model saved at path: ", path)

        print("---------------------------------------------------------------")

        if val_results["f1"] < max(val_f1):
            patience = patience + 1
            if patience == config_data["EARLY_STOPPING_THRESHOLD"]:
                break
        else:
            patience = 1

        # keep top-3 models 
        model_foldernames = os.listdir(config_data["PATH_TO_MODEL_OUTPUT_DIR"])
        model_foldernames = [os.path.join(config_data["PATH_TO_MODEL_OUTPUT_DIR"], foldername) for foldername in model_foldernames]
        if len(model_foldernames) > 3:
            oldest_folderpath = min(model_foldernames, key=os.path.getctime)
            shutil.rmtree(oldest_folderpath)
            print(f"Deleted previously saved model: {oldest_folderpath}")

        del train_loss
        del val_loss
        del path
        gc.collect()
        torch.cuda.empty_cache()

def generate_results(model,
                     dataset,
                     use_crf:bool=False):
    """
    Args: 
        model: PyTorch model, with a transformers based text-encoder,
            can be either [`InLegalBERTforTokenClassification`,
            `XLMRobertaforTokenClassification`,
            `XLMRobertaCRFforTokenClassification`,
            `Ensembler (accumulation of 2 or more model-types)`]
        dataset: dataset: object of type InLegalNERDataset
        use_crf: (bool) specifies whether the `model` extends to 
            Conditional Random Fiels
    Returns:
        Classification report over all the NER classes alongwith
            Overall F1 score and Accuracy
    """
    predictions = []
    ground_truth = []

    print("Loading validation data...")
    data_loader = dataset.set_up_data_loader("dev")
        
    if use_crf:
        for batch in tqdm(data_loader):
            input_ids, attention_mask, labels, _ = batch
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.predictions.cpu().tolist())
            ground_truth.extend(labels.to("cpu").tolist()) 
    else:
        for batch in tqdm(data_loader):
            input_ids, attention_mask, labels, _ = batch
            outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)
            batch_predictions = probabilities.argmax(dim=-1).to("cpu").tolist()
            predictions.extend(batch_predictions)
            ground_truth.extend(labels.to("cpu").tolist())

    del probabilities
    del batch_predictions
    gc.collect()
    torch.cuda.empty_cache()
    
    results = get_scores((predictions, ground_truth),
                         ner_labels_encoding=dataset.ner_label_encodings,
                         ner_labels_list=dataset.ner_labels_list, 
                         use_crf=use_crf,
                         full_rep=True)
    
    path_to_file=os.path.join(str(Path(config_data['PATH_TO_RESULT_OUTPUT_DIR']).parent), "./final_results_dev/" + config_data["MODEL_CHECKPOINT"].split("/")[-1] + "-finetuned-for-token-classification-" + config_data["VERSION"] + ".csv")
    save_results(path_to_file=path_to_file, results=results)
    print("results file saved @ ", path_to_file)

    return results


def generate_confusion_matrix(model,
                            dataset, 
                            use_crf: bool=False):
    """
    Args: 
        model: PyTorch model, with a transformers based text-encoder,
            can be either [`InLegalBERTforTokenClassification`,
            `XLMRobertaforTokenClassification`,
            `XLMRobertaCRFforTokenClassification`,
            `Ensembler (accumulation of 2 or more model-types)`]
        dataset: dataset: object of type InLegalNERDataset
        use_crf: (bool) specifies whether the `model` extends to 
            Conditional Random Fiels
    Returns:
        generate confusion matrix and saves the plot @ PATH_TO_RESULT_OUTPUT_DIR
    """

    predictions = []
    ground_truth = []

    print("Loading validation data...")
    data_loader = dataset.set_up_data_loader("dev")
        
    if use_crf:
        for batch in tqdm(data_loader):
            input_ids, attention_mask, labels, mask = batch
            outputs = model(input_ids, attention_mask)
            predictions.extend(outputs.predictions.cpu().tolist())
            ground_truth.extend(labels.to("cpu").tolist()) 
    else:
        for batch in tqdm(data_loader):
            input_ids, attention_mask, labels, _ = batch
            outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)
            batch_predictions = probabilities.argmax(dim=-1).to("cpu").tolist()
            predictions.extend(batch_predictions)
            ground_truth.extend(labels.to("cpu").tolist())

    # if use_crf [<s>, </s>, <pad>] else [-100]
    ignore_idx_list = [0, 2, 1] if use_crf else [-100] 

    true_predictions = [
        [dataset.ner_labels_list[p] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, ground_truth)
    ]

    true_labels = [
        [dataset.ner_labels_list[l] for (p, l) in zip(prediction, label) if l not in ignore_idx_list]
        for prediction, label in zip(predictions, ground_truth)
    ]

    true_predictions = [tag.split('-')[-1] for instance in true_predictions for tag in instance]
    true_labels = [tag.split('-')[-1] for instance in true_labels for tag in instance]

    cm = confusion_matrix(true_predictions, true_labels, labels=dataset.class_labels) # generate confusion_matrix

    sns.set()
    sns.heatmap(cm, annot=True, cmap='PuRd', fmt='d', xticklabels=dataset.class_labels, yticklabels=dataset.class_labels)

    # set plot labels
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    PATH_TO_RESULT_OUTPUT_DIR = config_data["PATH_TO_RESULT_OUTPUT_DIR"] # retrieve path to `results` directory
    # save heatmap
    plt.savefig(os.path.join(str(Path(PATH_TO_RESULT_OUTPUT_DIR).parent), "./confusion_matrix/" + config_data["MODEL_CHECKPOINT"].split("/")[-1] + "-finetuned-for-token-classification-" + config_data["VERSION"] + '_confusion_matrix.jpg'), dpi=300, bbox_inches='tight')
    print("confusion matrix saved @ ", os.path.join(str(Path(PATH_TO_RESULT_OUTPUT_DIR).parent), "./confusion_matrix/" + config_data["MODEL_CHECKPOINT"].split("/")[-1] + "-finetuned-for-token-classification-" + config_data["VERSION"] + '_confusion_matrix.png'))
    plt.show()