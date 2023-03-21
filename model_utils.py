import os
import gc
import shutil
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.modeling_outputs import TokenClassifierOutput
from helpers import config_data, check_and_create_directory
from evaluate import load
seqeval = load("seqeval")

def prediction_procedure(outputs: TokenClassifierOutput,
                         offsets: list,
                         label_map: dict,
                         texts: list = None):
    
    """
    mimics the oringinal tokenclassification prediction procedure achieved through the transformers
    pipeline
    """
    
    probabilities = F.softmax(outputs.logits, dim=-1)
    predictions = probabilities.argmax(dim=-1).tolist()

    final_results = []
    for seq_idx, _ in enumerate(predictions):

        results = []
        idx = 0
        while idx < len(predictions[seq_idx]):
            pred = predictions[seq_idx][idx]
            label = label_map[pred]
            if label != "O":
                # Remove the B- or I-
                label = label[2:]
                start, end = offsets[seq_idx][idx]

                while idx < len(predictions[seq_idx]) and label_map[predictions[seq_idx][idx]] == f"I-{label}":
                    _, end = offsets[seq_idx][idx]
                    idx += 1

                results.append({
                    "entity_group": label, "score": probabilities[seq_idx][idx][pred],
                    "start": start, "end": end
                })
                if texts is not None:
                    word =  texts[seq_idx][start: end]
                    result = results.pop()
                    result["word"] = word
                    results.append(result)
                
            idx += 1
        final_results.append(results)

    return final_results

def get_scores(p, NER_label_encoding_dict, NER_labels_list, full_rep: bool=False):
    predictions, labels = p
    
    ignore_tags = [NER_label_encoding_dict['<s>'],
                   NER_label_encoding_dict['</s>'],
                   NER_label_encoding_dict['<pad>']]

    true_predictions = [
        [NER_labels_list[p] for (p, l) in zip(prediction, label) if l not in ignore_tags]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [NER_labels_list[l] for (p, l) in zip(prediction, label) if l not in ignore_tags]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=1)
    
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
                device="cuda"):
    model.train()
    epoch_train_loss = 0.0
    pbar = tqdm(data_loader, desc="Training Iteration")
    for step, batch in enumerate(pbar):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels, mask = batch
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids,
                     attention_mask=attention_mask,
                     labels=labels,
                     mask=mask,
                     reduction='mean')
        
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
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_train_loss / step

def val_epoch(model,
              data_loader,
              device="cuda"):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, mask = batch

            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          mask=mask,
                          reduction='mean')
            
            loss = outputs.loss
            epoch_val_loss += loss.item()

    del batch
    del input_ids
    del attention_mask
    del labels
    del mask
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_val_loss / step

def test_epoch(model,
               tokenizer,
               data_loader,
               desc,
               device="cuda",
               **gen_kwargs):
    model.eval()
    out_predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, mask = batch

            outputs = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                **gen_kwargs)
            
            # check for the model outputs {loss, logits, hidden_states, attentions}
            # print(outputs.logits.shape) ideally should be (batch_size x sequence_length x num_classes)
            
            # probabilities = F.softmax(seqs, dim=-1)
            # predictions = probabilities.argmax(dim=-1).cpu().tolist()

            predictions = outputs.predictions
            
            out_predictions.extend(predictions.cpu().tolist())
            gold.extend(labels.cpu().tolist())

    del batch
    del input_ids
    del attention_mask
    del labels
    del mask
    # del probabilities
    del predictions
    gc.collect()
    torch.cuda.empty_cache()           

    return out_predictions, gold

def get_val_scores(model,
                   tokenizer,
                   data_loader,
                   desc,
                   epoch,
                   NER_label_encoding_dict,
                   NER_labels_list,
                   **gen_kwargs):
    predictions, gold = test_epoch(model,
                                   tokenizer,
                                   data_loader,
                                   desc=desc,
                                   **gen_kwargs)
    result = get_scores((predictions, gold),
                        NER_label_encoding_dict=NER_label_encoding_dict,
                        NER_labels_list=NER_labels_list)

    MODEL_CHECKPOINT = config_data["MODEL_CHECKPOINT"]
    VERSION = config_data["VERSION"]

    if "Validation" in desc:
        val_df = pd.DataFrame(list(zip(gold, predictions)), columns=["ground_truth", "prediction"])
        file_name = check_and_create_directory(config_data["PATH_TO_RESULT_OUTPUT_DIR"] + "val/") + f"./{MODEL_CHECKPOINT.split('/')[-1]}_{VERSION}_epoch_" + str(epoch+1) + "_val_results.csv"
        val_df.to_csv(file_name, index=False)
        print("Validation File Saved")
    elif "Test" in desc:
        test_df = pd.DataFrame(list(zip(gold, predictions)), columns=["ground_truth", "prediction"])
        file_name = check_and_create_directory(config_data["PATH_TO_RESULT_OUTPUT_DIR"] + "test/") + f"./{MODEL_CHECKPOINT.split('/')[-1]}_{VERSION}_epoch_" + str(epoch+1) + "_test_results.csv"
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
          tokenizer,
          dataset,
          **gen_kwargs):
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
                                 optimizer=optimizer)
        train_losses.append(train_loss)

        val_loss = val_epoch(model=model,
                             data_loader=val_data_loader)
        val_losses.append(val_loss)

        val_results = get_val_scores(model,
                                     tokenizer,
                                     val_data_loader,
                                     desc="Validation Generation Iteration",
                                     epoch=epoch,
                                     NER_label_encoding_dict=dataset.ner_label_encodings,
                                     NER_labels_list=dataset.ner_labels_list,
                                     **gen_kwargs)
        val_f1.append(val_results["f1"])

        print("Epoch: {:0.2f}\ttrain_loss: {:0.2f}\tval_loss: {:0.2f}\tmin_validation_loss: {:0.2f}".format(
            epoch+1, train_loss, val_loss, min(val_losses)))

        print("val_precision: {:0.2f}\tval_recall: {:0.2f}\tval_f1: {:0.2f}\tval_accuracy: {:0.2f}".format(
            val_results["precision"], val_results["recall"], val_results["f1"], val_results["accuracy"]))

        path = config_data["PATH_TO_MODEL_OUTPUT_DIR"] + f"{MODEL_CHECKPOINT.split('/')[-1]}_{VERSION}_epoch_" + str(epoch+1) + "_" + datetime.now().strftime("%d-%m-%Y-%H:%M")
        
        save_model(model,
                   path,
                   tokenizer)
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
