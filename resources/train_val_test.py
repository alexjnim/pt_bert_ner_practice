import time
import torch
import json
import numpy as np
from datetime import date
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from resources.build_model import nerModel
from utils.helper_functions import format_time
from utils.pytorchtools import EarlyStopping


def train_model(args, model, train_dataloader, val_dataloader=None):
    """Train and validate the NER BERT model."""
    training_stats = []
    train_loss_set = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    name = (
        args.savedir
        + str(args.epochs)
        + "_epochs_"
        + str(date.today().strftime("%d_%m_%Y"))
    )
    save_model_path = name + "_model.pt"
    save_results_path = name + "_results.json"

    early_stopping = EarlyStopping(
        patience=args.patience, verbose=True, path=save_model_path
    )
    optimizer, scheduler = build_optimizer_scheduler(
        model=model, epochs=args.epochs, train_dataloader=train_dataloader
    )

    print("Start training...\n")
    for epoch_i in range(args.epochs):
        # =======================================
        #               Training
        # =======================================
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
        )
        print("-" * 70)
        t0 = time.time()
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids = batch["input_ids"].to(device)
            b_attention_mask = batch["attention_mask"].to(device)
            b_true_labels = batch["labels"].to(device)

            model.zero_grad()
            optimizer.zero_grad()

            _, loss = model(b_input_ids, b_attention_mask, b_true_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            train_loss_set.append(loss.item())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                )
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("-" * 70)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader:
            avg_val_loss, avg_val_accuracy, validation_time = evaluate(
                model, val_dataloader
            )
            time_elapsed = time.time() - t0_epoch
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_val_loss:^10.6f} | {avg_val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
            )
            print("-" * 70)

            early_stopping(avg_val_loss, model)
        else:
            early_stopping(avg_train_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

        with open(save_results_path, "w") as f:
            json.dump(training_stats, f)

    print("\n")
    print("Training complete!")
    print("Time taken to complete training: {}".format(time.time() - t0))
    return model, training_stats, train_loss_set


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    t0 = time.time()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tracking variables
    avg_val_accuracy = []
    avg_val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        b_true_labels = batch["labels"].to(device)

        # Compute logits
        with torch.no_grad():
            tags, loss = model(b_input_ids, b_attention_mask, b_true_labels)

        avg_val_loss.append(loss.item())
        tag_preds = tags.argmax(2).cpu().numpy().reshape(-1)

        tag_accuracy = (
            tag_preds == b_true_labels.cpu().numpy().reshape(-1)
        ).mean() * 100
        avg_val_accuracy.append(tag_accuracy)

    # Compute the average accuracy and loss over the validation set.
    avg_val_loss = np.mean(avg_val_loss)
    avg_val_accuracy = np.mean(avg_val_accuracy)
    validation_time = format_time(time.time() - t0)
    return avg_val_loss, avg_val_accuracy, validation_time


def test_model(args, test_dataloader, num_tags, label_map):
    """[Here we will test the final model by generating a classification report of the model against the test data]

    Args:
        test_dataloader ([pytorch dataloader]): [this will contain the appropriate test data for the NER model]
        num_tags ([int]): [number of NER labels in the dataset]
    """

    save_model_path = (
        args.savedir
        + str(args.epochs)
        + "_epochs_"
        + str(date.today().strftime("%d_%m_%Y"))
        + "_model.pt"
    )
    model = nerModel(num_tags, args.model_name)
    model.load_state_dict(torch.load(save_model_path))
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predicted_tag, true_tag = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Predict
    for batch in test_dataloader:
        b_input_ids = batch["input_ids"].to(device)
        b_attention_mask = batch["attention_mask"].to(device)
        b_true_labels = batch["labels"].to(device)

        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            tag, _ = model(b_input_ids, b_attention_mask, b_true_labels)

        # Move logits and labels to CPU
        tag = tag.argmax(2).cpu().numpy().reshape(-1).tolist()
        b_true_labels = b_true_labels.cpu().numpy().reshape(-1).tolist()
        # Store predictions and true labels
        predicted_tag.extend(tag)
        true_tag.extend(b_true_labels)

    print(
        classification_report(
            true_tag,
            predicted_tag,
            target_names=list(label_map.keys()),
            zero_division=0,
        )
    )


def build_optimizer_scheduler(model, epochs, train_dataloader):

    # setting custom optimization parameters for huggingface model and implement a scheduler here as well.
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=5e-5,  # Default learning rate
        eps=1e-8,  # Default epsilon value
    )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value
        num_training_steps=total_steps,
    )

    return optimizer, scheduler
