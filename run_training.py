import argparse
import torch
from utils.plot_results import plot_results
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from resources.build_model import nerModel
from resources.get_cleaned_data import get_cleaned_data
from resources.build_dataloader import build_dataloader
from resources.train_val_test import train_model, test_model


def run(args):
    ##################################
    #            get data
    ##################################

    sentences, labels, label_map = get_cleaned_data(
        data_path="data/restauranttrain.bio"
    )
    test_sentences, test_labels, _ = get_cleaned_data(
        data_path="data/restauranttest.bio"
    )

    (train_sentences, val_sentences, train_labels, val_labels,) = train_test_split(
        sentences, labels, shuffle=True, random_state=42, test_size=0.15
    )

    print("there are {} train sentences".format(len(train_sentences)))
    print("there are {} val sentences".format(len(val_sentences)))
    print("there are {} test sentences".format(len(test_sentences)))

    ##################################
    #        build data loaders
    ##################################

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    train_dataloader = build_dataloader(
        tokenizer=tokenizer,
        max_length=args.maxtokenlen,
        sentences=train_sentences[:100],
        labels=train_labels[:100],
        label_map=label_map,
        trainset=True,
    )
    val_dataloader = build_dataloader(
        tokenizer=tokenizer,
        max_length=20,
        sentences=val_sentences[:50],
        labels=val_labels[:50],
        label_map=label_map,
        trainset=False,
    )
    test_dataloader = build_dataloader(
        tokenizer=tokenizer,
        max_length=20,
        sentences=test_sentences[:50],
        labels=test_labels[:50],
        label_map=label_map,
        trainset=False,
    )

    ##################################
    #        build ner model
    ##################################

    model = nerModel(num_tags=len(label_map), BERT_MODEL_NAME=args.model_name)

    ##################################
    #     train and val model
    ##################################

    trained_model, training_stats, train_loss_set = train_model(
        args,
        model,
        train_dataloader,
        val_dataloader=val_dataloader,
    )

    if args.plot:
        plot_results(training_stats, train_loss_set)

    # ##################################
    # #     test model
    # ##################################

    test_model(
        args=args,
        test_dataloader=test_dataloader,
        num_tags=len(label_map),
        label_map=label_map,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-m",
        default="bert-base-uncased",
        action="store",
        help="name of huggingface bert model",
    )

    parser.add_argument(
        "--maxtokenlen",
        "-l",
        default=20,
        action="store",
        help="max length of sentences",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        default=2,
        action="store",
        help="number of epochs to run training",
    )

    parser.add_argument(
        "--patience",
        "-p",
        default=2,
        action="store",
        help="number of epochs to improve before early stopping",
    )

    parser.add_argument(
        "--savedir",
        "-sd",
        default="saved_models/",
        action="store",
        help="path to save model",
    )

    parser.add_argument(
        "--plot",
        "-pl",
        default=False,
        action="store_false",
        help="plot results from training or not",
    )
    args = parser.parse_args()

    run(args)
