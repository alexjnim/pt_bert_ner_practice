import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from config import config
import csv


def get_cleaned_data(data_path):

    # List of all sentences in the dataset.
    sentences, labels = [], []

    # Lists to store the current sentence.
    word_tokens, token_labels = [], []

    # Gather the set of unique labels.
    unique_labels = []

    # Read the dataset line by line. Each line of the file
    with open(data_path, newline="") as lines:
        # Use the `csv` class to split the lines on the tab character.
        line_reader = csv.reader(lines, delimiter="\t")

        # For each line in the file...
        for line in line_reader:
            # If we encounter a blank line, it means we've completed the previous
            # sentence.
            if line == []:
                # Add the completed sentence.
                sentences.append(word_tokens)
                labels.append(token_labels)
                # Start a new sentence.
                word_tokens = []
                token_labels = []
            else:
                # Add the token and its label to the current sentence.
                word_tokens.append(line[1])
                token_labels.append(line[0])
                # Add the label to the list (no effect if it already exists).
                if line[0] not in unique_labels:
                    unique_labels.append(line[0])

    label_map = {}
    # For each label...
    for (i, label) in enumerate(unique_labels):
        # Map it to its integer.
        label_map[label] = i
    # add ignore tag for subtokens and [CLS], [SEP], [PAD] tokens
    label_map["IGNORE"] = -100
    return sentences, labels, label_map
