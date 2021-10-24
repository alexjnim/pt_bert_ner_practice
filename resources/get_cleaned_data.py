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
    label_map = {}

    # Read the dataset line by line. Each line of the file
    with open(data_path, newline="") as lines:
        # Use the `csv` class to split the lines on the tab character.
        line_reader = csv.reader(lines, delimiter="\t")
        label_id = 0
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
                if line[0] not in label_map:
                    label_map[line[0]] = label_id
                    label_id += 1
    # add IGNORE label for [CLS], [SEP], [PAD] and subtokens
    label_map["IGNORE"] = label_id
    return sentences, labels, label_map
