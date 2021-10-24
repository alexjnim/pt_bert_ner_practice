import torch
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class _dataset(Dataset):
    def __init__(self, tokenizer, max_length, sentences, labels=None, label_map=None):
        # sentences: [["hi", ",", "my", "name", "is", "Alex"], ["hello".....]]
        # pos/tags: [[1 2 3 4 1 5], [....].....]]
        self.sentences = sentences
        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):

        word_tokens = self.sentences[index]
        labels = self.labels[index]

        # Reconstruct the sentence--otherwise `tokenizer` will interpret the list
        # of string tokens as having already been tokenized by BERT.
        sentence = " ".join(word_tokens)

        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = self.tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=self.max_length,  # Pad & truncate all sentences.
            padding="max_length",  # makes sure it pads to max_length
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )

        input_ids = encoded_dict["input_ids"][0]
        attention_mask = encoded_dict["attention_mask"][0]

        if labels:
            assert len(word_tokens) == len(labels)
            # add IGNORE label to all subtokens from tokenizer
            fixed_labels = list(
                chain(
                    *[
                        [self.label_map[label]]
                        + [self.label_map["IGNORE"]]
                        * (len(self.tokenizer.tokenize(word)) - 1)
                        for label, word in zip(labels, word_tokens)
                    ]
                )
            )
            # add IGNORE label for [CLS] token at the front
            fixed_labels = [self.label_map["IGNORE"]] + fixed_labels

            # truncate fixed_labels if longer than self.max_length
            fixed_labels = fixed_labels[
                : min(
                    len(fixed_labels),
                    self.max_length - 1,
                )
            ]
            # add IGNORE label for [SEP] and [PAD] tokens at the end
            fixed_labels = fixed_labels + [self.label_map["IGNORE"]] * (
                self.max_length - len(fixed_labels)
            )
            assert len(input_ids) == len(fixed_labels)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(fixed_labels, dtype=torch.long),
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }


def build_dataloader(
    tokenizer, max_length, sentences, labels=None, label_map=None, trainset=False
):
    dataset = _dataset(tokenizer, max_length, sentences, labels, label_map)

    if trainset:
        sampler = RandomSampler(sentences)
    else:
        sampler = SequentialSampler(sentences)

    return DataLoader(dataset, batch_size=32, sampler=sampler)
