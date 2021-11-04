import torch
import transformers
import torch.nn as nn


def loss_fn(logits, target, attention_mask, num_labels):
    """calculates the loss of the model output against the target labels
    this is taken from the loss function calculation of BertForTokenClassification
    https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForTokenClassification

    Args:
        logits (tensor): logits from the last head of model
        target (tensor): true values
        attention_mask (tensor): attention mask from bert model
        num_labels (int): number of unique true values

    Returns:
        loss (tensor): contains the loss value
    """
    loss_func = nn.CrossEntropyLoss()
    # only consider the loss for the active tokens (some are padded)
    active_loss = attention_mask.view(-1) == 1

    # flatten predictions
    active_logits = logits.view(-1, num_labels)

    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(loss_func.ignore_index).type_as(target),
    )
    loss = loss_func(active_logits, active_labels)
    return loss


class nerModel(nn.Module):
    def __init__(self, num_tags, BERT_MODEL_NAME, freeze_bert=False):
        super(nerModel, self).__init__()
        self.num_tags = num_tags
        self.hidden_dim = 200

        self.bert = transformers.BertModel.from_pretrained(
            BERT_MODEL_NAME, return_dict=False
        )

        self.classify_tags = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.bert.config.hidden_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.hidden_dim, self.num_tags),
        )

        if freeze_bert:
            print("freezing bert parameters")
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, true_labels=None):
        embeddings, _ = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classify_tags(embeddings)

        if true_labels is None:
            # this is for prediction when we don't have the real labels
            return logits
        else:
            # if we have the real labels, we can get the loss (for train, val and test data)
            loss = loss_fn(logits, true_labels, attention_mask, self.num_tags)
            return tags, loss
