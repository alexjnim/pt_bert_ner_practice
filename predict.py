import json
import argparse
import torch
from resources.build_model import nerModel
from resources.build_dataloader import _dataset
from transformers import BertTokenizerFast as BertTokenizer
from config import config


def predict(args, sentence):
    ###############################
    #       prerequisites
    ###############################
    with open(args.savedir + "label_map.json") as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}

    num_tags = len(label_map)

    ###############################
    #       process data
    ###############################

    sentence = sentence.split()
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    dataset = _dataset(
        tokenizer, args.maxtokenlen, [sentence], [["O"] * len(sentence)], label_map
    )

    ###############################
    #   load model
    ###############################

    model = nerModel(num_tags=len(label_map), BERT_MODEL_NAME=args.model_name)
    model.load_state_dict(torch.load(args.savedir + "2_epochs_24_10_2021_model.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###############################
    #          get results
    ###############################

    with torch.no_grad():
        data = dataset.__getitem__(0)

        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)

        input_ids = torch.reshape(input_ids, (1, -1))
        attention_mask = torch.reshape(attention_mask, (1, -1))

        tags = model(input_ids, attention_mask)
        print([tokenizer.decode(i) for i in input_ids[0][1 : len(sentence) + 1]])

        result_tags = tags.argmax(2).cpu().numpy().reshape(-1)[1 : len(sentence) + 1]
        print([inv_map[t] for t in result_tags])


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
        "--savedir",
        "-sd",
        default="saved_models/",
        action="store",
        help="path to save model",
    )

    parser.add_argument(
        "--maxtokenlen",
        "-l",
        default=20,
        action="store",
        help="max length of sentences",
    )

    sentence = """i would give them 5 stars"""
    args = parser.parse_args()
    predict(args, sentence)
