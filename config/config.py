from datetime import date

today = date.today()
date = today.strftime("%d_%m_%Y")

# make this "all" if you want all words to be trained
num_of_words = 500
# max number of words per sentence
max_token_len = 50
BERT_MODEL_NAME = "bert-base-uncased"

epochs = 2

model_path = "saved_model/"
model_name = "ner_model_" + str(epochs) + "_epochs_" + str(date) + ".pt"
save_model_path = model_path + model_name
save_label_encoders_path = model_path + "label_encoders_" + str(date) + ".bin"
