from transformers import RobertaModel

model_name_or_path = "./pretrained_models/roberta"

model = RobertaModel.from_pretrained(model_name_or_path)

