import os

src_path = os.path.abspath(os.path.dirname(__file__))

project_path = src_path + '/../'

bert_base_chinese = root + 'bert-base-chinese/'


# cuda
device = 'cpu'
load_from = root + 'models/bert_classifier/xxx.pt'
vocab_path = root + 'bert-base-chinese/vocab.txt'
bert_config_path = root + 'bert-base-chinese/config.json'