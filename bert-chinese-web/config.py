import os

root = os.path.abspath(os.path.dirname(__file__))

bert_base_chinese = os.path.join(root, 'bert-base-chinese/')

# run device
# or cuda
device = 'cpu'

# model
max_summary_size = 128
load_from = os.path.join(root, 'models/bert_classifier/model_s.pt')
vocab_path = os.path.join(bert_base_chinese, 'vocab.txt')
bert_config_path = os.path.join(bert_base_chinese, 'config.json')

# web
iphost = '127.0.0.1'
port = 8080
