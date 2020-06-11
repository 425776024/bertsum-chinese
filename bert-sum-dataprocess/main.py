import pandas as pd
import json
from src.utils import filter, have_dirty_key, doc_split, save_data_arr_to_json


def get_input_data_iter():

    data_pd = pd.read_csv(data_path, sep='\t')
    print(data_pd.shape)
    doc_list = data_pd['doc'].tolist()
    json_list = data_pd['json'].tolist()

    for i, json_str_i in enumerate(json_list):
        item_list = json.loads(json_str_i)
        item_dict_list = item_list[0]
        # 关键句
        item_key_sents = [filter(item['sentence']) for item in item_dict_list]

        item_key_sents = [si for si in item_key_sents if have_dirty_key(si) == False]
        if len(item_key_sents) == 0:
            continue
        doc_sents = doc_split(doc_list[i])
        doc_sents = [filter(di) for di in doc_sents]
        # 组成：文档句，关键句。（比如，文档10句话，其中3句话是关键句子（拿来被做抽取式摘要））
        item = {'doc_sents': doc_sents, 'key_sents': item_key_sents}
        yield item


data_path = 'data/scope.csv'

data_arr_iter = get_input_data_iter()

save_data_arr_to_json(data_arr_iter, chunk_size=2000, file_name='json_data/scope.train')
