# -*- coding: utf-8 -*-
import argparse
import time
from src.others.logging import init_logger
from src.prepro import data_builder_LAI


def do_format_to_bert(args):
    print(time.clock())
    data_builder_LAI.format_to_bert(args)
    print(time.clock())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-raw_path", default='json_data')
    # 模型输入训练，保存
    parser.add_argument("-save_path", default='train_data')

    ###change from 2000 to 16000
    parser.add_argument("-shard_size", default=16000, type=int)
    # 最小句子量，文章不能低于3句话
    parser.add_argument('-min_nsents', default=3, type=int)
    # 最大句子量，文章超过100句话
    parser.add_argument('-max_nsents', default=100, type=int)

    # 句子最短长度
    parser.add_argument('-min_src_ntokens', default=3, type=int)
    # 句子最大长度
    parser.add_argument('-max_src_ntokens', default=150, type=int)

    parser.add_argument('-max_position_embeddings', default=512, type=int)
    parser.add_argument('-log_file', default='logs/preprocess.log')

    parser.add_argument('-n_cpus', default=4, type=int)
    parser.add_argument('-dataset', default='gonggao', type=str)

    # bert_base_chinese = '/appcom/apps/chengmengli704/pretrained_model/bert_base/bert-base-chinese/'
    bert_base_chinese = 'bert-base-chinese/'
    parser.add_argument("-bert_base_chinese", type=str, default=bert_base_chinese)

    args = parser.parse_args()
    init_logger(args.log_file)
    data_builder_LAI.format_to_bert(args)
