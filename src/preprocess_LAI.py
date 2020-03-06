# -*- coding: utf-8 -*
import argparse
import time

from others.logging import init_logger
from prepro import data_builder_LAI


def do_format_to_bert(args):
    print(time.clock())
    data_builder_LAI.format_to_bert(args)
    print(time.clock())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='format_to_bert', type=str,
                        help='format_raw, format_to_lines or format_to_bert')
    parser.add_argument("-oracle_mode",
                        default='greedy',
                        type=str,
                        help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')

    parser.add_argument("-raw_path", default='../testjson_data')

    # bert 输入训练，保存
    parser.add_argument("-save_path", default='../bert_data_test')

    ###change from 2000 to 16000
    parser.add_argument("-shard_size", default=16000, type=int)
    # 最小句子量，文章不能低于3句话
    parser.add_argument('-min_nsents', default=3, type=int)
    # 最大句子量，文章超过100句话
    parser.add_argument('-max_nsents', default=100, type=int)
    # 句子最短长度
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    parser.add_argument('-log_file', default='../logs/preprocess.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=1, type=int)

    mode_path = '/Users/jiang/Documents/bert/bert-base-chinese'
    parser.add_argument("-mode_path", type=str, default=mode_path)

    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder_LAI.' + args.mode + '(args)')
