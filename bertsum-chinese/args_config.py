#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 3:13 PM
# @Author  : xinfa.jiang
# @Site    : 
# @File    : args_config.py
# @Software: PyCharm

import argparse
from src.others.utils import str2bool
import os

root = os.path.abspath(os.path.dirname(__file__))

results_path = os.path.join(root, 'results')
models_path = os.path.join(root, 'models')
# bert_base_chinese = os.path.join(root, 'bert-base-chinese')
bert_base_chinese = '/Users/jiang/Documents/bert/bert-base-chinese'

parser = argparse.ArgumentParser()
parser.add_argument("-encoder", default='classifier', type=str,
                    choices=['classifier'])

# 训练还是测试，目前支持 train , test
parser.add_argument("-mode", default='train', type=str, choices=['train', 'test'])

# bert_data_path：训练的pt数据目录，bert_data/LCSTS ： 取目录下LCSTS开头的数据
parser.add_argument("-bert_data_path", default='bert_data/LCSTS')
parser.add_argument("-model_path", default='models/bert_classifier')
parser.add_argument("-result_path", default='results/result')
parser.add_argument("-temp_dir", default='temp')

# 必须：预训练的pytorch 的bert-base-chinese模型路径下的配置目录
bert_mode_json_path = os.path.join(bert_base_chinese, 'config.json')
parser.add_argument("-bert_config_path", default=bert_mode_json_path)

parser.add_argument("-batch_size", default=600, type=int)

parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-hidden_size", default=128, type=int)
parser.add_argument("-ff_size", default=2048, type=int)
parser.add_argument("-heads", default=8, type=int)
parser.add_argument("-inter_layers", default=2, type=int)
parser.add_argument("-rnn_size", default=512, type=int)

parser.add_argument("-param_init", default=0, type=float)
parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-dropout", default=0.1, type=float)
parser.add_argument("-optim", default='adam', type=str)
parser.add_argument("-lr", default=2e-3, type=float)
parser.add_argument("-beta1", default=0.9, type=float)
parser.add_argument("-beta2", default=0.999, type=float)
parser.add_argument("-decay_method", default='noam', type=str)
parser.add_argument("-warmup_steps", default=8000, type=int)
parser.add_argument("-max_grad_norm", default=0, type=float)
parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

parser.add_argument("-save_checkpoint_steps", default=1000, type=int)

# 批次训练数，3个batch
parser.add_argument("-accum_count", default=3, type=int)

# 最多训练次数
parser.add_argument("-train_steps", default=40000, type=int)

parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)
parser.add_argument('-log_file', default='logs/bert_classifier')

parser.add_argument('-seed', default=666, type=int)

# 在test的时候有用，告诉加载哪个保存的step模型进行预测
parser.add_argument("-test_from", default='')

# 训练制定起始模型，没有这个，请设置为空 :'' ,有的话会基于这个模型增量训练
parser.add_argument("-train_from", default='')

# 必须：预训练的pytorch 的bert-base-chinese模型路径

parser.add_argument("-bert_base_chinese", type=str, default=bert_base_chinese)

args = parser.parse_args()
