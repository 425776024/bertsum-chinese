#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 3:13 PM
# @Author  : xinfa.jiang
# @Site    : 
# @File    : args_config.py
# @Software: PyCharm

import argparse
from others.utils import str2bool

'''
    训练：
    python train_LAI.py -mode train -encoder classifier -dropout 0.1 
    -bert_data_path ../bert_data/LCSTS 
    -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 0  
    -gpu_ranks 0 -world_size 1 -report_every 200 
    -save_checkpoint_steps 2000 -batch_size 3000 
    -decay_method noam -train_steps 30000 
    -accum_count 2 -log_file ../logs/bert_classifier 
    -use_interval true -warmup_steps 10000 
    -train_from ../models/bert_classifier/model_step_28000.pt
    '''

parser = argparse.ArgumentParser()

parser.add_argument("-encoder", default='classifier', type=str,
                    choices=['classifier', 'transformer', 'rnn', 'baseline'])

parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])

# 测试，拷贝部分出来：：bert_data_test
parser.add_argument("-bert_data_path", default='../bert_data_test/LCSTS')
parser.add_argument("-model_path", default='../models/bert_classifier')
parser.add_argument("-result_path", default='../results/result')
parser.add_argument("-temp_dir", default='../temp')

# Bert config path's json
parser.add_argument("-bert_config_path", default='/Users/jiang/Documents/bert/bert-base-chinese/config.json')

# 这里的batch size 不一样，服务器上3000+-，本地测试60
parser.add_argument("-batch_size", default=3000, type=int)

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

parser.add_argument("-save_checkpoint_steps", default=1000, type=int)

# 批次训练数，3个batch
parser.add_argument("-accum_count", default=3, type=int)
# GUP数量
parser.add_argument("-world_size", default=1, type=int)

parser.add_argument("-report_every", default=50, type=int)

# 最多训练次数
parser.add_argument("-train_steps", default=10000, type=int)
parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument('-gpu_ranks', default='0', type=str)
parser.add_argument('-log_file', default='../logs/bert_classifier')
parser.add_argument('-dataset', default='')
parser.add_argument('-seed', default=666, type=int)

parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-test_from", default='../models/bert_classifier/model_step_2.pt')
parser.add_argument("-train_from", default='')

parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)

parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
# bert pre trained model
mode_path = '/Users/jiang/Documents/bert/bert-base-chinese'
parser.add_argument("-mode_path", type=str, default=mode_path)


args = parser.parse_args()
