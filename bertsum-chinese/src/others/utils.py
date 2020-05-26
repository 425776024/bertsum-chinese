# -*- coding: utf-8 -*-

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def int_arr_to_str(arr: list):
    arr = [str(i) for i in arr]
    return ' '.join(arr)


def label_to_idx(label_arr: list):
    # 词袋形 label arr，转成 索引位置：[1,0,1,1,0]>>>>>[0,2,3]
    return [i for i, li in enumerate(label_arr) if li == 1]


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params
