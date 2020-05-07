import os
import re
import shutil
import time
import argparse

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def get_sentence(x: str):
    doc_sents = re.split('。|\?|！|；|;', doc)
    doc_sents = [str(ds) for ds in doc_sents if ds != '']
    doc_sents.append('')
    doc_sents=[''.join(i) for i in zip(doc_sents[0::2],doc_sents[1::2])]
    return doc_sents


def sent_token_split(doc: str):
    doc_split = list(doc)
    return doc_split


def filter(x: str):
    dr = re.compile(r'<[^>]+>', re.S)
    dr2 = re.compile(r'{[^>]+}', re.S)
    if x is None or x == 'nan':
        return x
    x = dr.sub('', x)
    x = dr2.sub('', x)
    x = x.replace('\u3000', '').strip()
    return x


def int_arr_to_str(arr: list):
    arr = [str(i) for i in arr]
    return ' '.join(arr)


def label_to_idx(label_arr: list):
    # 词袋形 label arr，转成 索引位置：[1,0,1,1,0]>>>>>[0,2,3]
    return [i for i, li in enumerate(label_arr) if li == 1]
