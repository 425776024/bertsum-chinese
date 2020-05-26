# -*- coding: utf-8 -*-
import gc
import glob

import json
import os
from os.path import join as pjoin

import torch
from multiprocessing.pool import Pool
from transformers import BertTokenizer


class BertData():
    def __init__(self, args):
        self.args = args
        # 加载中文词汇表
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_base_chinese, do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src: str, key_sents_ids: list) -> tuple:
        if len(src) < self.args.min_nsents:
            return None
        original_src_txt = [' '.join(s) for s in src]
        labels = [0] * len(src)
        for k_idx in key_sents_ids:
            labels[k_idx] = 1
        # 满足大于min_src_ntokens的句子才会被选中
        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        # 截取超过max_src_ntokens部分的不要
        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        # 所有句子连接成一大长文本
        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        # 限定最终最长长度
        src_subtokens = src_subtokens[:self.args.max_position_embeddings - 2]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        # 文本字，转成vocab映射后的 token
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        # 拿到[SEP]分割点位置
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        # 计算前后[sep]距离
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        # 单双，每过一个[SEP]，segment为0/1
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        # [CLS]分类标记位置
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt


def _format_to_bert(params) -> None:
    json_file, args, save_file = params
    if os.path.exists(save_file):
        print('Ignore %s' % save_file)
        return

    bert = BertData(args)

    print('Processing %s' % json_file)
    jobs = json.load(open(json_file, encoding='utf-8'))
    datasets = []
    for d in jobs:
        source, oracle_ids = d['src'], d['ids']
        # 转成 src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt
        b_data = bert.preprocess(source, oracle_ids)
        if b_data is None:
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt = b_data
        # 以字典形式保存
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt}
        datasets.append(b_data_dict)
    print('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_bert(args) -> None:
    if args.dataset != '':
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        pts = sorted(glob.glob(pjoin(args.raw_path, '*' + corpus_type + '*.json')))
        for json_f in pts:
            # 请注意，windows 和linux不一样
            if '\\' in json_f:
                real_name = json_f.split('\\')[-1]
            else:
                real_name = json_f.split('/')[-1]

            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print('a_lst:', a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()
