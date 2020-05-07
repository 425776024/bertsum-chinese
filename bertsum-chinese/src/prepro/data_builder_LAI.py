# -*- coding: utf-8 -*
import gc
import glob

import json
import os
import re
from os.path import join as pjoin

import torch
from multiprocessing import Pool
from pytorch_pretrained_bert import BertTokenizer

from others.logging import logger
import emoji


class BertData():
    def __init__(self, args):
        self.args = args
        # 加载中文词汇表
        self.tokenizer = BertTokenizer.from_pretrained(args.mode_path,
                                                       do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src, oracle_ids):

        if len(src) == 0:
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = [0] * len(src)
        # 找出与参考摘要最接近的n句话(相似程度以ROUGE衡量)，标注为1(属于摘要)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]
        # 保留大于最小长度（min_src_ntokens）的内容
        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510] # 限定最大长度512了
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt


def _format_to_bert(params):
    json_file, args, save_file = params
    if os.path.exists(save_file):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
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
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


def format_to_bert(args):
    if args.dataset != '':
        datasets = [args.dataset]
    else:
        datasets = ['train', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        pool.imap(_format_to_bert, a_lst)


        pool.close()
        pool.join()



def sent_token_split(doc, is_short_summary=False):
    '''
    给doc str切成 字级别 token
    '''
    doc_modified = re.sub(r' ', "", doc)
    doc_modified = re.sub(r':\w+:', "", emoji.demojize(doc_modified))

    ### if the doc is a very short summary, just don't split sentence
    if is_short_summary:
        doc_split = [list(doc_modified)]
        return doc_split

    doc_modified = re.sub(r'。', "。 ", doc_modified)
    doc_modified = re.sub(r'！', "！ ", doc_modified)
    doc_modified = re.sub(r'？', "？ ", doc_modified)

    doc_split = re.split(r' ', doc_modified)
    doc_split = [i for i in doc_split if len(i) >= 2]

    if len(doc_split) < 2:
        doc_modified = re.sub(r'，', "， ", doc_modified)
        doc_modified = re.sub(r'；', "； ", doc_modified)
        doc_split = re.split(r' ', doc_modified)
        doc_split = [i for i in doc_split if len(i) >= 2]

    doc_split = [list(i) for i in doc_split]

    return doc_split
