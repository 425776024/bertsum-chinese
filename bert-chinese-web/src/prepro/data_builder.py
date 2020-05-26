# -*- coding: utf-8 -*

from transformers import BertTokenizer
from src.others.utils import filter, doc_split, sent_token_split
import torch


class BatchExample(object):
    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, batch_example=None, device=None):
        if batch_example is not None:
            self.batch_size = len(batch_example)
            if batch_example != []:
                pre_src = [e.src.cpu().numpy().tolist()[0] for e in batch_example]
                pre_segs = [e.segs.cpu().numpy().tolist()[0] for e in batch_example]
                pre_clss = [e.clss.cpu().numpy().tolist()[0] for e in batch_example]

                src = torch.tensor(self._pad(pre_src, 0))
                segs = torch.tensor(self._pad(pre_segs, 0))
                mask = ~(src == 0)

                clss = torch.tensor(self._pad(pre_clss, -1))
                mask_cls = ~ (clss == -1)
                clss[clss == -1] = 0

                setattr(self, 'clss', clss.to(device))
                setattr(self, 'mask_cls', mask_cls.to(device))
                setattr(self, 'src', src.to(device))
                setattr(self, 'segs', segs.to(device))
                setattr(self, 'mask', mask.to(device))

    def __len__(self):
        return self.batch_size


class Example(object):
    def __init__(self, data: list, device=None):
        pre_src = [data[0]]
        pre_segs = [data[1]]
        pre_clss = [data[2]]
        src = torch.tensor(pre_src)
        src_mask = ~(src == 0)
        segs = torch.tensor(pre_segs)
        clss = torch.tensor(pre_clss)
        cls_mask = ~ (clss == -1)

        setattr(self, 'src', src.to(device))
        setattr(self, 'src_mask', src_mask.to(device))
        setattr(self, 'segs', segs.to(device))
        setattr(self, 'clss', clss.to(device))
        setattr(self, 'cls_mask', cls_mask.to(device))


class BertData(object):
    def __init__(self, vocab_path, device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def split_long_doc(self, document: str, max_num=510):
        document = filter(document)
        doc_sents = doc_split(document)
        document_list = []
        a_temp_doc = ''
        if len(doc_sents) <= 1:
            return doc_sents

        for si in doc_sents:
            if len(a_temp_doc) + len(si) > max_num:
                document_list.append(a_temp_doc)
                a_temp_doc = si
            else:
                a_temp_doc += si
        if a_temp_doc != '':
            document_list.append(a_temp_doc)
        return document_list

    def preprocess(self, document: str, min_sent_num=3):
        document = filter(document)
        doc_sents = doc_split(document)
        if len(doc_sents) <= min_sent_num:
            return None, doc_sents

        src = [sent_token_split(sent) for sent in doc_sents]

        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        # bert,512写死了
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        data = [src_subtoken_idxs, segments_ids, cls_ids]
        example = Example(data, self.device)
        return example, doc_sents
