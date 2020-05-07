# -*- coding: utf-8 -*
import torch
from transformers import BertTokenizer
from src.others.utils import filter, get_sentence, sent_token_split


class Example(object):
    def __init__(self, data: list, device):
        pre_src = [data[0]]
        pre_segs = [data[1]]
        pre_clss = [data[2]]

        src = torch.tensor(pre_src)
        src_mask = ~(src == 0)

        segs = torch.tensor(pre_segs)

        clss = torch.tensor(pre_clss)
        clss_mask = ~(clss == -1)

        setattr(self, 'src', src.to(device))
        setattr(self, 'src_mask', src_mask.to(device))
        setattr(self, 'segs', segs.to(device))
        setattr(self, 'clss', clss.to(device))
        setattr(self, 'clss_mask', clss_mask.to(device))


class BertData():
    def __init__(self, vocab_path,device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path,do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def split_long_doc(self,document:str,max_num=510):
        document=filter(document)
        doc_sents=get_sentence(document)
        document_list=[]
        a_temp_doc=''
        for si in doc_sents:
            if len(a_temp_doc)+len(si)>max_num:
                document_list.append(a_temp_doc)
                a_temp_doc = si
            else:
                a_temp_doc += si
        if a_temp_doc != '':
            document_list.append(a_temp_doc)

        return document_list


    def preprocess(self, document: str):

        document = filter(document)
        doc_sents = get_sentence(document)
        src = [sent_token_split(sent) for sent in doc_sents]

        src_text = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_text)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]' + src_subtokens + '[SEP]']

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

        data = [src_subtoken_idxs, segments_ids, cls_ids]
        example = Example(data, self.device)
        return example, doc_sents
