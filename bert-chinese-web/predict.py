#!/usr/bin/env python
import torch
import sys
from src.models.model_builder_LAI import Summarizer
from src.prepro.data_builder import BertData

from src.config import load_from, bert_config_path, vocab_path, src_path

sys.path.append(src_path)


class Bert_summary_model:
    def __init__(self):
        self.device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        self.data_process = BertData(vocab_path=vocab_path, device=self.device)
        self.model = self.load_model()
        self.max_process_len = self.model.bert_config.max_position_embeddings - 2

    def load_model(self):
        checkpoint = torch.load(load_from, map_location=lambda storage, loc: storage)
        model = Summarizer(self.device, bert_config=bert_config_path)
        model.load_cp(checkpoint)
        model.eval()
        return model

    def long_predict(self, document: str, num=4):
        document_splits = self.data_process.split_long_doc(document, self.max_process_len)

        if len(document_splits) <= num:
            return ''.join(document_splits)

        predict_s = [self.predict(doc_i) for doc_i in document_splits]
        rt = ''.join(predict_s)

        if len(rt) > self.max_position_embeddings * 1.5 and len(predict_s) > num:
            rt = self.long_predict(rt)

        return rt

    def predict(self, document: str, num=3):
        example, doc_sents = self.data_process.preprocess(document, min_num=num)

        if example is None or len(document) < 30 and len(doc_sents) <= num:
            return ''.join(doc_sents)

        assert num > 1, 'error:num<1'

        sent_score, _ = self.model(example.src, example.segs, example.clss, example.src_mask, example.clss_mask)

        # note：GPU tensor不能直接转为numpy数组，必须先转到CPU tensor
        key_idx = sent_score.argsort().cpu().numpy().tolist()[0][-num:]
        key_idx = sorted(key_idx)
        ken_sents = [doc_sents[i] for i in key_idx]

        rt = ''.join(ken_sents)
        return rt

    def force_predict(self, document: str, num=3):
        example, doc_sents = self.data_process.preprocess(document, min_num=num)

        if example is None or len(document) < 30 and len(doc_sents) <= num:
            return ''.join(doc_sents)

        assert num > 1, 'error:num<1'

        sent_score, _ = self.model(example.src, example.segs, example.clss, example.src_mask, example.clss_mask)

        # note：GPU tensor不能直接转为numpy数组，必须先转到CPU tensor
        key_idx = sent_score.argsort().cpu().numpy().tolist()[0][-num:]
        key_idx = sorted(key_idx)
        ken_sents = [doc_sents[i] for i in key_idx]

        rt = ''.join(ken_sents)
        return rt
