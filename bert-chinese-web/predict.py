#!/usr/bin/env python

import torch
from src.models.model_builder_LAI import Summarizer
from src.prepro.data_builder import BertData, BatchExample
from config import load_from, bert_config_path, vocab_path, max_summary_size
import os


class Bert_summary_model(object):
    def __init__(self, device=torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")):
        self.device = device
        self.data_process = BertData(vocab_path=vocab_path, device=device)
        self.model = self.load_model(load_from)
        self.max_process_len = self.model.bert_config.max_position_embeddings - 2

    def load_model(self, load_from):
        checkpoint = torch.load(load_from, map_location=lambda storage, loc: storage)
        print('loading....', load_from)
        model = Summarizer(self.device, bert_config_path=bert_config_path)
        model.load_cp(checkpoint)
        model.eval()
        return model

    def save(self):
        model_state_dict = self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
        }

        checkpoint_path = os.path.join('models/bert_classifier', 'model_s.pt')
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path
        print('saved:', checkpoint_path)

    def long_predict(self, document: str, max_summary_size=max_summary_size, min_sent_num=3):
        assert len(document) > self.max_process_len, '不够长'
        # 超过这个长度的切开
        document_splits = self.data_process.split_long_doc(document, self.max_process_len)

        predict_s = [self.predict(document=doc_i, max_summary_size=max_summary_size) for doc_i in document_splits]
        rt = ''.join(predict_s)
        # 新的摘要，如果句子还太多
        # document_splits = self.data_process.split_long_doc(rt, self.max_process_len)
        example, document_splits = self.data_process.preprocess(rt, min_sent_num=min_sent_num)

        if len(rt) > self.max_process_len and len(document_splits) <= 3:
            txt = document_splits[0]
            # 如果第一句话就超过了最大限定长度（总有一些奇葩句子就是这么变态）
            if len(txt) > max_summary_size:
                txt_arr = txt.split('，')
                txt = ''
                for ti in txt_arr:
                    if len(txt + ti) < max_summary_size:
                        txt += ti
                    else:
                        txt += ti
                        txt = txt[:max_summary_size]
                        break

            else:
                for ti in document_splits[1:]:
                    if len(txt + ti) < self.max_process_len:
                        txt += ti
                    else:
                        txt += ti
                        txt = txt[:max_summary_size]
                        break
            rt = txt
        # 依然满足长文本预测逻辑，继续递归下去
        elif len(rt) > self.max_process_len and len(document_splits) > min_sent_num:
            rt = self.long_predict(rt)
        # 句子量满足了，但是总文本还是太长了，就缩小句子数
        else:
            # 此时 len(rt)一定 < self.max_process_len ，进行正式predict逻辑
            rt = self.predict(rt, max_summary_size, min_sent_num)
        return rt

    def predict(self, document: str, max_summary_size=max_summary_size, min_sent_num=3):
        # 如果低于最大要求长度，就不做摘要了
        if len(document) <= max_summary_size:
            return document
        # 进行切分，如果句子数量低于min_sent_num返回的会是None（就2句话，模型取min_sent_num句最核心的），
        example, doc_sents = self.data_process.preprocess(document, min_sent_num=min_sent_num)
        if example is None or (len(document) > self.max_process_len) or len(doc_sents) <= min_sent_num:
            # 特殊问题特殊处理，（就2句话，还非常长，还预测干嘛？直接截断返回）
            return ''.join(doc_sents)[:max_summary_size]
        # _____推断_____
        o_sent_scores, _ = self.model(example.src, example.segs, example.clss, example.src_mask, example.cls_mask)
        o_sent_scores_np = o_sent_scores.cpu().detach().numpy()
        sort_idx = o_sent_scores_np.argsort()
        # socore,大到小 索引
        key_idx = sort_idx.tolist()[0][::-1]
        summary_idx = []
        tp_summary = ''
        for ki in key_idx:
            sent_i = doc_sents[ki]
            if len(tp_summary) + len(sent_i) < max_summary_size:
                summary_idx.append(ki)
                tp_summary += sent_i

        # 以文章顺序写出
        summary_idx = sorted(summary_idx)
        key_sents = [doc_sents[i] for i in summary_idx]
        rt = ''.join(key_sents)
        return rt



if __name__ == '__main__':
    bert_summary_model = Bert_summary_model()
    bert_summary_model.test_batch_example()
