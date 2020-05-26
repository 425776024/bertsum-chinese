# -*- coding: utf-8 -*-
import gc
import glob
import random
import torch
from src.others.logging import logger


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, minibatch=None, device=None, is_test=False):
        # minibatch:包含一个最小训练批次比如2个文本内容
        data = minibatch
        # DataIterator：batch_buffer(self.dataset) > create_batches > minibatch
        if data is not None:
            self.batch_size = len(data)
            if data != []:
                pre_src = [x[0] for x in data]
                pre_labels = [x[1] for x in data]
                pre_segs = [x[2] for x in data]
                pre_clss = [x[3] for x in data]

                src = torch.tensor(self._pad(pre_src, 0))

                labels = torch.tensor(self._pad(pre_labels, 0))
                segs = torch.tensor(self._pad(pre_segs, 0))
                mask = ~(src == 0)

                clss = torch.tensor(self._pad(pre_clss, -1))
                mask_cls = ~ (clss == -1)
                clss[clss == -1] = 0

                setattr(self, 'clss', clss.to(device))
                setattr(self, 'mask_cls', mask_cls.to(device))
                setattr(self, 'src', src.to(device))
                setattr(self, 'labels', labels.to(device))
                setattr(self, 'segs', segs.to(device))
                setattr(self, 'mask', mask.to(device))
                # src, labels, segs, clss, src_txt
                if is_test:
                    src_str = [x[-1] for x in data]
                    setattr(self, 'src_str', src_str)

    def __len__(self):
        return self.batch_size


def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch:
        yield minibatch


def load_dataset(args, corpus_type, shuffle):
    '''
    加载所有 XX.pt文件，返回的是pt文件的对象：dataset(list 包含多个字典，字典是文本处理好可直接输入的tensor)
    '''
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        logger.info('loaded:%s' % (pt_file))
        return dataset

    # 以正则表达式匹配的文件路径集
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.*.pt'))
    if pts:
        if shuffle:
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    # 不断累求当前调节数据长度，以当前发现的最大长度(max_size) * count
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size,
                 device, shuffle, is_test):
        self.args = args
        # 迭代器，每次迭代返回一个LCSTS.train.0.bert.pt内容，9000+文本数据
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        # d 是一个LCSTS.train.xx.bert.pt，9000+个文本内容的yield，
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            # 上一个LCSTS.train.0.bert.pt完了，开始下一个pt的yield迭代，直到None
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # 内存手动清理
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None
        # 最终self.cur_dataset的数据，分self.batch_size去返回
        return DataIterator(args=self.args,
                            dataset=self.cur_dataset, batch_size=self.batch_size,
                            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        # dataset是data_loader.py -> load_dataset()加载的pt文件（在data_builder_LAI->format_to_bert生成）
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def preprocess(self, a_example, is_test):
        ex = a_example
        src = ex['src']
        if 'labels' in ex:
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']

        segs = ex['segs']
        if not self.args.use_interval:
            segs = [0] * len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']

        if is_test:
            return src, labels, segs, clss, src_txt
        else:
            return src, labels, segs, clss

    def batch_buffer(self, data, batch_size):
        # data（1个pt文件，9000+个文本数据）迭代每一个数据，追加到minibatch，直到总长度batch_size左右
        minibatch, size_so_far = [], 0
        for ex in data:
            if len(ex['src']) == 0:
                continue
            ex = self.preprocess(ex, self.is_test)
            if ex is None:
                continue
            minibatch.append(ex)
            # 以整个 minibatch最大长 * len(minibatch) 作为size_so_far，直到size_so_far>=batch_size(一般3000+那个size)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0

            # 超过了最后一个不反回，作为下一个batch的第一个
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch and len(minibatch) > 0:
            yield minibatch

    def create_batches(self):
        # 从self.datase（1个pt文件，9000+个文本数据）选一批数据，返回一batch
        if self.shuffle:
            random.shuffle(self.dataset)
        for buffer in self.batch_buffer(self.dataset, self.batch_size * 50):
            # buffer：从data中拿上千个数据，到函数batch()组成多个批次
            # 以句子数量排序
            p_batch = sorted(buffer, key=lambda x: len(x[3]))
            p_batch = batch(p_batch, self.batch_size)
            # 如果一个batch size 22，p_batch包含一批多个batch
            p_batch = list(p_batch)
            if self.shuffle:
                random.shuffle(p_batch)
            # 多个batch的p_batch，每次返回一个batch
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                # batch_buffer(self.dataset) > create_batches > minibatch
                batch = Batch(minibatch, self.device, self.is_test)
                yield batch
            return
