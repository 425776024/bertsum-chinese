#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import random
import torch
from transformers import BertConfig

from src.models import data_loader, model_builder_LAI
from src.models.data_loader import load_dataset
from src.models.model_builder_LAI import Summarizer
from src.models.trainer import build_trainer
from src.others.logging import logger, init_logger
from args_config import args

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def test(args, test_from, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    model.load_cp(checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.batch_size, device,
                                       shuffle=False, is_test=True)
    trainer = build_trainer(args, model, None)
    trainer.test(test_iter, step)


def train(args, device_id):
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        # 测试，不shuffle
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = Summarizer(args, device, load_pretrained_bert=True)
    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
        model.load_cp(checkpoint)
        optim = model_builder_LAI.build_optim(args, model, checkpoint)
    else:
        optim = model_builder_LAI.build_optim(args, model, None)
    logger.info('model load success............')
    logger.info(model)
    trainer = build_trainer(args, model, optim)
    trainer.train(train_iter_fct, args.train_steps)


if __name__ == '__main__':
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if args.mode == 'train':
        train(args, device_id)
    elif args.mode == 'test':
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test(args, args.test_from, step)
