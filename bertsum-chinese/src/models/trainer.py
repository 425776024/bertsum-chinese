# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from src.others.logging import logger
import src.others.utils as utils


def build_trainer(args, model, optim):
    trainer = Trainer(args, model, optim, args.accum_count)
    if model:
        n_params = utils.tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    def __init__(self, args, model, optim, grad_accum_count=1):
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        if model:
            self.model.train()

    def train(self, train_iter_fct, train_steps):
        logger.info('Start training...')
        step = self.optim._step + 1

        # 最终给的batch 是这个
        true_batchs = []
        # 训练了的batch（true_batchs） 次数
        accum = 0
        train_iter = train_iter_fct()

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                true_batchs.append(batch)
                accum += 1
                # true_batchs append的batch有grad_accum_count个时，开始给进去训练
                if accum == self.grad_accum_count:
                    reduce_counter += 1
                    # 训练
                    loss = self._gradient_accumulation(true_batchs)
                    if step % 2 == 0: print('step:', step, 'loss:', loss.cpu().detach().numpy())
                    true_batchs = []
                    accum = 0
                    if step % self.save_checkpoint_steps == 0:
                        self._save(step)
                    step += 1
            train_iter = train_iter_fct()
        if step > train_steps:
            self._save(step)

    def test(self, test_iter, step):
        self.model.eval()
        result_s = {'real_idx': [], 'predict_idx': [], 'src': []}
        save_path = self.args.result_path + '_step_' + str(step) + '.csv'
        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                # 从大到小
                selected_ids = np.argsort(-sent_scores, 1)

                for i, idx in enumerate(selected_ids):
                    _pred_idx = []
                    if len(batch.src_str[i]) == 0:
                        continue
                    for j in selected_ids[i][:len(batch.src_str[i])]:
                        if j >= len(batch.src_str[i]):
                            continue
                        # candidate = batch.src_str[i][j].strip()
                        _pred_idx.append(j)
                        if not self.args.recall_eval and len(_pred_idx) == 3:
                            break

                    result_s['src'].append('[SEP]'.join(batch.src_str[i]))
                    result_s['predict_idx'].append(utils.int_arr_to_str(_pred_idx))
                    label_idx = utils.label_to_idx(labels[i].tolist())
                    result_s['real_idx'].append(utils.int_arr_to_str(label_idx))

        save_df = pd.DataFrame()
        save_df['real_idx'] = result_s['real_idx']
        save_df['predict_idx'] = result_s['predict_idx']
        save_df['src'] = result_s['src']
        save_df.to_csv(save_path, sep='\t', index=False)

    def _gradient_accumulation(self, true_batchs):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            # .numel()：Returns the total number of elements in the input tensor.
            (loss / loss.numel()).backward()

            self.optim.step()
        return loss

    def _save(self, step):

        model_state_dict = self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }

        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path
