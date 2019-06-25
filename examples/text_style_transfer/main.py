# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text style transfer

This is a simplified implementation of:

Toward Controlled Generation of Text, ICML2017
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing

Download the data with the cmd:

$ python prepare_data.py

Train the model with the cmd:

$ python main.py --config config
"""

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
import importlib
import numpy as np
import torch
#import tensorflow as tf
import texar as tx
from texar.core.optimization import get_optimizer, get_train_op

import argparse

from ctrl_gen_model import CtrlGenModel

import traceback


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", default="config",
    help="The config to use.")

args = parser.parse_args()

config = importlib.import_module(args.config)

def main():
    # Data
    train_data = tx.data.MultiAlignedData(config.train_data)
    val_data = tx.data.MultiAlignedData(config.val_data)
    test_data = tx.data.MultiAlignedData(config.test_data)
    vocab = train_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.

    iterator = tx.data.DataIterator(
        {'train_g': train_data, 'train_d': train_data,
         'val': val_data, 'test': test_data}
    )

    # Model
    model = CtrlGenModel(vocab, config.model)

    vars_g = []
    vars_d = []
    for name, param in model.named_parameters():
        if not name.startswith("clas"):
            vars_g.append(param)
        else:
            vars_d.append(param)
    optimizer_g = get_optimizer(vars_g, config.model["opt"])
    train_op_g = get_train_op(optimizer_g, config.model["opt"])
    optimizer_d = get_optimizer(vars_d, config.model["opt"])
    train_op_d = get_train_op(optimizer_d, config.model["opt"])

    def _train_epoch(gamma_, lambda_g_, epoch, verbose=True):
        model.train()

        step = 0
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)

        it1 = iterator.get_iterator("train_d")
        it2 = iterator.get_iterator("train_g")

        while True:
            try:
                step += 1
                print("step: {}".format(step))
                optimizer_d.zero_grad()
                batch_1 = it1.next()._batch

                vals_d, _, _= model(batch_1, gamma_, lambda_g_)
                for key, val in vals_d.items():
                    if key.startswith("loss"):
                        val.backward(retain_graph=True)

                optimizer_d.step()
                avg_meters_d.add(vals_d)
                optimizer_g.zero_grad()
                batch_2 = it2.next()._batch
                _, vals_g, _ = model(batch_2, gamma_, lambda_g_)
                for key, val in vals_g.items():
                    if key.startswith("loss"):
                        val.backward(retain_graph=True)

                #train_op_g()
                optimizer_g.step()
                avg_meters_g.add(vals_g)

                if verbose and step % config.display_eval == 0:
                    _eval_epoch(gamma_, lambda_g_, epoch)

            except:
                print('epoch: {}, {}'.format(epoch, avg_meters_d.to_str(4)))
                print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                traceback.print_exc()
                break

    def _eval_epoch(gamma_, lambda_g_, epoch, val_or_test='val'):
        avg_meters = tx.utils.AverageRecorder()
        it = iterator.get_iterator(val_or_test)
        model.eval()

        while True:
            try:
                batch = it.next()._batch
                _, _, vals = model(batch, gamma_, lambda_g_)
                batch_size = vals.pop('batch_size')

                 # Computes BLEU
                samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)

                refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                refs = np.expand_dims(refs, axis=1)

                bleu = tx.evals.corpus_bleu_moses(refs, hyps)
                vals['bleu'] = bleu

                avg_meters.add(vals, weight=batch_size)

                # Writes samples
                tx.utils.write_paired_text(
                    refs.squeeze(), hyps,
                    os.path.join(config.sample_path, 'val.%d'%epoch),
                    append=True, mode='v')
            except:
                print('{}: {}'.format(
                    val_or_test, avg_meters.to_str(precision=4)))
                traceback.print_exc()
                break
        return avg_meters.avg()

    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    # Runs the logics

    gamma_ = 1.
    lambda_g_ = 0.

    for epoch in range(1, config.max_nepochs+1):
        if epoch > config.pretrain_nepochs:
            # Anneals the gumbel-softmax temperature
            gamma_ = max(0.001, gamma_ * config.gamma_decay)
            lambda_g_ = config.lambda_g
        print('gamma: {}, lambda_g: {}'.format(gamma_, lambda_g_))
        _train_epoch(gamma_, lambda_g_, epoch)
    
        _eval_epoch(gamma_, lambda_g_, epoch, 'val')
        states = {
            'model': model.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
        }
        torch.save(states, os.path.join(args.output_dir + '/model.ckpt'))
        _eval_epoch(sess, gamma_, lambda_g_, epoch, 'test')

if __name__ == '__main__':
    main()
