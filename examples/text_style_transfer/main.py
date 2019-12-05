# Copyright 2018 The Texar Authors. All Rights Reserved.
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
import argparse
import texar.torch as tx

from ctrl_gen_model import CtrlGenModel

parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config', help="The config to use.")

args = parser.parse_args()

config = importlib.import_module(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _main():
    # Data
    train_data = tx.data.MultiAlignedData(hparams=config.train_data, device=device)
    val_data = tx.data.MultiAlignedData(hparams=config.val_data, device=device)
    test_data = tx.data.MultiAlignedData(hparams=config.test_data, device=device)
    vocab = train_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.DataIterator(
        {'train_g': train_data, 'train_d': train_data,
         'val': val_data, 'test': test_data})

    # Model
    gamma_ = 1.
    lambda_g_ = 0.

    # Model
    model = CtrlGenModel(vocab, hparams=config.model)
    model.to(device)

    # create optimizers
    train_op_d = tx.core.get_optimizer(
        params=model.d_vars,
        hparams=config.model['opt']
    )

    train_op_g = tx.core.get_optimizer(
        params=model.g_vars,
        hparams=config.model['opt']
    )

    train_op_g_ae = tx.core.get_optimizer(
        params=model.g_vars,
        hparams=config.model['opt']
    )

    def _train_epoch(gamma_, lambda_g_, epoch, verbose=True):
        model.train()
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        iterator.switch_to_dataset("train_g")
        step = 0
        for batch in iterator:
            train_op_d.zero_grad()
            train_op_g_ae.zero_grad()
            train_op_g.zero_grad()
            step += 1

            vals_d = model(batch, gamma_, lambda_g_, mode="train", component="D")
            loss_d = vals_d['loss_d']
            loss_d.backward()
            train_op_d.step()
            recorder_d = {key: value.detach().cpu().data for (key, value) in vals_d.items()}
            avg_meters_d.add(recorder_d)

            vals_g = model(batch, gamma_, lambda_g_, mode="train", component="G")
            loss_g = vals_g['loss_g']
            loss_g_ae = vals_g['loss_g_ae']
            loss_g_ae.backward(retain_graph=True)
            loss_g.backward()
            train_op_g_ae.step()
            train_op_g.step()

            recorder_g = {key: value.detach().cpu().data for (key, value) in vals_g.items()}
            avg_meters_g.add(recorder_g)

            if verbose and (step == 1 or step % config.display == 0):
                print('step: {}, {}'.format(step, avg_meters_d.to_str(4)))
                print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))

            if verbose and step % config.display_eval == 0:
                iterator.switch_to_dataset("val")
                _eval_epoch(gamma_, lambda_g_, epoch)

        print('epoch: {}, {}'.format(epoch, avg_meters_d.to_str(4)))
        print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))

    @torch.no_grad()
    def _eval_epoch(gamma_, lambda_g_, epoch, val_or_test='val'):
        model.eval()
        avg_meters = tx.utils.AverageRecorder()
        iterator.switch_to_dataset(val_or_test)
        for batch in iterator:
            vals, samples = model(batch, gamma_, lambda_g_, mode='eval')

            batch_size = vals.pop('batch_size')

            # Computes BLEU
            hyps = tx.data.map_ids_to_strs(samples['transferred'].cpu(), vocab)

            refs = tx.data.map_ids_to_strs(samples['original'].cpu(), vocab)
            refs = np.expand_dims(refs, axis=1)

            bleu = tx.evals.corpus_bleu_moses(refs, hyps)
            vals['bleu'] = bleu

            avg_meters.add(vals, weight=batch_size)

            # Writes samples
            tx.utils.write_paired_text(
                refs.squeeze(), hyps,
                os.path.join(config.sample_path, 'val.%d' % epoch),
                append=True, mode='v')

        print('{}: {}'.format(
            val_or_test, avg_meters.to_str(precision=4)))

        return avg_meters.avg()

    os.makedirs(config.sample_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Runs the logics
    if config.restore:
        print('Restore from: {}'.format(config.restore))
        ckpt = torch.load(args.restore)
        model.load_state_dict(ckpt['model'])
        train_op_d.load_state_dict(ckpt['optimizer_d'])
        train_op_g.load_state_dict(ckpt['optimizer_g'])

    for epoch in range(1, config.max_nepochs + 1):
        if epoch > config.pretrain_nepochs:
            # Anneals the gumbel-softmax temperature
            gamma_ = max(0.001, gamma_ * config.gamma_decay)
            lambda_g_ = config.lambda_g
        print('gamma: {}, lambda_g: {}'.format(gamma_, lambda_g_))

        # Train
        _train_epoch(gamma_, lambda_g_, epoch)

        # Val
        _eval_epoch(gamma_, lambda_g_, epoch, 'val')

        states = {
            'model': model.state_dict(),
            'optimizer_d': train_op_d.state_dict(),
            'optimizer_g': train_op_g.state_dict()
        }
        torch.save(states, os.path.join(config.checkpoint_path, 'ckpt'))

        # Test
        _eval_epoch(gamma_, lambda_g_, epoch, 'test')


if __name__ == '__main__':
    _main()
