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
"""Example for building the Variational Autoencoder.

This is an impmentation of Variational Autoencoder for text generation

To run:

$ python vae_train.py

Hyperparameters and data path may be specified in config_trans.py

"""

# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, redefined-variable-type

import os
import sys
import time
import argparse
import importlib
from io import open

import numpy as np
import torch
import torch.nn as nn

import texar as tx
from texar.custom import MultivariateNormalDiag


parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default="config_model",
                    help="The config to use.")
parser.add_argument('--mode',
                    type=str,
                    default="train",
                    help="Train or predict.")
parser.add_argument('--model',
                    type=str,
                    default="train",
                    help="Model path for generating sentences.")
parser.add_argument('--out',
                    type=str,
                    default="train",
                    help="Generation output path.")

args = parser.parse_args()

config = importlib.import_module(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kl_dvg(means, logvars):
    """compute the KL divergence between Gaussian distribution
    """

    kl_cost = -0.5 * (logvars - means**2 -
                      torch.exp(logvars) + 1.0)
    kl_cost = torch.mean(kl_cost, 0)

    return torch.sum(kl_cost)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Vae(nn.Module):
    def __init__(self, train_data):
        super().__init__()
        # Model architecture
        self.encoder_w_embedder = tx.modules.WordEmbedder(
            vocab_size=train_data.vocab.size, hparams=config.enc_emb_hparams)

        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            input_size = self.encoder_w_embedder.dim,
            hparams={
                "rnn_cell": config.enc_cell_hparams,
            })

        self.decoder_w_embedder = tx.modules.WordEmbedder(
            vocab_size=train_data.vocab.size, hparams=config.dec_emb_hparams)

        if config.decoder_type == "lstm":
            self.decoder = tx.modules.BasicRNNDecoder(
                vocab_size=train_data.vocab.size,
                input_size = self.decoder_w_embedder.dim + config.batch_size,
                hparams={"rnn_cell": config.dec_cell_hparams})
            decoder_initial_state_size = (self.decoder.cell.hidden_size, self.decoder.cell.hidden_size)

        elif config.decoder_type == 'transformer':
            decoder_initial_state_size = torch.Size(
                [1, config.dec_emb_hparams["dim"]])
            # position embedding
            self.decoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
                position_size=config.max_pos, hparams=config.dec_pos_emb_hparams)
            # decoder
            self.decoder = tx.modules.TransformerDecoder(
                # tie word embedding with output layer
                output_layer=self.decoder_w_embedder.embedding,
                hparams=config.trans_hparams)
        else:
            raise NotImplementedError

        self.decoder_initial_state_size = decoder_initial_state_size

        self.connector_mlp = tx.modules.MLPTransformConnector(
            config.latent_dims * 2,
            linear_layer_dim=self.encoder.cell.hidden_size * 2)


        self.mlp_linear_layer = nn.Linear(
            32, tx.modules.connectors._sum_output_size(decoder_initial_state_size))

    def forward(self, data_batch, kl_weight):
        ## encoder -> connector -> decoder
        tmp_batch = []
        for i, data in enumerate(data_batch["text_ids"]):
            filter_data = data[data!=3]
            delta = data.size(0) - filter_data.size(0)
            tmp_batch.append(data[data!=3])
            data_batch["length"][i] -= delta
        text_ids = torch.stack(tmp_batch)
        input_embed = self.encoder_w_embedder(text_ids.to(device))
        output_w_embed = self.decoder_w_embedder(text_ids[:, :-1].to(device))
        _, ecdr_states = self.encoder(
            input_embed,
            sequence_length=data_batch["length"].to(device))

        if config.decoder_type == "lstm":
            output_embed = output_w_embed

        elif config.decoder_type == 'transformer':

            batch_size = text_ids.size(0)
            max_seq_len = text_ids.size(1) - 1
            batch_max_seq_len = torch.ones([batch_size], dtype=torch.long, device=device) * max_seq_len
            output_p_embed = self.decoder_p_embedder(sequence_length=batch_max_seq_len)
            output_w_embed = output_w_embed * config.hidden_size ** 0.5
            output_embed = output_w_embed + output_p_embed

        else:
            raise NotImplementedError
        mean_logvar = self.connector_mlp(ecdr_states)
        mean, logvar = torch.chunk(mean_logvar, 2, 1)
        kl_loss = kl_dvg(mean, logvar)

        dst = MultivariateNormalDiag(
            loc=mean,
            scale_diag=torch.exp(0.5 * logvar))

        latent_z = dst.rsample()
        dcdr_states = tx.modules.connectors._mlp_transform(
                latent_z,
                self.decoder_initial_state_size,
                self.mlp_linear_layer)

        # decoder
        if config.decoder_type == "lstm":
            # concat latent variable to input at every time step
            latent_z = torch.unsqueeze(latent_z, 1)

            latent_z = latent_z.repeat([1, output_embed.size(1), 1])
            output_embed = torch.cat([output_embed, latent_z], 2)

            train_helper = self.decoder.create_helper(decoding_strategy='train_greedy')
            outputs, _, _ = self.decoder(
                initial_state=dcdr_states,
                helper=train_helper,
                inputs=output_embed,
                sequence_length=data_batch["length"]-1)
        else:
            outputs = self.decoder(
                inputs=output_embed,
                memory=dcdr_states,
                memory_sequence_length=torch.ones(dcdr_states.size(0)))

        logits = outputs.logits

        seq_lengths = data_batch["length"] - 1
        # Losses & train ops
        rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=text_ids[:, 1:].to(device),
            logits=logits,
            sequence_length=(data_batch["length"]-1).to(device))

        nll = rc_loss + kl_weight * kl_loss

        fetches = {
            "nll": nll,
            "kl_loss": kl_loss.detach(),
            "rc_loss": rc_loss.detach(),
            "lengths": seq_lengths.detach()
        }

        return fetches


def main():
    # Data
    train_data = tx.data.MonoTextData(config.train_data_hparams,
                                      device=device)
    val_data = tx.data.MonoTextData(config.val_data_hparams,
                                    device=device)
    test_data = tx.data.MonoTextData(config.test_data_hparams,
                                     device=device)

    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)

    opt_vars = {
        'learning_rate': config.lr_decay_hparams["init_lr"],
        'best_valid_nll': 1e100,
        'steps_not_improved': 0,
        'kl_weight': config.kl_anneal_hparams["start"]
    }

    decay_cnt = 0
    max_decay = config.lr_decay_hparams["max_decay"]
    decay_factor = config.lr_decay_hparams["decay_factor"]
    decay_ts = config.lr_decay_hparams["threshold"]

    save_dir = "./models/%s" % config.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    suffix = "%s_%sDecoder.ckpt" % \
            (config.dataset, config.decoder_type)

    save_path = os.path.join(save_dir, suffix)

    # KL term annealing rate
    anneal_r = 1.0 / (config.kl_anneal_hparams["warm_up"] * \
        (train_data.__len__() / config.batch_size))
    
    model = Vae(train_data)
    model.to(device)

    optimizer = tx.core.get_optimizer(params=model.parameters(),
                                    hparams=config.opt_hparams)

    def _run_epoch(epoch, mode, display=10):
        if mode == 'train':
            iterator.switch_to_train_data()
        elif mode == 'valid':
            iterator.switch_to_val_data()
        elif mode == 'test':
            iterator.switch_to_test_data()
        if mode == 'train':
            model.train()
        else:
            model.eval()
        step = 0
        start_time = time.time()
        num_words = num_sents = 0
        nll_ = 0.
        kl_loss_ = rc_loss_ = 0.

        for batch in iterator:
            if mode == 'train':
                opt_vars["kl_weight"] = min(
                    1.0, opt_vars["kl_weight"] + anneal_r)

                kl_weight_ = opt_vars["kl_weight"]
            else:
                kl_weight_ = 1.0

            mode = mode

            if mode == "train":
                adjust_learning_rate(optimizer, opt_vars["learning_rate"])

            fetches_ = model(batch, kl_weight_)
            if mode == "train":
                fetches_["nll"].backward()
                #train_op()
                optimizer.step()
                optimizer.zero_grad()

            batch_size_ = len(fetches_["lengths"])
            num_sents += batch_size_
            num_words += sum(fetches_["lengths"])
            nll_ += fetches_["nll"].item() * batch_size_
            kl_loss_ += fetches_["kl_loss"] * batch_size_
            rc_loss_ += fetches_["rc_loss"] * batch_size_

            if step % display == 0 and mode == 'train':
                nll = nll_ / float(num_sents)
                klw = opt_vars["kl_weight"]
                KL = kl_loss_ / num_sents
                rc = rc_loss_ / num_sents
                log_ppl = nll_ / float(num_words)
                ppl = np.exp(log_ppl)
                time_cost = time.time() - start_time

                print(f"{mode}: epoch {epoch}, step {step}, nll {nll:.4f}, " 
                      f"klw {klw:.4f}, KL {KL:.4f}, rc {rc:.4f}, log_ppl {log_ppl:.4f}, "
                      f"ppl {ppl:.4f}, time_cost {time_cost}")
                sys.stdout.flush()

            step += 1

        nll = nll_ / float(num_sents)
        KL = kl_loss_ / num_sents
        rc = rc_loss_ / num_sents
        log_ppl = nll_ / float(num_words)
        ppl = np.exp(log_ppl)
        print(f"\n{mode}: epoch {epoch}, nll {nll:.4f}, KL {KL:.4f}, rc {rc:.4f}, log_ppl {log_ppl:.4f}, ppl {ppl:.4f}")
        return nll, ppl

    @torch.no_grad()
    def _generate(fname=None):
        ckpt = torch.load(args.model)
        model.load_state_dict(ckpt['model'])
        model.eval()

        batch_size = train_data.batch_size

        dst = MultivariateNormalDiag(
            loc=torch.zeros([batch_size, config.latent_dims]),
            scale_diag=torch.ones([batch_size, config.latent_dims]))

        latent_z = dst.rsample()
        dcdr_states = tx.modules.connectors._mlp_transform(
            latent_z.to(device),
            model.decoder_initial_state_size,
            model.mlp_linear_layer)

        vocab = train_data.vocab
        start_tokens = torch.ones(batch_size).type(torch.long) * vocab.bos_token_id
        end_token = vocab.eos_token_id.item()

        if config.decoder_type == "lstm":
            def _cat_embedder(ids):
                """Concatenates latent variable to input word embeddings
                """
                embedding = model.decoder_w_embedder(ids.to(device))
                return torch.cat([embedding, latent_z.to(device)], 1)

            infer_helper = model.decoder.create_helper(
                embedding=_cat_embedder,
                decoding_strategy='infer_sample',
                start_tokens=start_tokens.to(device),
                end_token=end_token)

            outputs, _, _ = model.decoder(
                initial_state=dcdr_states,
                helper=infer_helper,
                max_decoding_length=100)
        else:
            def _embedding_fn(ids, times):
                w_embed = model.decoder_w_embedder(ids)
                p_embed = model.decoder_p_embedder(times)
                return w_embed * config.hidden_size ** 0.5 + p_embed

            outputs, _ = model.decoder(
                memory=dcdr_states,
                memory_sequence_length=torch.ones(dcdr_states.size(0)),
                max_decoding_length=100,
                decoding_strategy='infer_sample',
                embedding=_embedding_fn,
                start_tokens=start_tokens.to(device),
                end_token=end_token)

        sample_tokens = vocab.map_ids_to_tokens_py(outputs.sample_id.cpu())

        sample_tokens_ = sample_tokens

        if fname is None:
            fh = sys.stdout
        else:
            fh = open(fname, 'w', encoding='utf-8')

        for sent in sample_tokens_:
            sent = tx.utils.compat_as_text(list(sent))
            end_id = len(sent)
            if vocab.eos_token in sent:
                end_id = sent.index(vocab.eos_token)
            fh.write(' '.join(sent[:end_id+1]) + '\n')

        print('Output done')
        fh.close()


    if args.mode == "predict":
        _generate(args.out)
        return
    # Counts trainable parameters
    total_parameters = 0
    for name, variable in model.named_parameters():
        size = variable.size()
        total_parameters += np.prod(size)
    print(f"{total_parameters} total parameters")

    best_nll = best_ppl = 0.

    for epoch in range(config.num_epochs):
        _, _ = _run_epoch(epoch, 'train', display=200)
        val_nll, _ = _run_epoch(epoch, 'valid')
        test_nll, test_ppl = _run_epoch(epoch, 'test')

        if val_nll < opt_vars['best_valid_nll']:
            opt_vars['best_valid_nll'] = val_nll
            opt_vars['steps_not_improved'] = 0
            best_nll = test_nll
            best_ppl = test_ppl

            states = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(states, save_path)
        else:
            opt_vars['steps_not_improved'] += 1
            if opt_vars['steps_not_improved'] == decay_ts:
                old_lr = opt_vars['learning_rate']
                opt_vars['learning_rate'] *= decay_factor
                opt_vars['steps_not_improved'] = 0
                new_lr = opt_vars['learning_rate']

                print(f"-----\nchange lr, old lr: {old_lr}, " \
                      f"new lr: {new_lr}\n-----")

                ckpt = torch.load(save_path)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                decay_cnt += 1
                if decay_cnt == max_decay:
                    break

    print('\nbest testing nll: %.4f, best testing ppl %.4f\n' %
        (best_nll, best_ppl))


if __name__ == '__main__':
    main()
