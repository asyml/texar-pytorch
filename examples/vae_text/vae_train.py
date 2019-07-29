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


import os
import sys
import time
import argparse
import importlib
from typing import Optional, List, Any, Union, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

import texar as tx
from texar.modules import BasicRNNDecoderOutput, TransformerDecoderOutput
from texar.custom import MultivariateNormalDiag
from texar.modules.decoders.decoder_helpers import Helper


parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default=None,
                    help="The config to use.")
parser.add_argument('--mode',
                    type=str,
                    default='train',
                    help="Train or predict.")
parser.add_argument('--model',
                    type=str,
                    default=None,
                    help="Model path for generating sentences.")
parser.add_argument('--out',
                    type=str,
                    default=None,
                    help="Generation output path.")

args = parser.parse_args()


def kl_divergence(means: Tensor, logvars: Tensor) -> Tensor:
    """Compute the KL divergence between Gaussian distribution
    """

    kl_cost = -0.5 * (logvars - means**2 -
                      torch.exp(logvars) + 1.0)
    kl_cost = torch.mean(kl_cost, 0)

    return torch.sum(kl_cost)


class VAE(nn.Module):
    _latent_z: Tensor

    def __init__(self,
                 vocab_size: int, config_model):
        super().__init__()
        # Model architecture
        self._config = config_model
        self.encoder_w_embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=config_model.enc_emb_hparams)

        self.encoder = tx.modules.UnidirectionalRNNEncoder(
            input_size=self.encoder_w_embedder.dim,
            hparams={
                "rnn_cell": config_model.enc_cell_hparams,
            })

        self.decoder_w_embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=config_model.dec_emb_hparams)

        if config_model.decoder_type == "lstm":
            self.decoder = tx.modules.BasicRNNDecoder(
                input_size=(self.decoder_w_embedder.dim +
                    config_model.batch_size),
                vocab_size=vocab_size,
                token_embedder=self._embed_fn_rnn,
                hparams={"rnn_cell": config_model.dec_cell_hparams})
            decoder_initial_state_size = (self.decoder.cell.hidden_size,) * 2
            sum_state_size = self.decoder.cell.hidden_size * 2

        elif config_model.decoder_type == 'transformer':
            decoder_initial_state_size = torch.Size(
                [1, config_model.dec_emb_hparams["dim"]])
            sum_state_size = config_model.dec_emb_hparams["dim"]
            # position embedding
            self.decoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
                position_size=config_model.max_pos,
                hparams=config_model.dec_pos_emb_hparams)
            # decoder
            self.decoder = tx.modules.TransformerDecoder(
                # tie word embedding with output layer
                output_layer=self.decoder_w_embedder.embedding,
                token_pos_embedder=self._embed_fn_transformer,
                hparams=config_model.trans_hparams)
        else:
            raise ValueError("Decoder type must be 'lstm' or 'transformer'")

        self.decoder_initial_state_size = decoder_initial_state_size

        self.connector_mlp = tx.modules.MLPTransformConnector(
            config_model.latent_dims * 2,
            linear_layer_dim=self.encoder.cell.hidden_size * 2)

        self.mlp_linear_layer = nn.Linear(
            config_model.latent_dims,
            sum_state_size)

    def forward(self, data_batch: tx.data.Batch,
                kl_weight: float, start_tokens: torch.LongTensor,
                end_token: int) \
            -> Dict:
        # encoder -> connector -> decoder
        text_ids = data_batch["text_ids"]
        input_embed = self.encoder_w_embedder(text_ids)
        _, encoder_states = self.encoder(
            input_embed,
            sequence_length=data_batch["length"])

        mean_logvar = self.connector_mlp(encoder_states)
        mean, logvar = torch.chunk(mean_logvar, 2, 1)
        kl_loss = kl_divergence(mean, logvar)

        dst = MultivariateNormalDiag(
            loc=mean,
            scale_diag=torch.exp(0.5 * logvar))

        latent_z = dst.rsample()

        fc_output = self.mlp_linear_layer(latent_z)
        labels = text_ids[:, 1:]
        if self._config.decoder_type == "lstm":
            decoder_states = torch.split(
                fc_output, self.decoder_initial_state_size, dim=1)

            helper = self.decoder.create_helper(
                decoding_strategy="train_greedy",
                start_tokens=start_tokens,
                end_token=end_token,)
            latent_z = torch.unsqueeze(latent_z, 0)
            latent_z = latent_z.repeat([text_ids.size(1), 1, 1])
        else:
            decoder_states = torch.reshape(
                fc_output, (-1, ) + self.decoder_initial_state_size)
            helper = None
            text_ids = text_ids[:, :-1]

        seq_lengths = data_batch["length"] - 1
        # decode
        outputs = self.decode(
            decoder_states=decoder_states,
            helper=helper,
            latent_z=latent_z,
            seq_lengths=seq_lengths,
            text_ids=text_ids,
            )

        logits = outputs.logits

        # Losses & train ops
        rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits,
            sequence_length=seq_lengths)

        nll = rc_loss + kl_weight * kl_loss

        ret = {
            "nll": nll,
            "kl_loss": kl_loss,
            "rc_loss": rc_loss,
            "lengths": seq_lengths
        }

        return ret

    def _embed_fn_rnn(self, tokens: torch.LongTensor) -> Tensor:
        r"""Generates word embeddings
        """
        embedding = self.decoder_w_embedder(
            tokens)
        latent_z = self._latent_z
        return torch.cat([embedding, latent_z], dim=-1)

    def _embed_fn_transformer(self,
                              tokens: torch.LongTensor,
                              positions: torch.LongTensor) -> Tensor:
        r"""Generates word embeddings combined with positional embeddings
        """
        output_p_embed = self.decoder_p_embedder(positions)
        output_w_embed = self.decoder_w_embedder(tokens)
        output_w_embed = output_w_embed * \
            self._config.hidden_size ** 0.5
        output_embed = output_w_embed + output_p_embed
        return output_embed

    def decode(self,
               decoder_states: Union[Tensor, Tuple],
               helper: Optional[Helper],
               latent_z: Optional[Tensor] = None,
               seq_lengths: Optional[Tensor] = None,
               max_decoding_length: int = None,
               text_ids: torch.LongTensor = None) \
            -> Union[BasicRNNDecoderOutput, TransformerDecoderOutput]:

        self._latent_z = latent_z
        if self._config.decoder_type == "lstm":
            outputs, _, _ = self.decoder(
                initial_state=decoder_states,
                inputs=text_ids,
                helper=helper,
                sequence_length=seq_lengths,
                max_decoding_length=max_decoding_length)
        else:
            outputs = self.decoder(
                inputs=text_ids,
                memory=decoder_states,
                memory_sequence_length=torch.ones(decoder_states.size(0)),
                helper=helper,
                max_decoding_length=max_decoding_length)
        return outputs


class VAEData(tx.data.MonoTextData):
    r""" Wrap `tx.data.MonoTextData` to implement data preprocessing.
    """
    def process(self, raw_example: List[str]) -> List[str]:
        # Truncates sentences and appends BOS/EOS tokens.
        raw_example = super().process(raw_example)
        words = " ".join(raw_example[1:-1]).strip().split(self._delimiter)
        words = [raw_example[0]] + words + [raw_example[-1]]
        return words


def main():
    """Entrypoint.
    """
    config: Any = importlib.import_module(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = VAEData(config.train_data_hparams,
                                      device=device)
    val_data = VAEData(config.val_data_hparams,
                                    device=device)
    test_data = VAEData(config.test_data_hparams,
                                     device=device)

    iterator = tx.data.DataIterator(
        {"train": train_data, "valid": val_data, "test": test_data})

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

    save_dir = f"./models/{config.dataset}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    suffix = f"{config.dataset}_{config.decoder_type}Decoder.ckpt"

    save_path = os.path.join(save_dir, suffix)

    # KL term annealing rate
    anneal_r = 1.0 / (config.kl_anneal_hparams["warm_up"] *
        (len(train_data) / config.batch_size))

    vocab = train_data.vocab
    model = VAE(train_data.vocab.size, config)
    model.to(device)

    start_tokens = torch.full(
            (config.batch_size,),
            vocab.bos_token_id,
            dtype=torch.long).to(device)
    end_token = vocab.eos_token_id
    optimizer = tx.core.get_optimizer(
        params=model.parameters(),
        hparams=config.opt_hparams)
    scheduler = ExponentialLR(optimizer, decay_factor)

    def _run_epoch(epoch: int, mode: str, display: int = 10) \
            -> Tuple[Tensor, float]:
        iterator.switch_to_dataset(mode)

        if mode == 'train':
            model.train()
            opt_vars["kl_weight"] = min(
                    1.0, opt_vars["kl_weight"] + anneal_r)

            kl_weight = opt_vars["kl_weight"]
        else:
            model.eval()
            kl_weight = 1.0
        step = 0
        start_time = time.time()
        num_words = 0
        nll_total = 0.

        avg_rec = tx.utils.AverageRecorder()
        for batch in iterator:

            ret = model(batch, kl_weight, start_tokens, end_token)
            if mode == "train":
                opt_vars["kl_weight"] = min(
                    1.0, opt_vars["kl_weight"] + anneal_r)
                kl_weight = opt_vars["kl_weight"]
                ret["nll"].backward()
                optimizer.step()
                optimizer.zero_grad()

            batch_size = len(ret["lengths"])
            num_words += torch.sum(ret["lengths"]).item()
            nll_total += ret["nll"].item() * batch_size
            avg_rec.add(
                [ret["nll"].item(),
                 ret["kl_loss"].item(),
                 ret["rc_loss"].item()],
                batch_size)
            if step % display == 0 and mode == 'train':
                nll = avg_rec.avg(0)
                klw = opt_vars["kl_weight"]
                KL = avg_rec.avg(1)
                rc = avg_rec.avg(2)
                log_ppl = nll_total / num_words
                ppl = np.exp(log_ppl)
                time_cost = time.time() - start_time

                print(f"{mode}: epoch {epoch}, step {step}, nll {nll:.4f}, "
                      f"klw {klw:.4f}, KL {KL:.4f}, rc {rc:.4f}, "
                      f"log_ppl {log_ppl:.4f}, ppl {ppl:.4f}, "
                      f"time_cost {time_cost:.1f}", flush=True)

            step += 1

        nll = avg_rec.avg(0)
        KL = avg_rec.avg(1)
        rc = avg_rec.avg(2)
        log_ppl = nll_total / num_words
        ppl = np.exp(log_ppl)
        print(f"\n{mode}: epoch {epoch}, nll {nll:.4f}, KL {KL:.4f}, "
              f"rc {rc:.4f}, log_ppl {log_ppl:.4f}, ppl {ppl:.4f}")
        return nll, ppl

    @torch.no_grad()
    def _generate(start_tokens: torch.LongTensor,
                  end_token: int,
                  filename: Optional[str] = None):
        ckpt = torch.load(args.model)
        model.load_state_dict(ckpt['model'])
        model.eval()

        batch_size = train_data.batch_size

        dst = MultivariateNormalDiag(
            loc=torch.zeros([batch_size, config.latent_dims]),
            scale_diag=torch.ones([batch_size, config.latent_dims]))

        latent_z = dst.rsample().to(device)

        fc_output = model.mlp_linear_layer(latent_z)
        if not isinstance(model.decoder_initial_state_size, torch.Size):
            decoder_states = torch.split(
                fc_output, model.decoder_initial_state_size, dim=1)
        else:
            decoder_states = torch.reshape(
                fc_output, (-1, ) + model.decoder_initial_state_size)

        if config.decoder_type == "lstm":

            helper = model.decoder.create_helper(
                decoding_strategy='infer_sample',
                start_tokens=start_tokens,
                end_token=end_token)
            outputs = model.decode(
                decoder_states=decoder_states,
                helper=helper,
                latent_z=latent_z,
                max_decoding_length=100)
        else:

            helper = model.decoder.create_helper(
                decoding_strategy='infer_sample',
                start_tokens=start_tokens,
                end_token=end_token)
            outputs, _ = model.decode(
                decoder_states=decoder_states,
                helper=helper,
                max_decoding_length=100,
                seq_lengths=None)

        sample_tokens = vocab.map_ids_to_tokens_py(outputs.sample_id.cpu())

        if filename is None:
            fh = sys.stdout
        else:
            fh = open(filename, 'w', encoding='utf-8')

        for sent in sample_tokens:
            sent = tx.utils.compat_as_text(list(sent))
            end_id = len(sent)
            if vocab.eos_token in sent:
                end_id = sent.index(vocab.eos_token)
            fh.write(' '.join(sent[:end_id + 1]) + '\n')

        print('Output done')
        fh.close()

    if args.mode == "predict":
        _generate(start_tokens, end_token, args.out)
        return
    # Counts trainable parameters
    total_parameters = 0
    for _, variable in model.named_parameters():
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
                "scheduler": scheduler.state_dict()
            }
            torch.save(states, save_path)
        else:
            opt_vars['steps_not_improved'] += 1
            if opt_vars['steps_not_improved'] == decay_ts:
                old_lr = opt_vars['learning_rate']
                opt_vars['learning_rate'] *= decay_factor
                opt_vars['steps_not_improved'] = 0
                new_lr = opt_vars['learning_rate']
                ckpt = torch.load(save_path)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                scheduler.step()
                print(f"-----\nchange lr, old lr: {old_lr}, "
                      f"new lr: {new_lr}\n-----")

                decay_cnt += 1
                if decay_cnt == max_decay:
                    break

    print('\nbest testing nll: %.4f, best testing ppl %.4f\n' %
        (best_nll, best_ppl))


if __name__ == '__main__':
    main()
