# Transformer for Machine Translation #

This is an implementation of the Transformer model described in
[Vaswani, Ashish, et al. "Attention is all you need."](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).

[Quick Start](https://github.com/asyml/texar/tree/master/examples/transformer#quick-start):
Prerequisites & use on machine translation datasets.

[Run Your Customized Experiments](https://github.com/asyml/texar/tree/master/examples/transformer#run-your-customized-experiments):
Hands-on tutorial of data preparation, configuration, and model training/testing.

## Quick Start ##

### Prerequisites ###

Run the following command to install necessary packages for the example: 

```bash
pip install -r requirements.txt
```

### Datasets ###

Two example datasets are provided:

- **IWSLT'15 EN-VI** for English-Vietnamese translation
- **WMT'14 EN-DE** for English-German translation

Download and pre-process the **IWSLT'15 EN-VI** data with the following commands: 

```bash
sh scripts/iwslt15_en_vi.sh
sh preprocess_data.sh spm en vi
```
By default, the downloaded dataset is in `./data/en_vi`. 
As with the [official implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py),
`spm` (`sentencepiece`) encoding is used to encode the raw text as data pre-processing. The encoded data is by default
in `./temp/run_en_vi_spm`. 

For the **WMT'14 EN-DE** data, download and pre-process with:

```bash
sh scripts/wmt14_en_de.sh
sh preprocess_data.sh bpe en de
```

Note that this is a large dataset and preprocessing requires large amounts of memory.

By default, the downloaded dataset is in `./data/en_de`. Note that for this dataset, `bpe` encoding (Byte pair encoding)
is used instead. The encoded data is by default in `./temp/run_en_de_bpe`. 

### Train and evaluate the model ###

Train the model with the command:

```bash
python transformer_main.py \
    --run-mode=train_and_evaluate \
    --config-model=config_model \
    --config-data=config_iwslt15
```
* Specify `--output-dir` to dump model checkpoints and training logs to a desired directory.
  By default it is set to `./outputs`. 
* Specifying `--output-dir` will also restore the latest model checkpoint under the directory, if any checkpoint exists.
* Specify `--config-data=config_wmt14` to train on the WMT'14 data.
* Additionally, you can also specify `--load-checkpoint` to load a previously trained checkpoint from `output_dir`.

### Test a trained model ###

To only evaluate a model checkpoint without training, first load the checkpoint and generate samples: 

```bash
python transformer_main.py \
    --run-mode=test \
    --config-data=config_iwslt15 \
    --output-dir=./outputs
```
The latest checkpoint in `./outputs` is used. Generated samples are in the file `./outputs/test.output.hyp`, and
reference sentences are in the file `./outputs/test.output.ref`. The script shows the cased BLEU score as provided by
the [`tx.evals.file_bleu`](https://texar-pytorch.readthedocs.io/en/latest/code/evals.html#file-bleu) function. 

Alternatively, you can also compute the BLEU score with the raw sentences using the `bleu_main` script:

```bash
python bleu_main.py --reference=data/en_vi/test.vi --translation=temp/test.output.hyp
```

### Results

* On **IWSLT'15**, the implementation achieves around `BLEU_cased=29.05` and `BLEU_uncased=29.94` (reported by
  [bleu_main.py](./bleu_main.py)), which are comparable to the base_single_gpu results by the
  [official implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
  (`28.12` and `28.97`, respectively, as reported [here](https://github.com/tensorflow/tensor2tensor/pull/611)).

* On **WMT'14**, the implementation achieves around `BLEU_cased=25.02` following the setting in `config_wmt14.py`
  (setting: base_single_gpu, batch_size=3072). It takes more than 18 hours to finish training 250k steps. You can
  modify `max_train_epoch` in `config_wmt14.py` to adjust the training time.

### Example training log

```
INFO 2019-08-15 22:04:15 : Begin running with train_and_evaluate mode
WARNING 2019-08-15 22:04:15 : Specified checkpoint directory 'outputs' exists, previous checkpoints might be erased
INFO 2019-08-15 22:04:15 : Training started
INFO 2019-08-15 22:04:15 : Model architecture:
ModelWrapper(
  (model): Transformer(
    ...
  )
)
2019-08-15 22:05:51 : Epoch  1 @    500it (13.0%, 172.63ex/s), lr = 2.184e-05, loss = 7.497
2019-08-15 22:07:27 : Epoch  1 @   1000it (26.0%, 172.91ex/s), lr = 4.367e-05, loss = 6.784
2019-08-15 22:09:03 : Epoch  1 @   1500it (39.0%, 172.52ex/s), lr = 6.551e-05, loss = 6.365
2019-08-15 22:10:40 : Epoch  1 @   2000it (51.9%, 172.03ex/s), lr = 8.735e-05, loss = 5.847
2019-08-15 22:15:50 : Epoch 1, valid BLEU = 2.075
INFO 2019-08-15 22:15:54 : Current checkpoint saved to outputs/1565921750.7879117.pt

```
Using an NVIDIA GTX 1080Ti, the model usually converges within 5 hours (~15 epochs) on **IWSLT'15**.

---

## Run Your Customized Experiments

Here is a hands-on tutorial on running Transformer with your own customized dataset.

### 1. Prepare raw data

Create a data directory and put the raw data in the directory. To be compatible with the data preprocessing in the next
step, you may follow the convention below:

* The data directory should be named as `data/${src}_${tgt}/`. Take the data downloaded with `scripts/iwslt15_en_vi.sh`
  for example, the data directory is `data/en_vi`.
* The raw data should have 6 files, which contain source and target sentences of training/dev/test sets, respectively.
  In the `iwslt15_en_vi` example, `data/en_vi/train.en` contains the source sentences of the training set, where each
  line is a sentence. Other files are `train.vi`, `dev.en`, `dev.vi`, `test.en`, `test.vi`. 

### 2. Preprocess the data

To obtain the processed dataset, run

```bash
preprocess_data.sh ${encoder} ${src} ${tgt} ${vocab_size} ${max_seq_length}
```
where

* The `encoder` parameter can be `bpe`(byte pairwise encoding), `spm` (sentence piece encoding), or
`raw`(no subword encoding).
* `vocab_size` is optional. The default is 32000. 
  - At this point, this parameter is used only when `encoder` is set to `bpe` or `spm`. For `raw` encoding, you'd have
    to truncate the vocabulary by yourself.
  - For `spm` encoding, the preprocessing may fail (due to the Python sentencepiece module) if `vocab_size` is too
    large. So you may want to try smaller `vocab_size` if it happens. 
* `max_seq_length` is optional. The default is 70.

In the `iwslt15_en_vi` example, the command is `sh preprocess_data.sh spm en vi`.

By default, the preprocessed data are dumped under `temp/run_${src}_${tgt}_${encoder}`. In the `iwslt15_en_vi` example,
the directory is `temp/run_en_vi_spm`.

If you choose to use `raw` encoding method, notice that:

- By default, the word embedding layer is built with the combination of source vocabulary and target vocabulary. For
  example, if the source vocabulary is of size 3K and the target vocabulary of size 3K and there is no overlap between
  the two vocabularies, then the final vocabulary used in the model is of size 6K.
- By default, the final output layer of transformer decoder (hidden_state -> logits) shares the parameters with the word
  embedding layer.

### 3. Specify data and model configuration

Customize the Python configuration files to config the model and data.

Please refer to the example configuration files `config_model.py` for model configuration and `config_iwslt15.py` for
data configuration.

### 4. Train the model

Train the model with the following command:

```bash
python transformer_main.py \
    --run-mode=train_and_evaluate \
    --config-model=<custom_config_model> \
    --config-data=<custom_config_data>
```
where the model and data configuration files are `custom_config_model.py` and `custom_config_data.py`, respectively.

Outputs such as model checkpoints are by default under `outputs/`.

### 5. Test the model

Test with the following command:

```bash
python transformer_main.py \
    --run-mode=test \
    --config-data=<custom_config_data> \
    --output-dir=./outputs
```

Generated samples on the test set are in `outputs/test.output.hyp`, and reference sentences are in
`outputs/test.output.ref`. If you've used `bpe` or `spm` encoding in the data preprocessing step, make sure to set
`encoding` in the data configuration to the appropriate encoding type. The generated output will be decoded using the
specified encoding.

Finally, to evaluate the BLEU score against the ground truth on the test set:

```bash
python bleu_main.py --reference=<your_reference_file> --translation=temp/test.output.hyp.final
```
For the `iwslt15_en_vi` example, use `--reference=data/en_vi/test.vi`.
