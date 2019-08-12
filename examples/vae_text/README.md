# Variational Autoencoder (VAE) for Text Generation

This example builds a VAE for text generation, with an LSTM as encoder and an LSTM or [Transformer](https://arxiv.org/pdf/1706.03762.pdf) as decoder. Training is performed on the official PTB data and Yahoo data, respectively. 

The VAE with LSTM decoder is first decribed in [(Bowman et al., 2015) Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)

The Yahoo dataset is from [(Yang et al., 2017) Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/pdf/1702.08139.pdf), which is created by sampling 100k documents from the original Yahoo Answer data. The average document length is 78 and the vocab size is 200k. 

## Data
The datasets can be downloaded by running:
```shell
python prepare_data.py --data ptb
python prepare_data.py --data yahoo
```

## Training
Train with the following command:

```shell
python vae_train.py --config config_trans_ptb
```

Here:

* `--config` specifies the config file to use, including model hyperparameters and data paths. We provide 4 config files:
  - [config_lstm_ptb.py](./config_lstm_ptb.py): LSTM decoder, on the PTB data
  - [config_lstm_yahoo.py](./config_lstm_yahoo.py): LSTM decoder, on the Yahoo data
  - [config_trans_ptb.py](./config_trans_ptb.py): Transformer decoder, on the PTB data
  - [config_trans_yahoo.py](./config_trans_yahoo.py): Transformer decoder, on the Yahoo data

## Generation
Generating sentences with pre-trained model can be performed with the following command:
```shell
python vae_train.py --config config_file --mode predict --model /path/to/model.ckpt --out /path/to/output
```

Here `--model` specifies the saved model checkpoint, which is saved in `./models/dataset_name/` at training time. For example, the model path is `./models/ptb/ptb_lstmDecoder.ckpt` when generating with a LSTM decoder trained on PTB dataset. Generated sentences will be written to standard output if `--out` is not specifcied.

## Results

### Language Modeling

|Dataset    |Metrics   | VAE-LSTM |VAE-Transformer |
|---------------|-------------|----------------|------------------------|
|Yahoo | Test PPL<br>Test NLL | 69.85<br>339.15 |60.78<br>328.03|
|PTB | Test PPL<br>Test NLL | 107.64<br>102.54 | 103.14<br>101.61 |

### Generated Examples
We show the generated examples with transformer as decoder trained  on PTB training data.

|Examples|
|:---------|
|mostly in its annual <unk> spending N in september to $ N billion from $ N billion a year earlier \<EOS\>|
|note it 's a N rating of N N owned by shareholders from the banking and the expectation of interest payment since trading of a sale of a stock was a effect \<EOS\>|
|but the profit margin eye on its rise disrupted that cms had slashed its quarterly profit in the quarter \<EOS\>|
|that weeks ago the new business was to follow the dollar \<EOS\>|
|the issue was quoted at N marks and for november delivery of N cent \<EOS\>|