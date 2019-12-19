# Language Model on PTB #

This example builds an LSTM language model, and trains on PTB data. Model and training are described in   
[(Zaremba, et. al.) Recurrent Neural Network Regularization](https://arxiv.org/pdf/1409.2329.pdf). This is a PyTorch implementation of the TensorFlow official PTB example in [tensorflow/models/rnn/ptb](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb).

The example shows:

  * Contruction of simple model, involving the `Embedder` and `RNN Decoder`.
  * Use of Texar-PyTorch with external Python data pipeline ([ptb_reader.py](./ptb_reader.py)).
  * Specification of various features of `train op`, like *gradient clipping* and *learning rate decay*.

## Usage ##

The following command trains a small-size model:

```
python lm_ptb.py --config config_small --data-path ./
```

Here:

  * `--config` specifies the configuration file to use. E.g., the above use the configuration defined in [config_small.py](./config_small.py)
  * `--data-path` specifies the directory containing PTB raw data (e.g., `ptb.train.txt`). If the data files do not exist, the program will automatically download, extract, and pre-process the data.

The model will begin training, and will evaluate on the validation data periodically, and evaluate on the test data after the training is done. 
