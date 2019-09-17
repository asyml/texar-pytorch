# XLNet for Classification and Generation 

This is a Texar PyTorch implementation of the [XLNet model](https://github.com/zihangdai/xlnet), which supports loading
pre-trained model weights downloaded from the official release and building/fine-tuning downstream applications.

To summarize, this example showcases:

- [Fine-tuning XLNet for classification](#classification)
- [XLNet for text generation](#generation)
- [XLNet for other custom tasks](#extend-to-custom-tasks)

**Note:**
- For classification, this example has reproduced the reported results on STS-B and IMDB on GPUs. As per
  [the official repository](https://github.com/zihangdai/xlnet#memory-issue-during-finetuning), computational resources
  (e.g., GPU memory) can affect the results.
- This example supports classification and GLUE datasets. Other datasets can be supported by adding respective data
  modules. See [this section](#extend-to-custom-tasks).

**Future Work:**

- Distributed / Multi-GPU training.
- Fine-tuning on SQuAD & RACE datasets.
- *Please propose an issue for what you expect*

## Prerequisite

#### Install dependencies

Apart from requiring Texar-PyTorch, you should also satisfy dependencies in `requirements.txt` by running:

```bash
pip install -r requirements.txt
```

Specifically, TensorFlow is required to load the pre-trained models from the official release, and sentencepiece is
required for tokenization.

## Classification

#### Download dataset

We will use the STS-B sentence pair relevance dataset as an example. Routines for other datasets are similar.

```bash
sh scripts/download_glue_data.sh
```

Data will be downloaded to `data/STS-B` along with other GLUE datasets.

Note that this is a regression task, the evaluation metric is Pearson's r correlation.

#### Fine-tune the model

To fine-tune the model on the dataset, run the following command:

```bash
python xlnet_classification_main.py \
    --config-data configs/config_data_stsb.py
```

Configuration for the dataset is loaded from `configs/config_data_stsb.py`.

Data will be processed and cached when the dataset is loaded for the first time. By default, the data is loaded from
`data/STS-B` and cached to `processed_data/STS-B`. These paths can be customized by setting `--data-dir` and
`--cache-dir` flags. Note that it is not required to indicate that the task is a regression task --- the data loader
will figure it out.

You can use `--pretrained-model-name` to specify the pre-trained model you want to use. Models are saved every 500
steps under the directory `saved_models/STS-B`.

We use a batch size of 4 because this is the maximum batch size that fits under 12GB GPU memory. The official training
procedure uses an effective batch size of 32 (4 GPUs each with batch size 8), we simulate the behavior by gradient
accumulation with the `--backwards-per-step` flags set to 8. This means the parameters are updated (`optim.step()`) for
every 8 `loss.backward()` calls.

Fine-tuning takes about 45 minutes on a single NVIDIA GTX 1080 Ti. The expected Pearson's r on the development set is
0.9204. Result on the test set is NaN because test labels are not provided.

Note that we manually specify the random seed for reproducibility. You can override this by setting the `--seed` flag 
to -1.

An example of training output is as follows:

```
$ python xlnet_classification_main.py --config-data configs/config_data_stsb.py
Random seed set to 19260817
Using CUDA device 0
>> Downloading cased_L-24_H-1024_A-16.zip 100.0%%
Successfully downloaded cased_L-24_H-1024_A-16.zip 1338042341 bytes.
INFO:root:Extract texar_download/XLNet/xlnet-large-cased/cased_L-24_H-1024_A-16.zip
INFO:root:Creating dataset in directory processed_data/STS-B.
100%|█████████████████████████████████████| 5749/5749 [00:01<00:00, 4927.52it/s]
100%|█████████████████████████████████████| 1500/1500 [00:00<00:00, 4899.25it/s]
100%|█████████████████████████████████████| 1379/1379 [00:00<00:00, 5198.48it/s]
INFO:root:Loading records with prefix "xlnet-large-cased.length128." from processed_data/STS-B
Dataset constructed
Using cached pre-trained XLNet model from: texar_download/XLNet/xlnet-large-cased.
WARNING: Certain weights from checkpoint are not loaded: ['model/transformer/mask_emb/mask_emb', 'model/lm_loss/bias']
Model constructed
INFO 2019-08-06 10:51:30 : Training started
INFO 2019-08-06 10:51:30 : Model architecture:
RegressorWrapper(
  (_encoder): XLNetEncoder(
    (word_embed): Embedding(32000, 1024)
    (pos_embed): RelativePositionalEncoding(
      (sinusoid_embed): PositionalEmbedding()
    )
    (dropout): Dropout(p=0.1)
    (attn_layers): ModuleList(
      (ids 0-23): RelativeMultiheadAttention(
        (head_projection): Linear(in_features=1024, out_features=3072, bias=False)
        (pos_projection): Linear(in_features=1024, out_features=1024, bias=False)
        (dropout): Dropout(p=0.1)
        (dropout_attn): Dropout(p=0.1)
        (output_projection): Linear(in_features=1024, out_features=1024, bias=False)
        (layer_norm): LayerNorm(torch.Size([1024]), eps=1e-12, elementwise_affine=True)
      )
    )
    (ff_layers): ModuleList(
      (ids 0-23): PositionWiseFF(
        (linear1): Linear(in_features=1024, out_features=4096, bias=True)
        (activation_fn): GPTGELU()
        (dropout): Dropout(p=0.1, inplace)
        (linear2): Linear(in_features=4096, out_features=1024, bias=True)
        (layer_norm): LayerNorm(torch.Size([1024]), eps=1e-12, elementwise_affine=True)
      )
    )
  )
  (projection): Linear(in_features=1024, out_features=1024, bias=True)
  (dropout): Dropout(p=0.1)
  (hidden_to_logits): Linear(in_features=1024, out_features=1, bias=True)
)
2019-08-06 10:54:50 : Epoch 1 @   800it (16.01ex/s), LR = 4.208e-05, loss = 2.863
2019-08-06 10:58:16 : Epoch 2 @  1600it (15.40ex/s), LR = 4.625e-05, loss = 0.880
2019-08-06 11:01:41 : Epoch 2 @  2400it (15.58ex/s), LR = 4.162e-05, loss = 0.527
2019-08-06 11:05:05 : Epoch 3 @  3200it (15.59ex/s), LR = 3.699e-05, loss = 0.451
2019-08-06 11:08:31 : Epoch 3 @  4000it (15.57ex/s), LR = 3.236e-05, loss = 0.324
2019-08-06 11:08:57 : Epoch 3, valid result = {PearsonR: 0.907, loss: 0.402}
INFO 2019-08-06 11:09:13 : Current checkpoint saved to saved_models/STS-B/1565104137.1357086.pt
2019-08-06 11:12:37 : Epoch 4 @  4800it (15.60ex/s), LR = 2.773e-05, loss = 0.274
2019-08-06 11:16:02 : Epoch 4 @  5600it (15.59ex/s), LR = 2.310e-05, loss = 0.238
2019-08-06 11:19:28 : Epoch 5 @  6400it (15.59ex/s), LR = 1.847e-05, loss = 0.179
2019-08-06 11:22:52 : Epoch 6 @  7200it (15.30ex/s), LR = 1.384e-05, loss = 0.167
2019-08-06 11:26:18 : Epoch 6 @  8000it (15.58ex/s), LR = 9.213e-06, loss = 0.130
2019-08-06 11:26:43 : Epoch 6, valid result = {PearsonR: 0.919, loss: 0.359}
INFO 2019-08-06 11:26:48 : Previous checkpoint 1565104137.1357086.pt removed due to `max_to_keep`(=1) limit
INFO 2019-08-06 11:27:04 : Current checkpoint saved to saved_models/STS-B/1565105208.3423455.pt
2019-08-06 11:30:29 : Epoch 7 @  8800it (15.61ex/s), LR = 4.583e-06, loss = 0.126
2019-08-06 11:33:54 : Epoch 7 @  9600it (15.57ex/s), LR = -4.630e-08, loss = 0.102
INFO 2019-08-06 11:33:54 : Training terminated
INFO 2019-08-06 11:33:58 : Previous checkpoint 1565105208.3423455.pt removed due to `max_to_keep`(=1) limit
INFO 2019-08-06 11:34:16 : Current checkpoint saved to saved_models/STS-B/1565105638.0698519.pt
2019-08-06 11:34:40 : Epoch 7, dev result = {PearsonR: 0.920, loss: 0.352}
2019-08-06 11:35:04 : Epoch 7, test result = {PearsonR: nan, loss: 9.151}
```

#### Evaluate saved models

To evaluate a saved model, run the following command:

```bash
python xlnet_classification_main.py \
    --config-data configs/config_data_stsb.py \
    --checkpoint saved_models/path_to_checkpoint \
    --mode eval
```

## Generation

Since XLNet is in essence a language model, it could be used to autoregressively generate text. We have also provided
examples to showcase text generation abilities of XLNet:

- [Interactive mode (to generate samples with context)](#interactive-mode-to-generate-samples-with-context)
- [Non-interactive mode (to generate samples from scratch)](#non-interactive-mode-to-generate-samples-from-scratch) 
- [IPython mode (to play with different decoding strategies)](#ipython-mode-to-play-with-different-decoding-strategies)

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

#### Interactive mode (to generate samples with context)

This mode will initialize an interactive interface, which allows users to type in the context sentence. The model then
generates continuation of the context. The example supports both Top-K and Top-P sample decoding. 

```bash
python xlnet_generation_main.py --interactive \
    --max-decoding-length=200 \
    --temperature=0.7 \
    --top-k=40
```

Here:

- `--interactive`: Specifies interactive mode.
- `--max-decoding-length`: The maximum number of tokens in the sample. **Note that this includes tokens in the context**. 
- `--nsamples`: Number of samples to generate for each input. 

For *top-k decoding*: 

- `--temperature`: Softmax temperature of top-k sample decoding. Larger values (above 1.0) result in more random samples,
  while smaller values push the sampling distribution towards the argmax. Must be strictly greater than 0.
  Defaults to `0.7`.
- `--top-k`: Number of top most likely candidates from a vocab distribution in each decoding step. Defaults to `40`.

For *top-p decoding*:
- `--top-p`: Select tokens with cumulative probability of at most 'top-p' as candidates for sampling. Do not specify it
  if you want to use top-k decoding. 


**Example input:**

```
Model input >>> Micheal Jordan is the greatest player in history !
```

**Example output:**

```
======================================== SAMPLE 1 ========================================

He was born George Jordan March 22, 1928, in Tobago, Trinidad and Tobago. Jordan walked super fast 
and moved fast. He was also a tremendous downhill skier. He will go down in history with basketball as 
an ancient foe. 
Teleprint: This publication provides print service through the help of acertified Inter Print Printer. 
Teleprint is intended for users who are not physical print service providers ("HSPs") or printers 
who are not dealers of or in possession of services offered by a specific HP. Note allowed: Users 
who are in possession of services offered by a specific HP are authorized to use high-speed inter print 
services.

================================================================================
```

#### Non-interactive mode (to generate samples from scratch)

This mode generates a batch of samples from scratch.

```bash
python xlnet_generation_main.py \
    --nsamples=1 \
    --batch-size=1 \
    --max-decoding-len=100 \
    --temperature=0.7 \
    --top-k=40
```

Here:

- `--nsamples`: Total number of samples to generate, must be dividable by the `--batch-size`.
- `--batch-size`: Each iteration generates `--batch-size` number of samples.

**Example output:**

```
"A new government and a healthy economy have a chance to take this up."

After he said the election's outcome in the House was important and had helped to build 
confidence in the House, former Ukip leader Nigel Farage spoke about working to boost 
the economy, saying the vote for the "lefties" and others "were bad optics for Labour 
in this way".
```

#### IPython mode (to play with different decoding strategies)

The IPython mode allows you to play with different decoding strategies (top-k, top-p, greedy, etc) and other
hyperparameters.

Install IPython, and run the following command to enter an interactive console.

```bash
python xlnet_generation_ipython.py
```
Here we show an example output:

```
Generate text by calling: sample("<your prompt text>", ...).
For options, refer to `decode` method of `XLNetDecoder`.

Python 3.7.3 (default, Mar 27 2019, 22:11:17)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: sample("In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously un
   ...: explored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorn
   ...: s spoke perfect English.", cache_len=512, n_samples=1)
=== Prompt ===
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the
 Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.
=== Sample 0 ===
The reason for this astonishing find is an odd enough one - "i.e. horses talk, but don't drink...." When researching an
 "impression of the human brain schema" which resembled that of the unicorns, researchers discovered that these animals
 were not only adapted to live in a particular environment, but were able to communicate and communicate with one anoth
er; this was the result of brain mutation. The "brain schema" of the unicorn included a number of parts which were comp
atible with English speaking, people of that time. This had an interesting effect on the species, allowing them to comm
unicate better, including giving them a "little bit of a leg up" on the English speaking people of the time, from whom 
many "English-speaking" ("in many respects") settlers became.
```

*This text generation example is largely inspired by the works of: https://github.com/rusiaaman/XLNet-gen. Especially,
we borrowed the trick of [adding random text for padding](https://github.com/rusiaaman/XLNet-gen#methodology), so
shorter prompts will not suffer from lack of attentional context.*

## Extend to custom tasks

The interfaces of Texar XLNet are designed to be extensible. You can use your own dataset, or use XLNet as a standalone
module in other tasks.

#### Use your own dataset by writing a custom data processor

It is easy to adapt the code to fine-tune XLNet on your custom dataset. To do this, you will need to write a custom
data processor inheriting `utils.processor.DataProcessor`. For concrete examples, please refer to the built-in
processors under `utils/processor.py`.

The processor should implement `get_train_examples` and `get_dev_examples` (also `get_test_examples` if test set
exists). Each method returns a list of `utils.processor.InputExample`s, each representing an example in the data split.
`InputExample` is a named tuple, consisting of the following fields:

- `guid`: A unique ID for the example, can be set to `None`.
- `text_a`: The untokenized text of the first sequence.
- `text_b`: The untokenized text of the second sequence. For tasks involving only a single sequence, `text_b` can be set
  to None.
- `label`: The ground truth label. If the task is a regression task, this should be a float; if a classification task,
  this should be a valid string.

The processor should also set class attributes `labels` and `is_regression`. `labels` is the list of acceptable string
labels for the task, while `is_regression` is a boolean flag. Finally, you should register your processor using the
decorator `@DataProcessor.register("task_name")`.

Now, simply import your processor into `run.py`, and run the training command with `--task` flags set to your task name.
