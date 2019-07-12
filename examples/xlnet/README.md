# XLNet: Pre-trained models and downstream applications

This is a Texar PyTorch implementation of the [XLNet model](https://github.com/zihangdai/xlnet), which supports loading
pre-trained model weights downloaded from the official release and building/fine-tuning downstream applications.

To summarize, this example showcases:

- Use of pre-trained XLNet models in Texar-PyTorch.
- Building and fine-tuning on downstream tasks.
- Use of Texar-PyTorch RecordData module for data loading and processing.

**Note:**
- This example has reproduced the reported results on STS-B and IMDB on GPUs. As per
  [the official repository](https://github.com/zihangdai/xlnet#memory-issue-during-finetuning), computational resources
  (e.g., GPU memory) can affect the results.
- This example supports classification and GLUE datasets. Other datasets can be supported by adding respective data
  modules. See [this section](https://github.com/huzecong/texar-xlnet#extend-to-custom-tasks).

**Future Work:**
- Distributed / Multi-GPU training.
- Fine-tuning on SQuAD & RACE datasets.

## Quickstart

### Install dependencies

Apart from requiring Texar-PyTorch, you should also satisfy dependencies in `requirements.txt` by running:
```bash
pip install -r requirements.txt
```

Specifically, TensorFlow is required to load the pre-trained models from the official release, and sentencepiece is
required for tokenization.

### Download pre-trained model

```bash
sh scripts/download_model.sh
```

By default, the pre-trained model (XLNet-Large Cased) will be downloaded to `pretrained/xlnet_cased_L-24_H-1024_A-16`.

### Download dataset

We will use the STS-B sentence pair relevance dataset as an example. Routines for other datasets are similar.

```bash
sh scripts/download_glue_data.sh
```

Data will be downloaded to `data/STS-B` along with other GLUE datasets.

Note that this is a regression task, the evaluation metric is Pearson's r correlation.

### Fine-tune the model

To fine-tune the model on the dataset, run the following command:
```bash
python xlnet_classification_main.py \
    --config-data configs/config_data_stsb.py \
    --pretrained pretrained/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --spm-model-path pretrained/xlnet_cased_L-24_H-1024_A-16/spiece.model
```

Configuration for the dataset is loaded from `configs/config_data_stsb.py`.

Data will be processed and cached when the dataset is loaded for the first time. By default, the data is loaded from
`data/STS-B` and cached to `processed_data/STS-B`. These paths can be customized by setting `--data-dir` and
`--cache-dir` flags. Note that it is not required to indicate that the task is a regression task --- the data loader
will figure it out.

For XLNet-Large Cased model, flags `--pretrained` and `--spm-model-path` can be omitted. Models are saved every 500
steps under the directory `saved_models`.

We use a batch size of 4 because this is the maximum batch size that fits under 12GB GPU memory. The official training
procedure uses an effective batch size of 32 (4 GPUs each with batch size 8), we simulate the behavior by gradient
accumulation with the `--backwards-per-step` flags set to 8. This means the parameters are updated (`optim.step()`) for
every 8 `loss.backward()` calls.

Fine-tuning takes about 45 minutes on a single NVIDIA GTX 1080 Ti. The expected Pearson's r on the development set is
0.9168. Result on the test set is NaN because test labels are not provided.

Note that we manually specify the random seed for reproducibility. You can override this by setting the `--seed` flag
to -1.

An example of training output is as follows:

```
$ python xlnet_classification_main.py --config-data configs/config_data_stsb.py
Random seed set to 19260817
Using CUDA device 0
INFO:root:Creating dataset in directory processed_data/STS-B.
100%|█████████████████████████████████████| 5749/5749 [00:01<00:00, 4809.99it/s]
100%|█████████████████████████████████████| 1500/1500 [00:00<00:00, 4522.86it/s]
100%|█████████████████████████████████████| 1379/1379 [00:00<00:00, 5063.21it/s]
INFO:root:Loading records with prefix "length128." from processed_data/STS-B
Dataset constructed
Weights initialized
WARNING: Certain weights from checkpoint are not loaded: ['model/transformer/mask_emb/mask_emb', 'global_step', 'model/lm_loss/bias']
Loaded pretrained weights from pretrained/xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt
Model constructed
Model structure: XLNetRegressor(
  (xlnet): XLNet(
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
Step: 100, LR = 4.167e-05, loss = 2.4108
Step: 200, LR = 4.630e-05, loss = 0.9530
Step: 300, LR = 4.167e-05, loss = 0.5697
Step: 400, LR = 3.704e-05, loss = 0.4628
Step: 500, LR = 3.241e-05, loss = 0.3435
Model at 500 steps saved to saved_models/STS-B_step500_20190703_120310
Pearsonr: 0.9014549605576836, loss: 0.4714
Step: 600, LR = 2.778e-05, loss = 0.2814
Step: 700, LR = 2.315e-05, loss = 0.2319
Step: 800, LR = 1.852e-05, loss = 0.2023
Step: 900, LR = 1.389e-05, loss = 0.1897
Step: 1000, LR = 9.259e-06, loss = 0.1441
Model at 1000 steps saved to saved_models/STS-B_step1000_20190703_122056
Pearsonr: 0.9149586649636707, loss: 0.3806
Step: 1100, LR = 4.630e-06, loss = 0.1279
Step: 1200, LR = 0.000e+00, loss = 0.1111
9599it [42:23,  4.14it/s]
Model at 1200 steps saved to saved_models/STS-B_step1200_20190703_122818
Evaluating on dev
100%|██████████████████████████| 24/24 [00:25<00:00,  1.10it/s, pearsonR=0.9168]
Pearsonr: 0.9167765866800505, loss: 0.3682
Evaluating on test
22it [00:24,  1.04it/s, pearsonR=nan]
Pearsonr: nan, loss: 9.1475
```

### Evaluate saved models

To evaluate a saved model, run the following command:
```bash
python xlnet_classification_main.py \
    --config-data configs/config_data_stsb.py \
    --checkpoint saved_models/path_to_checkpoint \
    --mode eval
```

## Text generation

Since XLNet is in essence a language model, it could be used to autoregressively generate text. We have also provided
an example to showcase text generation abilities of XLNet.

To run the text generation, run the following command:
```bash
python xlnet_generation_ipython.py
```
It is recommended to install IPython before running the command. If IPython is installed, you will enter an interactive
console in which you can perform sampling with different options. Here we show an example output:
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

This text generation example is largely inspired by the works of: https://github.com/rusiaaman/XLNet-gen. Especially, we
borrowed the trick of [adding random text for padding](https://github.com/rusiaaman/XLNet-gen#methodology), so
shorter prompts will not suffer from lack of attentional context.

## Extend to custom tasks

The interfaces of Texar XLNet are designed to be extensible. You can use your own dataset, or use XLNet as a standalone
module in other tasks.

### Use your own dataset by writing a custom data processor

It is easy to adapt the code to fine-tune XLNet on your custom dataset. To do this, you will need to write a custom
data processor inheriting `xlnet.data.DataProcessor`. For concrete examples, please refer to the built-in processors
under `xlnet/data/classification.py` and `xlnet/data/glue.py`.

The processor should implement `get_train_examples` and `get_dev_examples` (also `get_test_examples` if test set
exists). Each method returns a list of `xlnet.data.InputExample`s, each representing an example in the data split.
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

### Use XLNet as a standalone module

`xlnet.model.XLNet` can be used as a standalone module in a similar way to a Texar encoder. For convenience, we also
provide `XLNetClassifier` and `XLNetRegressor` for classification and regression tasks. Please refer to module
docstrings for their usage.
