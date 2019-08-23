# GPT-2: Pre-trained Langauge Model

This is a Texar PyTorch implementation of [OpenAI GPT-2 (Generative Pre-Trainning)](https://github.com/openai/gpt-2)
language model, which allows to load official pre-trained model parameters, generate samples, fine-tune the model,
and much more.

Texar provides ready-to-use modules including
[`GPT2Decoder`](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2decoder),
[`GPT2Encoder`](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2encoder),
and [`GPT2Classifier`](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2classifier). This example
shows the use of `GPT2Decoder` for generation tasks.

In sum, this example showcases:

* Building and using pre-trained GPT-2 models in Texar
* Using GPT-2 to generate text samples with or without context
* **Train or fine-tune** the model
* Examples of other use cases

Future work:

* **Train or fine-tune** the model with **distributed GPU** (coming soon)

## Prerequisite

#### Install dependencies

Apart from requiring Texar-PyTorch, you should also satisfy dependencies in `requirements.txt` by running:

```bash
pip install -r requirements.txt
```

## Quick Start (I) - Generation with the Pre-trained Model

### Usage
| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

#### Interactive mode (to generate samples with context)

This mode will initialize an interactive interface, which allows users to type in the context sentence. The model then
generates continuation of the context. The example supports both Top-K and Top-P sample decoding. By default, the GPT-2
`gpt2-small` model with Top-K sample decoding is used.

```bash
python gpt2_generate_main.py --interactive \
    --max-decoding-length=100 \
    --temperature=0.7 \
    --top-k=40
```

Here:

- `--interactive`: Specifies interactive mode.
- `--max-decoding-length`: The maximum number of tokens in the sample. **Note that this includes tokens in the
  context**. 
- `--nsamples`: Number of samples to generate for each input. 

For *top-k decoding*: 

- `--temperature`: Softmax temperature of top-k sample decoding. Larger values (above 1.0) result in more random
  samples, while smaller values push the sampling distribution towards the argmax. Must be strictly greater than 0.
  Defaults to `0.7`.
- `--top-k`: Number of top most likely candidates from a vocab distribution in each decoding step. Defaults to `40`.

For *top-p decoding*:
- `--top-p`: Select tokens with cumulative probability of at most `p` as candidates for sampling. Do not specify it
  if you want to use top-k decoding. 

To use the GPT-2 `gpt2-medium` or `gpt2-large` model, specify `--pretrained-model-name`:

```bash
python gpt2_generate_main.py --interactive \
    --max-decoding-length=100 \
    --temperature=0.7 \
    --top-k=40 \
    --pretrained-model-name=gpt2-medium
```
Here:

- `pretrained-model-name`: Name of the pre-trained checkpoint to load. Available options are: `gpt2-small`, `gpt2-medium`, and `gpt2-large`. Defaults to `gpt2-small`. 

To use Top-P sample decoding, specify `--top-p`:

```bash
python gpt2_generate_main.py --interactive \
    --max-decoding-length=100 \
    --temperature=0.7 \
    --top-p=40 \
    --pretrained-model-name=gpt2-medium 
```

Here:

- `--top-p`: Select tokens with cumulative probability of at most `p` when arranged in decreasing order.
  Default to `None`. 


**Example input:**

```
Model input >>> Michael Jordan is the greatest player in history !
```

**Example output:**

```
======================================== SAMPLE 1 ========================================

He's the one who has made all the difference. He's a true legend. He's a great athlete, 
a great athlete. He's a great athlete. I'm so happy for him. I'm so happy for his family, 
the family, and I'm so happy for him. I'm so happy for his teammates, his teammates, and 
I'm so happy for him.

The last time we saw him on stage, he

================================================================================
```

#### Non-interactive mode (to generate samples from scratch)

This mode generates a batch of samples from scratch.

```bash
python gpt2_generate_main.py \
    --nsamples=1 \
    --batch-size=1 \
    --max-decoding-len=100 \
    --temperature=0.7 \
    --top-k=40
```

Here:

- `--nsamples`: Total number of samples to generate, must be divisible by the batch size.
- `--batch-size`: The batch size. Each iteration generates this many samples.

To use GPT-2 `gpt2-medium` or `gpt2-large` model, specify `--pretrained-model-name` as above.

**Example output:**

```
"A new government and a healthy economy have a chance to take this up."

After he said the election's outcome in the House was important and had helped to build 
confidence in the House, former Ukip leader Nigel Farage spoke about working to boost 
the economy, saying the vote for the "lefties" and others "were bad optics for Labour 
in this way".
```

## Quick Start (II) - Fine-tune the Pre-trained Model 

This section shows how we can fine-tune the pre-trained GPT2 model and use the resulting model for generation.

### Prepare data

We first preprocess data with the GPT-2 BPE encoding. 

A toy dataset is provided under [`data/toy/`](data/toy) which includes `train.txt`, `dev.txt`, and `test.txt`. This
example will fit the GPT-2 model on `train.txt`, evaluate perplexity on `dev.txt`, and do continuation generation using
`test.txt` as the context.

Run the following command to transform the data into [pickle](https://docs.python.org/3/library/pickle.html) format and
perform processing such as truncation, BPE encoding, adding special tokens, etc:

```bash
python prepare_data.py --data-dir data/toy \
    --max-seq-length=128 \
    --output-dir=data/toy \
    --pretrained-model-name=gpt2-small
```

- `--data-dir`: The directory of raw data, wherein data files must be named as 'train.txt', 'dev.txt', or 'test.txt'. It
  is *not* necessary to provide all three files.
- `--max-seq-length`: The maximum length of sequence after BPE encoding. This includes GPT-2 special tokens that will be
  automatically added. Longer sequence will be trimmed. 
- `--output-dir`: The output path where the resulting pickled files will be put in. Be default, it is set to be the same
  as `--data-dir`. 
- `--pretrained-model-name`: The name of a pre-trained model to load selected in the list of: `gpt2-small`, `gpt2-medium`, and `gpt2-large`.

The above command will output pickled files in the specified output directory. E.g., if `train.txt` is provided under
`data_dir`, the output file `train.pkl` will be produced under `output_dir`.

### Train and Evaluate

For **single-GPU** training (and evaluation), run the following command to fine-tune the pre-trained GPT-2
parameters and evaluate perplexity on the dev set.

```bash
python gpt2_train_main.py --do-train --do-eval \
    --config-train=config_train \
    --output-dir=output
```

Here:

- `--config-train`: Configurations of GPT-2 training, including data and optimization hyperparameters. By default, the
  config file [`configs/config_train.py`](config_train.py) is used. Remember to specify correct data path if you are
  using your own data.
- `--output-dir`: The output path where checkpoints are saved.

By default, the GPT-2 `gpt2-small` model is used. To use the GPT-2 `gpt2-medium` or `gpt2-large` model instead, specify relevant arguments as below:

```bash
python gpt2_train_main.py --do-train --do-eval \
    --pretrained-model-name=gpt2-medium \
    --config-train=configs.config_train \
    --output-dir=output
```

You can also specify `--checkpoint` to load your own previously trained checkpoint. 

Please see the arguments in the code for more options.

## Other Use Cases

Texar's `GPT2Decoder` (and other RNN-based decoders) easily supports common, advanced, or customized usages, such as:

* Sample or continuation generation
* Greedy / (top-k) sample / Gumbel-softmax / beam-search / ... / your customized decoding algorithms
* Training / fine-tuning in (un)conditional settings
* Perplexity evaluation

**For example**, after creating the language model

```python    
decoder = GPT2Decoder(hparams=gpt2_hparams)
```
We can do

**Use case 1): Continuation generation w/ greedy decoding**

```python
output, output_length = decoder(
    context=ctx,
    context_sequence_length=ctx_len,
    decoding_strategy='infer_greedy',
    end_token=end_token)
    
sample_id = output.sample_id
logits = output.logits
```

**Use case 2): Top-k sample decoding**

```python    
topk_helper = tx.modules.TopKSampleEmbeddingHelper(
    start_tokens=ctx[:,0],
    end_token=end_token,
    top_k=20,
    softmax_temperature=0.7)
    
output, output_length = decoder(
    context=ctx,
    context_sequence_length=ctx_len,
    helper=topk_helper)
```

**Use case 3): Fine-tuning for conditional generation**

```python
output = decoder(
    memory=source_hidden_states, 
    memory_sequence_length=src_len,
    inputs=input_ids,
    decoding_strategy='train_greedy') # teacher-forcing decoding
    
loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    lables=truth_target[:, 1:],
    logits=output.logits,
    sequence_length=tgt_len-1)
```
