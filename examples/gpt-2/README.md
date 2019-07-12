# GPT-2: Pre-trained Langauge Model

This is a Texar PyTorch implementation of [OpenAI GPT-2 (Generative Pre-Trainning)](https://github.com/openai/gpt-2) language model, which allows to load official pre-trained model parameters, generate samples, and fine-tune the model, etc.

Texar provides ready-to-use modules including [`GPT2Decoder`](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2decoder), [`GPT2Encoder`](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2encoder), [`GPT2Classifier`](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2classifier), etc. This example shows the use of `GPT2Decoder` for generation tasks.

In sum, this example showcases:

* Contructing and using pre-trained GPT-2 models in Texar
* Using GPT-2 to generate text samples with or without context
* **Train or fine-tune** the model with **distributed GPU** (coming soon)
* Examples of other use cases

## Quick Start (I) - Generation with the Pre-trained Model

### Usage
| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

#### Interactive mode (to generate samples with context)

This mode will initialize an interactive interface, which allows users to type in the context sentence. The model then generates continuation of the context. The example supports both Top-K and Top-P sample decoding. By default, the GPT-2 `117M` model with Top-K sample decoding is used.

```
python gpt2_generate_main.py --is_interactive \
--max_decoding_length=100 \
--temperature=0.7 \
--top_k=40
```

Here:

- `is_interactive`: Specifies interactive mode.
- `max_decoding_length`: The maximum number of tokens in the sample. **Note that this includes tokens in the context**. 
- `nsamples`: Number of samples to generate for each input. 

For *top-k decoding*: 

- `temperature`: Softmax temperature of top-k sample decoding. Larger values (above 1.0) result in more random samples, while smaller values push the sampling distribution towards the argmax. Must be strictly greater than 0. Defaults to `0.7`.
- `top_k`: Number of top most likely candidates from a vocab distribution in each decoding step. Defaults to `40`.

For *top-p decoding*:
- `top_p`: Select tokens with cumulative probability of at most 'top_p' as candidates for sampling. Do not specify it if you want to use top-k decoding. 

To use the GPT-2 `345M` model, specify `--config_model`:

```
python gpt2_generate_main.py --is_interactive \
--max_decoding_length=100 \
--temperature=0.7 \
--top_k=40 \
--config_model=configs.config_model_345M 
```

Here:

- `config_model`: Model configuration file. Default to `configs.config_model_117M`. 

To use Top-P sample decoding, specify `--top_p`:

```
python gpt2_generate_main.py --is_interactive \
--max_decoding_length=100 \
--temperature=0.7 \
--top_p=40 \
--config_model=configs.config_model_345M 
```

Here:

- `top_p`: Select tokens with cumulative probability of at most `p` when arranged in decreasing order. Default to be `None`. 


**Example input:**
```
Model input >>> Micheal Jordan is the greatest player in history !
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

```
python gpt2_generate_main.py
--nsamples=1 \
--batch_size=1 \
--max_decoding_len=100 \
--temperature=0.7 \
--top_k=40
```

Here:

- `nsamples`: Total number of samples to generate, must be dividable by the `batch_size`.
- `batch_size`: Each iteration generates `batch_size` number of samples.

To use GPT-2 `345M` model, `--config_model` as above.

**Example output:**

```
"A new government and a healthy economy have a chance to take this up."

After he said the election's outcome in the House was important and had helped to build 
confidence in the House, former Ukip leader Nigel Farage spoke about working to boost 
the economy, saying the vote for the "lefties" and others "were bad optics for Labour 
in this way".
```

## Quick Start (II) - Fine-tune the Pre-trained Model 

Coming soon!

## Other Use Cases

Texar's `GPT2Decoder` (and other RNN-based decoders) easily supports common, advanced, or customized use, such as:

* Sample or continuation generation
* Greedy / (top-k) sample / Gumbel-softmax / beam-search / ... / your-customized decoding
* Training / fine-tuning in (un)conditional settings
* Perplexity evaluation

**For example**, after creating the language model

```python    
decoder = GPT2Decoder(hparams=gpt2_hparams)
    
def _embedding_fn(ids, times):
    return decoder.word_embedder(ids) + decoder.position_embedder(times)
```
We can do

**Ex. Use 1): Continuation generation w/ greedy decoding**

```python
output, output_length = decoder(
    context=ctx,
    context_sequence_length=ctx_len,
    decoding_strategy='infer_greedy',
    end_token=end_token
    embedding=_embedding_fn)
    
sample_id = output.sample_id
logits = output.logits
```

**Ex. Use 2): Top-k sample decoding**

```python    
topk_helper = tx.modules.TopKSampleEmbeddingHelper(
    embedding=_embedding_fn,
    start_tokens=ctx[:,0],
    end_token=end_token,
    top_k=20,
    softmax_temperature=0.7)
    
output, output_length = decoder(
    context=ctx,
    context_sequence_length=ctx_len,
    helper=topk_helper)
```

**Ex. Use 3): Fine-tuning for conditional generation**

```python
tgt_embed = decoder.word_embedder(truth_target[:, :-1]) + decoder.position_embedder(sequence_length=tgt_len-1)

output = decoder(
    memory=source_hidden_states, 
    memory_sequence_length=src_len,
    inputs=tgt_embed,
    decoding_strategy='train_greedy') # teacher-forcing decoding
    
loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    lables=truth_target[:, 1:],
    logits=output.logits,
    sequence_length=tgt_len-1)
```
