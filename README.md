
<div align="center">
   <img src="./docs/_static/img/logo_h_035.png"><br><br>
</div>
 
-----------------

[![Build Status](https://travis-ci.org/asyml/texar-pytorch.svg?branch=master)](https://travis-ci.org/asyml/texar-pytorch)
[![Documentation Status](https://readthedocs.org/projects/texar-pytorch/badge/?version=latest)](https://texar-pytorch.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/asyml/texar-pytorch/blob/master/LICENSE)
 
*(Note: This is the alpha release of Texar-PyTorch.)* 
 
**Texar-PyTorch** is an open-source toolkit based on PyTorch, aiming to support a broad set of machine learning, especially text generation tasks, such as machine translation, dialog, summarization, content manipulation, language modeling, and so on. Texar is designed for both researchers and practitioners for fast prototyping and experimentation.

*If you work with TensorFlow, be sure to check out **[Texar (TensorFlow)](https://github.com/asyml/texar)** which has (mostly) the **same functionalities and interfaces**.*

With the design goals of **modularity, versatility, and extensibility** in mind, Texar extracts the common patterns underlying the diverse tasks and methodologies, creates a library of highly reusable modules and functionalities, and facilitates **arbitrary model architectures and algorithmic paradigms**, e.g., 
   * encoder(s) to decoder(s), sequential- and self-attentions, memory, hierarchical models, classifiers, ... 
   * maximum likelihood learning, reinforcement learning, adversarial learning, probabilistic modeling, ... 
   * **pre-trained models** such as **BERT**, **GPT2**, **XLNet**, ...

With Texar, cutting-edge complex models can be easily constructed, freely enriched with best modeling/training practices, readily fitted into standard training/evaluation pipelines, and quickly experimented and evolved by, e.g., plugging in and swapping out different modules.

<div align="center">
   <img src="./docs/_static/img/texar_stack.png"><br><br>
</div> 


### Key Features
* **Versatility**. Texar contains a wide range of modules and functionalities for composing arbitrary model architectures and implementing various learning algorithms, as well as for data processing, evaluation, prediction, etc.
* **Modularity**. Texar decomposes diverse, complex machine learning models and algorithms into a set of highly-reusable modules. In particular, model **architecture, losses, and learning processes** are fully decomposed.  
Users can construct their own models at a high conceptual level, just like assembling building blocks. It is convenient to plug in or swap out modules, and configure rich options for each module. For example, switching between maximum-likelihood learning and reinforcement learning involves only changing several lines of code.
* **Extensibility**. It is straightforward to integrate any user-customized, external modules. Also, Texar is fully compatible with the native PyTorch interfaces and can take advantage of the rich PyTorch features, and resources from the vibrant open-source community.
* Interfaces with different functionality levels. Users can customize a model through 1) simple **Python/YAML configuration files** of provided model templates/examples; 2) programming with **Python Library APIs** for maximal customizability.
* Easy-to-use APIs; rich configuration options for each module, all with default values.
* Well-structured high-quality code of uniform design patterns and consistent styles. 
* Clean, detailed [documentation](https://texar-pytorch.readthedocs.io) and rich [examples](./examples).


### Library API Example
A code portion that builds a (self-)attentional sequence encoder-decoder model:
```python
import texar.torch as tx

class Seq2Seq(tx.ModuleBase):
  def __init__(self, data):
    self.embedder = tx.modules.WordEmbedder(
        data.target_vocab.size, hparams=hparams_emb)
    self.encoder = tx.modules.TransformerEncoder(
        hparams=hparams_encoder)  # config through `hparams`
    self.decoder = tx.modules.AttentionRNNDecoder(
        token_embedder=self.embedder,
        input_size=self.embedder.dim,
      	encoder_output_size=self.encoder.output_size,
      	vocab_size=data.target_vocab.size,
        hparams=hparams_decoder)

  def forward(self, batch): 
    outputs_enc = self.encoder(
        inputs=self.embedder(batch['source_text_ids']),
        sequence_length=batch['source_length'])
     
    outputs, _, _ = self.decoder(
        memory=outputs_enc, 
        memory_sequence_length=batch['source_length'],
        helper=self.decoder.get_helper(decoding_strategy='train_greedy'), 
        inputs=batch['target_text_ids'],
        sequence_length=batch['target_length']-1)

    # Loss for maximum likelihood learning
    loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=batch['target_text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=batch['target_length']-1)  # Automatic masking

    return loss


data = tx.data.PairedTextData(hparams=hparams_data) 
iterator = tx.data.DataIterator(data)

model = Seq2seq(data)
for batch in iterator:
    loss = model(batch)
    # ...
```
Many more examples are available [here](./examples).


### Installation
Texar-PyTorch requires PyTorch 1.0 or higher. Please follow the [official instructions](https://pytorch.org/get-started/locally/#start-locally) to install the appropriate version.

After PyTorch is installed, please run the following commands to install Texar-PyTorch:
```
git clone https://github.com/asyml/texar-pytorch.git 
cd texar-pytorch
pip install -e .
```


### Getting Started
* [Examples](./examples)
* [Documentation](https://texar-pytorch.readthedocs.io)


### Reference
If you use Texar, please cite the [tech report](https://arxiv.org/abs/1809.00794) with the following BibTex entry:
```
Texar: A Modularized, Versatile, and Extensible Toolkit for Text Generation
Zhiting Hu, Haoran Shi, Bowen Tan, Wentao Wang, Zichao Yang, Tiancheng Zhao, Junxian He, Lianhui Qin, Di Wang, Xuezhe Ma, Zhengzhong Liu, Xiaodan Liang, Wanrong Zhu, Devendra Sachan and Eric Xing
2018

@article{hu2018texar,
  title={Texar: A Modularized, Versatile, and Extensible Toolkit for Text Generation},
  author={Hu, Zhiting and Shi, Haoran and Tan, Bowen and Wang, Wentao and Yang, Zichao and Zhao, Tiancheng and He, Junxian and Qin, Lianhui and Wang, Di and others},
  journal={arXiv preprint arXiv:1809.00794},
  year={2018}
}
```


### License
[Apache License 2.0](./LICENSE)
