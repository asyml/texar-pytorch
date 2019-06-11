# Overview #

**Texar** is an open-source toolkit based on PyTorch, aiming to support a broad set of machine learning especially **text generation tasks**, such as machine translation, dialog, summarization, content manipulation, language modeling, and so on. Texar is designed for both researchers and practitioners for fast prototyping and experimentation.
 
With the design goals of **modularity, versatility, and extensibility** in mind, Texar extracts the common patterns underlying the diverse tasks and methodologies, creates a library of highly reusable modules and functionalities, and facilitates **arbitrary model architectures and algorithmic paradigms**, e.g., 
   * encoder(s) to decoder(s), sequential- and self-attentions, memory, hierarchical models, classifiers... 
   * maximum likelihood learning, reinforcement learning, adversarial learning, probabilistic modeling, ... 

With Texar, cutting-edge complex models can be easily constructed, freely enriched with best modeling/training practices, readily fitted into standard training/evaluation pipelines, and rapidly experimented and evolved by, e.g., plugging-in and swapping-out different modules.

<div align="center">
   <img src="https://zhitinghu.github.io/texar_web/images/texar_stack.png"><br><br>
</div> 

### Key Features
* **Versatility**. Texar contains a wide range of modules and functionalities for composing arbitrary model architectures and implementing various learning algorithms, as well as for data processing, evaluation, prediction, etc.
* **Modularity**. Texar decomposes diverse complex machine learning models/algorithms into a set of highly-reusable modules. In particular, model **architecture, losses, and learning processes** are fully decomposed.  
Users can construct their own models at a high conceptual level just like assembling building blocks. It is convenient to plug in or swap out modules, and configure rich options of each module. For example, switching between maximum likelihood learning and reinforcement learning involves only changing several lines of code.
* **Extensibility**. It is straightforward to integrate any user-customized, external modules. Also, Texar is fully compatible with the native PyTorch interfaces and can take advantage of the rich PyTorch features, and resources from the vibrant open-source community.
* Interfaces with different functionality levels. Users can customize a model through 1) simple **Python/YAML configuration files** of provided model templates/examples; 2) programming with **Python Library APIs** for maximal customizability.
* Easy-to-use APIs: 1) Convenient automatic variable re-use---no worry about the complicated TF variable scopes; 2) PyTorch-like callable modules; 3) Rich configuration options for each module, all with default values; ...
* Well-structured high-quality code of uniform design patterns and consistent styles. 
* Clean, detailed [documentation](https://texar-pytorch.readthedocs.io) and rich [examples](https://github.com/asyml/texar-pytorch/tree/master/examples).

### Library API Example
Builds a (self-)attentional sequence encoder-decoder model, with different learning algorithms:
```python
import texar as tx

class Model(tx.ModuleBase):
    def __init__(self, data, hparams):
        self.src_embedder = tx.modules.WordEmbedder(data.source_vocab.size, hparams=hparams.embedder)
        self.tgt_embedder = tx.modules.WordEmbedder(data.target_vocab.size, hparams=hparams.embedder)
        self.encoder = tx.modules.BidirectionalRNNEncoder(input_size = self.src_embedder.dim, hparams=hparams.encoder)
        self.decoder = tx.modules.AttentionRNNDecoder(input_size=self.tgt_embedder.dim, encoder_output_size=)

# Data 
data = tx.data.PairedTextData(hparams=hparams_data) # Hyperparameter configs in `hparams` 
iterator = tx.data.DataIterator(data)

# Model architecture
embedder = 
encoder = tx.modules.BidirectionalRNNEncoder(input_size=embedder.dim,
                                             hparams=hparams_encoder)
outputs_enc = encoder(inputs=embedder(batch['source_text_ids']),
                      sequence_length=batch['source_length'])
                      
decoder = tx.modules.AttentionRNNDecoder(memory=output_enc, 
                                         memory_sequence_length=batch['source_length'],
                                         hparams=hparams_decoder)

attn_dim = hparams_decoder["attention_dim"]["kwargs"]["num_units"]
decoder = tx.modules.AttentionRNNDecoder(encoder_output_size=encoder.cell_fw.hidden_size \
                                                             + encoder.cell_bw.hidden_size,
                                         input_size=embedder.dim + attn_dim,
                                         vocab_size=data.target_vocab.size, 
                                         hparams=hparams_decoder)

for batch in iterator:
    training_outputs, _, _ = decoder(memory=torch.cat(enc_outputs, dim=2),
                                     memory_sequence_length=batch['source_length'],
                                     inputs=embedder(batch['target_text_ids'][:,:-1]),
                                     sequence_length=batch['target_length'] - 1)

    # mle loss
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                    labels=batch['target_text_ids'][:, 1:],
                    logits=training_outputs.logits,
                    sequence_length=batch['target_length'] - 1)
```
Many more examples are available [here](https://github.com/asyml/texar-pytorch/tree/master/examples).
  
### Installation
```
git clone https://github.com/asyml/texar.git
cd texar
pip install -e .
```

### Getting Started
* [Examples](https://github.com/asyml/texar-pytorch/tree/master/examples)
* [Documentations](https://texar-pytorch.readthedocs.io)
* [GitHub](https://github.com/asyml/texar-pytorch)

### Reference
If you use Texar, please cite the [report](https://arxiv.org/abs/1809.00794) with the following BibTex entry:
```
Texar: A Modularized, Versatile, and Extensible Toolkit for Text Generation
Zhiting Hu, Haoran Shi, Zichao Yang, Bowen Tan, Tiancheng Zhao, Junxian He, Wentao Wang, Xingjiang Yu, Lianhui Qin, Di Wang, Xuezhe Ma, Hector Liu, Xiaodan Liang, Wanrong Zhu, Devendra Singh Sachan, Eric P. Xing
2018

@article{hu2018texar, 
  title={Texar: A Modularized, Versatile, and Extensible Toolkit for Text Generation},
  author={Hu, Zhiting and Shi, Haoran and Yang, Zichao and Tan, Bowen and Zhao, Tiancheng and He, Junxian and Wang, Wentao and Yu, Xingjiang and Qin, Lianhui and Wang, Di and Ma, Xuezhe and Liu, Hector and Liang, Xiaodan and Zhu, Wanrong and Sachan, Devendra Singh and Xing, Eric},
  year={2018}
}
```

### License
[Apache License 2.0](https://github.com/asyml/texar-pytorch/blob/master/LICENSE)
