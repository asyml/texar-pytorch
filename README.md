
<div align="center">
   <img src="https://zhitinghu.github.io/texar_web/images/logo_h_035.png"><br><br>
</div>
 
-----------------

[![Build Status](https://travis-ci.com/ZhitingHu/texar-pytorch.svg?token=jLMuFgZqHJTkobCG4qaR&branch=master)](https://travis-ci.com/ZhitingHu/texar-pytorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/asyml/texar/blob/master/LICENSE)
 
*(Note: This is the alpha release of Texar-PyTorch.)* 
 
**Texar-PyTorch** is an open-source toolkit based on PyTorch, aiming to support a broad set of machine learning especially **text generation tasks**, such as machine translation, dialog, summarization, content manipulation, language modeling, and so on. Texar is designed for both researchers and practitioners for fast prototyping and experimentation.
 
With the design goals of **modularity, versatility, and extensibility** in mind, Texar extracts the common patterns underlying the diverse tasks and methodologies, creates a library of highly reusable modules and functionalities, and facilitates **arbitrary model architectures and algorithmic paradigms**, e.g., 
   * encoder(s) to decoder(s), sequential- and self-attentions, memory, hierarchical models, classifiers... 
   * maximum likelihood learning, reinforcement learning, adversarial learning, probabilistic modeling, ... 

With Texar, cutting-edge complex models can be easily constructed, freely enriched with best modeling/training practices, readily fitted into standard training/evaluation pipelines, and fastly experimented and evolved by, e.g., plugging-in and swapping-out different modules.

<div align="center">
   <img src="https://zhitinghu.github.io/texar_web/images/texar_stack.png"><br><br>
</div> 

### Key Features
* **Versatility**. Texar contains a wide range of modules and functionalities for composing arbitrary model architectures and implementing various learning algorithms, as well as for data processing, evaluation, prediction, etc.
* **Modularity**. Texar decomposes diverse complex machine learning models/algorithms into a set of highly-reusable modules. In particular, model **architecture, losses, and learning processes** are fully decomposed.  
Users can construct their own models at a high conceptual level just like assembling building blocks. It is convenient to plug-ins or swap-out modules, and configure rich options of each module. For example, switching between maximum likelihood learning and reinforcement learning involves only changing several lines of code.
* **Extensibility**. It is straightforward to integrate any user-customized, external modules. Also, Texar is fully compatible with the native TensorFlow interfaces and can take advantage of the rich TensorFlow features, and resources from the vibrant open-source community.
* Interfaces with different functionality levels. Users can customize a model through 1) simple **Python/YAML configuration files** of provided model templates/examples; 2) programming with **Python Library APIs** for maximal customizability.
* Easy-to-use APIs; rich configuration options for each module, all with default values.
* Well-structured high-quality code of uniform design patterns and consistent styles. 
* Clean, detailed [documentation](./docs) and rich [examples](./examples).

### Installation
```
git clone https://github.com/ZhitingHu/texar-pytorch.git 
cd texar-pytorch
pip install -e .
```

### Getting Started
* [Examples](./examples)
* [Documentation](./docs)

### Reference
If you use Texar, please cite the [report](https://arxiv.org/abs/1809.00794) with the following BibTex entry:
```
Texar: A Modularized, Versatile, and Extensible Toolkit for Text Generation
Zhiting Hu, Haoran Shi, Zichao Yang, Bowen Tan, Tiancheng Zhao, Junxian He, Wentao Wang, Lianhui Qin, Di Wang, Xuezhe Ma, Hector Liu, Xiaodan Liang, Wanrong Zhu, Devendra Singh Sachan, Eric P. Xing
2018

@article{hu2018texar,
  title={Texar: A Modularized, Versatile, and Extensible Toolkit for Text Generation},
  author={Hu, Zhiting and Shi, Haoran and Yang, Zichao and Tan, Bowen and Zhao, Tiancheng and He, Junxian and Wang, Wentao and Qin, Lianhui and Wang, Di and others},
  journal={arXiv preprint arXiv:1809.00794},
  year={2018}
}
```

### License
[Apache License 2.0](./LICENSE)
