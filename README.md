

[![Build Status](https://travis-ci.org/ZhitingHu/texar-pytorch.svg?branch=master)](https://travis-ci.org/ZhitingHu/texar-pytorch)

# Texar in Pytorch

The Pytorch version should keep the same with the [TF version](https://github.com/asyml/texar) in terms of:

* exact the same interfaces of *every* module, including function arguments and `hparams` structures, etc
* the same code structure, or even line-by-line correspondence whenever possible


### Code Style

* Texar follows [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html) 
* Make sure each line does not exceed 80 chars !
* Use [.pylintrc](./.pylintrc) to check code style 

### Reference repo

* [fairseq](https://github.com/pytorch/fairseq)
* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [GPytorch](https://github.com/cornellius-gp/gpytorch)

### Installation
```
git clone https://github.com/ZhitingHu/texar-pytorch.git 
cd texar-pytorch
pip install -e .
```

### License
[Apache License 2.0](./LICENSE)
