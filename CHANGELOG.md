## Unreleased

### New features

### Feature improvements

### Fixes

## [v0.1.1](https://github.com/asyml/texar-pytorch/releases/tag/v0.1.0) (2020-02-07)

### New features

* Support PyTorch 1.3. ([#249](https://github.com/asyml/texar-pytorch/pull/249))
* Add `T5` modules (`T5Encoder`, `T5Decoder`, and `T5EncoderDecoder`). ([#280](https://github.com/asyml/texar-pytorch/pull/280))
* Add `T5Tokenizer`. ([#283](https://github.com/asyml/texar-pytorch/pull/283))
* Support PyTorch 1.4. ([#291](https://github.com/asyml/texar-pytorch/pull/291))

### Feature improvements

* Refactor the interface of GPT2 modules. ([#238](https://github.com/asyml/texar-pytorch/pull/238)) 
* Support `gpt2-xl` checkpoint file in GPT2 modules. ([#242](https://github.com/asyml/texar-pytorch/pull/242))
* Add code coverage check in CI. ([#245](https://github.com/asyml/texar-pytorch/pull/245))
* Update the vocabulary files of RoBERTa modules. ([#255](https://github.com/asyml/texar-pytorch/pull/255))
* Disable `codecov/patch` check in CI. ([#265](https://github.com/asyml/texar-pytorch/pull/265))
* Provide option to freeze the embedding parameters. ([#271](https://github.com/asyml/texar-pytorch/pull/271))
* Add `encode_text_for_generation` function in `XLNetTokenizer`. ([#278](https://github.com/asyml/texar-pytorch/pull/278))
* Use warning instead of error in `map_token_to_id` function. ([#285](https://github.com/asyml/texar-pytorch/pull/285))
* Add copyright header to unit tests. ([#287](https://github.com/asyml/texar-pytorch/pull/287))
* Remove duplicated `pytest` in CI. ([#289](https://github.com/asyml/texar-pytorch/pull/289))
* Update the versions of `pylint`, `flake8`, and `mypy` in CI. ([#292](https://github.com/asyml/texar-pytorch/pull/292))

### Fixes

* Fix the documentation issues in `SentencePieceTokenizer`. ([#236](https://github.com/asyml/texar-pytorch/pull/236))
* Fix the bugs in RoBERTa checkpoint file loading procedure. ([#241](https://github.com/asyml/texar-pytorch/pull/241))
* Fix the documentation issues in `Executor`. ([#244](https://github.com/asyml/texar-pytorch/pull/244))
* Fix the documentation issues in `gpt-2` example. ([#250](https://github.com/asyml/texar-pytorch/pull/250))
* Fix the bugs in `bidirectional_dynamic_rnn` and `dynamic_rnn` functions. ([#252](https://github.com/asyml/texar-pytorch/pull/252))
* Fix the bugs in `vae_text` example. ([#253](https://github.com/asyml/texar-pytorch/pull/253))
* Fix the bugs in `sentence_classifier` example. ([#262](https://github.com/asyml/texar-pytorch/pull/262))
* Fix the path error when installing `texar-pytorch` in Windows. ([#268](https://github.com/asyml/texar-pytorch/pull/268))
* Fix the bugs in `XLNetTokenizer`. ([#273](https://github.com/asyml/texar-pytorch/pull/273))
* Fix the bugs in `download_checkpoint` function. ([#274](https://github.com/asyml/texar-pytorch/pull/274))
* Fix the bugs in google drive downloading function. ([#275](https://github.com/asyml/texar-pytorch/pull/275))
* Fix the bugs in the unit test of `GPT2Decoder`. ([#288](https://github.com/asyml/texar-pytorch/pull/288))
* Fix the documentation issues in `Decoder` module. ([#290](https://github.com/asyml/texar-pytorch/pull/290))

## [v0.1.0](https://github.com/asyml/texar-pytorch/releases/tag/v0.1.0) (2019-10-15)

The first formal release of Texar-PyTorch

## [v0.0.1](https://github.com/asyml/texar-pytorch/releases/tag/v0.0.1) (2019-08-02)

The first release of Texar-PyTorch
