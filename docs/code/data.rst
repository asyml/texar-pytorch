.. role:: hidden
    :class: hidden-section

Data
*******

Tokenizer
==========

:hidden:`TokenizerBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.TokenizerBase
    :members:

:hidden:`SentencePieceTokenizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.SentencePieceTokenizer
    :members:

:hidden:`BERTTokenizer`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.BERTTokenizer
    :members:

:hidden:`GPT2Tokenizer`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.GPT2Tokenizer
    :members:

:hidden:`RoBERTaTokenizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.RoBERTaTokenizer
    :members:

:hidden:`XLNetTokenizer`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.XLNetTokenizer
    :members:

Vocabulary
==========

:hidden:`SpecialTokens`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.SpecialTokens
    :members:

:hidden:`Vocab`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.Vocab
    :members:

:hidden:`map_ids_to_strs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.data.map_ids_to_strs

Embedding
==========

.. spelling::
    vec

:hidden:`Embedding`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.Embedding
    :members:

:hidden:`load_word2vec`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.data.load_word2vec

:hidden:`load_glove`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.data.load_glove


Data Sources
==============

:hidden:`DataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.DataSource
    :members:

:hidden:`SequenceDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.SequenceDataSource
    :members:

:hidden:`IterDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.IterDataSource
    :members:

:hidden:`ZipDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.ZipDataSource
    :members:

:hidden:`FilterDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.FilterDataSource
    :members:

:hidden:`RecordDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.RecordDataSource
    :members:

:hidden:`TextLineDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.TextLineDataSource
    :members:

:hidden:`PickleDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.PickleDataSource
    :members:



Data Loaders
=============

:hidden:`DatasetBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.DatasetBase
    :members:

    .. automethod:: process
    .. automethod:: collate

:hidden:`MonoTextData`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.MonoTextData
    :members:
    :exclude-members: make_vocab,make_embedding,process,collate

:hidden:`PairedTextData`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.PairedTextData
    :members:
    :exclude-members: make_vocab,make_embedding,process,collate

:hidden:`ScalarData`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.ScalarData
    :members:
    :exclude-members: process,collate

:hidden:`MultiAlignedData`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.MultiAlignedData
    :members:
    :exclude-members: make_vocab,make_embedding,process,collate,to

:hidden:`RecordData`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.RecordData
    :members:
    :exclude-members: process,collate

Data Iterators
===============

:hidden:`Batch`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.Batch
    :members:

:hidden:`DataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.DataIterator
    :members:

:hidden:`TrainTestDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.TrainTestDataIterator
    :members:

:hidden:`BatchingStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.BatchingStrategy
    :members:

:hidden:`TokenCountBatchingStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.data.TokenCountBatchingStrategy
    :members:
    :exclude-members: reset_batch,add_example


Data Utilities
===============

:hidden:`maybe_download`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.data.maybe_download

:hidden:`read_words`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.data.read_words

:hidden:`make_vocab`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.data.make_vocab

:hidden:`count_file_lines`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.data.count_file_lines
