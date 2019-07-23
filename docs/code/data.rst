.. role:: hidden
    :class: hidden-section

Data
*******

Vocabulary
==========

:hidden:`SpecialTokens`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.SpecialTokens
    :members:

:hidden:`Vocab`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.Vocab
    :members:

:hidden:`map_ids_to_strs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.map_ids_to_strs

Embedding
==========

.. spelling::
    vec

:hidden:`Embedding`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.Embedding
    :members:

:hidden:`load_word2vec`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.load_word2vec

:hidden:`load_glove`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.load_glove


Data Sources
==============

:hidden:`DataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.DataSource
    :members:

:hidden:`SequenceDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.SequenceDataSource
    :members:

:hidden:`IterDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.IterDataSource
    :members:

:hidden:`ZipDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.ZipDataSource
    :members:

:hidden:`FilterDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.FilterDataSource
    :members:

:hidden:`RecordDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.RecordDataSource
    :members:

:hidden:`TextLineDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.TextLineDataSource
    :members:

:hidden:`PickleDataSource`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.PickleDataSource
    :members:



Data Loaders
=============

:hidden:`DataBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.DataBase
    :members:

    .. automethod:: process
    .. automethod:: collate

:hidden:`MonoTextData`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.MonoTextData
    :members:
    :exclude-members: make_vocab,make_embedding,process,collate

:hidden:`PairedTextData`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.PairedTextData
    :members:
    :exclude-members: make_vocab,make_embedding,process,collate

:hidden:`ScalarData`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.ScalarData
    :members:
    :exclude-members: process,collate

:hidden:`MultiAlignedData`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.MultiAlignedData
    :members:
    :exclude-members: make_vocab,make_embedding,process,collate,to

:hidden:`RecordData`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.RecordData
    :members:
    :exclude-members: process,collate

Data Iterators
===============

:hidden:`DataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.DataIterator
    :members:

:hidden:`TrainTestDataIterator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.TrainTestDataIterator
    :members:

:hidden:`BatchingStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.BatchingStrategy
    :members:

:hidden:`TokenCountBatchingStrategy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.data.TokenCountBatchingStrategy
    :members:
    :exclude-members: reset_batch,add_example


Data Utilities
===============

:hidden:`maybe_download`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.maybe_download

:hidden:`read_words`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.read_words

:hidden:`make_vocab`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.make_vocab

:hidden:`count_file_lines`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.data.count_file_lines
