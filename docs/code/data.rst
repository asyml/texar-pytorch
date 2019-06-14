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

Data
==========

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
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`PairedTextData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.PairedTextData
    :members:
    :inherited-members:
    :exclude-members: make_vocab,make_embedding

:hidden:`ScalarData`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: texar.data.ScalarData
    :members:
    :inherited-members:

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
