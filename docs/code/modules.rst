.. role:: hidden
    :class: hidden-section

Modules
*******

ModuleBase
===========

.. autoclass:: texar.torch.ModuleBase
    :members:
    :exclude-members: forward

Embedders
=========

:hidden:`WordEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.WordEmbedder
    :members:

:hidden:`PositionEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.PositionEmbedder
    :members:

:hidden:`SinusoidsPositionEmbedder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.SinusoidsPositionEmbedder
    :members:

:hidden:`EmbedderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.EmbedderBase
    :members:
    :exclude-members: forward


Encoders
========

:hidden:`UnidirectionalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.UnidirectionalRNNEncoder
    :members:

:hidden:`BidirectionalRNNEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.BidirectionalRNNEncoder
    :members:

:hidden:`MultiheadAttentionEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.MultiheadAttentionEncoder
    :members:

:hidden:`TransformerEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.TransformerEncoder
    :members:

:hidden:`BERTEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.BERTEncoder
    :members:

:hidden:`RoBERTaEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.RoBERTaEncoder
    :members:

:hidden:`GPT2Encoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.GPT2Encoder
    :members:

:hidden:`XLNetEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.XLNetEncoder
    :members:
    :exclude-members: _forward

:hidden:`Conv1DEncoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.Conv1DEncoder
    :members:

:hidden:`EncoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.EncoderBase
    :members:

:hidden:`RNNEncoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.RNNEncoderBase
    :members:

:hidden:`default_transformer_poswise_net_hparams`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.modules.default_transformer_poswise_net_hparams

Decoders
========

.. spelling::
    Luong
    Bahdanau
    Gumbel

:hidden:`RNNDecoderBase`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.RNNDecoderBase
    :members:
    :exclude-members: initialize,step,finalize,output_size

:hidden:`BasicRNNDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.BasicRNNDecoder
    :members:
    :exclude-members: initialize,step,finalize,output_size

:hidden:`BasicRNNDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.BasicRNNDecoderOutput
    :members:

:hidden:`AttentionRNNDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.AttentionRNNDecoder
    :members:
    :exclude-members: initialize,step,finalize,output_size

:hidden:`AttentionRNNDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.AttentionRNNDecoderOutput
    :members:

:hidden:`GPT2Decoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.GPT2Decoder
    :members:

:hidden:`XLNetDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.XLNetDecoder
    :members:
    :exclude-members: initialize,step,finalize,_create_input

:hidden:`XLNetDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.XLNetDecoderOutput
    :members:

:hidden:`TransformerDecoder`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.TransformerDecoder
    :members:

:hidden:`TransformerDecoderOutput`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.TransformerDecoderOutput
    :members:

:hidden:`Helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.Helper
    :members:

:hidden:`TrainingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.TrainingHelper
    :members:
    :exclude-members: initialize,sample,next_inputs

:hidden:`EmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.EmbeddingHelper
    :members:
    :exclude-members: initialize

:hidden:`GreedyEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.GreedyEmbeddingHelper
    :members:
    :exclude-members: sample

:hidden:`SampleEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.SampleEmbeddingHelper
    :members:
    :exclude-members: sample
    
:hidden:`TopKSampleEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.TopKSampleEmbeddingHelper
    :members:

:hidden:`TopPSampleEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.TopPSampleEmbeddingHelper
    :members:

:hidden:`SoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.SoftmaxEmbeddingHelper
    :members:
    :exclude-members: sample_ids_shape,next_inputs

:hidden:`GumbelSoftmaxEmbeddingHelper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.GumbelSoftmaxEmbeddingHelper
    :members:

:hidden:`get_helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: texar.torch.modules.get_helper


Classifiers
============

:hidden:`BERTClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.BERTClassifier
    :members:

:hidden:`RoBERTaClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.RoBERTaClassifier
    :members:

:hidden:`GPT2Classifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.GPT2Classifier
    :members:

:hidden:`Conv1DClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.Conv1DClassifier
    :members:

:hidden:`XLNetClassifier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.XLNetClassifier
    :members:

Regressors
==========

:hidden:`XLNetRegressor`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.XLNetRegressor
    :members:

Pre-trained
===========

.. spelling::
    pooler

:hidden:`PretrainedMixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.PretrainedMixin
    :members:
    :private-members:
    :exclude-members: _name_to_variable

:hidden:`PretrainedBERTMixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.PretrainedBERTMixin
    :members:

:hidden:`PretrainedRoBERTaMixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.PretrainedRoBERTaMixin
    :members:

:hidden:`PretrainedGPT2Mixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.PretrainedGPT2Mixin
    :members: _init_from_checkpoint

:hidden:`PretrainedXLNetMixin`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.PretrainedXLNetMixin
    :members:

Connectors
==========

.. spelling::
    reparameterized
    reparameterization
    reparameterizable
    Reparameterization
    Autoencoders
    mlp

:hidden:`ConnectorBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.ConnectorBase
    :members:

:hidden:`ConstantConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.ConstantConnector
    :members:

:hidden:`ForwardConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.ForwardConnector
    :members:

:hidden:`MLPTransformConnector`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.MLPTransformConnector
    :members:

Networks
========

:hidden:`FeedForwardNetworkBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.FeedForwardNetworkBase
    :members:

:hidden:`FeedForwardNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.FeedForwardNetwork
    :members: default_hparams,forward,append_layer,has_layer,layer_by_name,layers_by_name,layers,layer_names

:hidden:`Conv1DNetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: texar.torch.modules.Conv1DNetwork
    :members: default_hparams,forward,append_layer,has_layer,layer_by_name,layers_by_name,layers,layer_names
