.. _machine_translation:

Machine Translation Models
=========================
Machine translation is the task of translating text from one language to another. For example, from English to Spanish.
Models are based on the Transformer sequence-to-sequence architecture :cite:`nlp-textclassify-vaswani2017attention`.

An example script on how to train the model can be found here: `NeMo/examples/nlp/machine_translation/enc_dec_nmt.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/machine_translation/enc_dec_nmt.py>`__.
The default configuration file for the model can be found at: `NeMo/examples/nlp/machine_translation/conf/aayn_base.yaml <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/machine_translation/conf/aayn_base.yaml>`__.

Quick Start
-----------

.. code-block:: python

    from nemo.collections.nlp.models import MTEncDecModel

    # to get the list of pre-trained models
    MTEncDecModel.list_available_models()

    # Download and load the a pre-trained to translate from English to Spanish
    model = MTEncDecModel.from_pretrained("nmt_en_es_transformer12x2")

    # Translate a sentence or list of sentences
    translations = model.translate(["Hello!"], source_lang="en", target_lang="es")

Available Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: *Pretrained Models*
   :widths: 5 10
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
   * - English -> German
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_de_transformer12x2
   * - German -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_de_en_transformer12x2
   * - English -> Spanish
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_es_transformer12x2
   * - Spanish -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_es_en_transformer12x2
   * - English -> French
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_fr_transformer12x2
   * - French -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_fr_en_transformer12x2
   * - English -> Russian
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_ru_transformer6x6
   * - Russian -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_ru_en_transformer6x6
   * - English -> Chinese
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_en_zh_transformer6x6
   * - Chinese -> English
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:nmt_zh_en_transformer6x6

Data Format
-----------
Supervised machine translation models require parallel corpora which comprise many examples of in a source sentences and their corresponding translation in a target language.
We use parallel data formatted as seperate text files for source and target language where sentences in corresponding files are aligned like in the table below.

.. list-table:: *Parallel Coprus*
   :widths: 5 10
   :header-rows: 1

   * - train.english.txt
     - train.spanish.txt
   * - Hello .
     - Hola .
   * - Thank you .
     - Gracias .
   * - You can now translate from English to Spanish in NeMo .
     - Ahora puedes traducir del inglés al español en NeMo .

It is common practice to apply data cleaning, normalization and tokenization to the data prior to training a translation model.

Tarred Datasets for Large Corpora
------------------


Model Training
--------------

.. code::

    python examples/nlp/text_classification/text_classification_with_bert.py \
        model.training_ds.file_path=<TRAIN_FILE_PATH> \
        model.validation_ds.file_path=<VALIDATION_FILE_PATH> \
        trainer.max_epochs=50 \
        trainer.gpus=[0,1] \
        optim.name=adam \
        optim.lr=0.0001 \
        model.nemo_path=<NEMO_FILE_PATH>

Model Arguments
^^^^^^^^^^^^^^^
The following table lists some of the model's parameters you may use in the config files or set them from command line when training a model:

+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   |   **Default**                                  | **Description**                                                                                              |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.class_labels.class_labels_file      | string          | null                                           | Path to an optional file containing the labels; each line is the string label corresponding to a label       |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.num_classes                 | int             | ?                                              | Number of the categories or classes, 0 < Label <num_classes                                                  |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.do_lower_case               | boolean         | true for uncased models, false for cased       | Specifies if inputs should be made lower case, would be set automatically if pre-trained model is used       |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.max_seq_length              | int             | 256                                            | Maximum length of the input sequences.                                                                       |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.class_balancing             | string          | null                                           | null or 'weighted_loss'. 'weighted_loss' enables the weighted class balancing to handle unbalanced classes   |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.dataset.use_cache                   | boolean         | false                                          | uses a cache to store the processed dataset, you may use it for large datasets for speed up                  |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.classifier_head.num_output_layers   | integer         | 2                                              | Number of fully connected layers of the Classifier on top of Bert model                                      |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| model.classifier_head.fc_dropout          | float           | 0.1                                            | Dropout ratio of the fully connected layers                                                                  |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.file_path   | string          | ??                                             | Path of the training '.tsv file                                                                              |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------------+
| {training,validation,test}_ds.batch_size  | integer         | 32                                             | Data loader's batch size                                                                                     |
+-------------------------------------------+-----------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------------+
| {training,validation,test}_ds.num_workers | integer         | 2                                              | Number of worker threads for data loader                                                                     |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.shuffle     | boolean         | true (training), false (test and validation)   | Shuffles data for each epoch                                                                                 |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.drop_last   | boolean         | false                                          | Specifies if last batch of data needs to get dropped if it is smaller than batch size                        |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.pin_memory  | boolean         | false                                          | Enables pin_memory of PyTorch's data loader to enhance speed                                                 |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+
| {training,validation,test}_ds.num_samples | integer         | -1                                             | Number of samples to be used from the dataset; -1 means all samples                                          |
+-------------------------------------------+-----------------+------------------------------------------------+--------------------------------------------------------------------------------------------------------------+


Training Procedure
^^^^^^^^^^^^^^^^^^

After each epoch, you should see a summary table of metrics on the validation set which include the following metrics:

* :code:`Precision`
* :code:`Recall`
* :code:`F1`

At the end of training, NeMo will save the last checkpoint at the path specified in '.nemo' format.

Model Evaluation and Inference
------------------------------

After saving the model in '.nemo' format, you may load the model and perform evaluation or inference on the model.
You may find some example in the example script: `NeMo/examples/nlp/text_classification/text_classification_with_bert.py <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/text_classification/text_classification_with_bert.py>`__

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-TEXTCLASSIFY
    :keyprefix: nlp-textclassify-
