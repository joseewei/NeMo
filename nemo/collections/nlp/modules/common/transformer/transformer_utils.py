# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from omegaconf.dictconfig import DictConfig
from nemo.collections.nlp.modules.common.transformer.transformer import TransformerDecoderNM, TransformerEncoderNM
from nemo.utils import logging
from typing import Optional, Union


def get_nemo_transformer_model(
    name: Optional[str] = None,
    config_dict: Optional[Union[dict, DictConfig]] = None,
    config_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    encoder: bool = True,
) -> Union[TransformerEncoderNM, TransformerDecoderNM]:
    """Returns NeMo transformer. Module configuration will be taken in the following order of precedence:
    config_file > config_dict > pretrained_model_name.
    The following configurations are mandatory:
        vocab_size: int
        hidden_size: int
        num_layers: int
        inner_size: int
    and must be specified if using config_dict or config_file.

    Args:
        name (Optional[str]): model name to download from NGC
        config_dict (Optional[dict], optional): model configuration parameters. Defaults to None.
        config_file (Optional[str], optional): path to json file containing model configuration. Defaults to None.
        checkpoint_file (Optional[str], optional): load weights from path to local checkpoint. Defaults to None.
        encoder (bool, optional): True will use EncoderTransformerNM, False will use DecoderTransformerNM. Defaults to True.
    """
    cfg = None
    if config_file is not None:
        logging.error('Configuring NeMo Transformer from json has not been implemented yet. Please use config_dict.')
        # raise ValueError(
        #     "Instantiate NeMo Transformer from json has not been implemented yet. Please use config_dict."
        # )

    elif config_dict is not None:
        assert (
            config_dict.get('vocab_size') is not None
            and config_dict.get('hidden_size') is not None
            and config_dict.get('num_layers') is not None
            and config_dict.get('inner_size') is not None
        ), 'vocab_size, hidden_size, num_layers, and inner_size must are mandatory arguments'
        cfg = config_dict

    elif name is not None:
        logging.info(f'NeMo transformers cannot be loaded from NGC yet. Using {name} with configuration {cfg}.')
    if encoder:
        model = TransformerEncoderNM(**cfg)
    else:
        model = TransformerDecoderNM(**cfg)

    if checkpoint_file is not None:
        if os.path.isfile(checkpoint_file):
            logging.info(f'Checkpoint found at {checkpoint_file} will be used to restore weights.')
            logging.error(f'Loading pretrained NeMo transformer has not been implemented yet.')
        else:
            logging.error(f'Checkpoint not found at {checkpoint_file}.')
    return model
