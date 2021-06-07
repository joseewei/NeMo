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

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import TextNormalizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="text_normalization_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {cfg.pretty()}')
    trainer = pl.Trainer(
        gpus=[0],
        precision=cfg.trainer.precision,
        amp_level=cfg.trainer.amp_level,
        logger=False,
        checkpoint_callback=False,
    )

    model = "/home/yzhang/code/NeMo/examples/nlp/text_normalization/model_8000.nemo"
    torch.set_grad_enabled(False)
    if not os.path.exists(model):
        raise ValueError(f'{model} not found.')
    model = TextNormalizationModel.restore_from(model)
    all_preds = model.infer(cfg.model.test_ds)

    num_samples = len(all_preds)
    predictions = [" ".join(all_preds[i]) for i in range(num_samples)]
    print(predictions)


if __name__ == '__main__':
    main()
