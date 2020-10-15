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

import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="quartznet_15x5")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    asr_model.change_vocabulary(new_vocabulary=cfg.model.labels)

    asr_model._trainer = trainer
    asr_model.setup_optimization(cfg.model.optim)
    # Point to the data we'll use for fine-tuning as the training set
    asr_model.setup_training_data(train_data_config=cfg.model.train_ds)
    # Point to the new validation data for fine-tuning
    asr_model.setup_validation_data(val_data_config=cfg.model.validation_ds)
    trainer.fit(asr_model)
    asr_model.save_to(save_path="QuartzNet15x5")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
