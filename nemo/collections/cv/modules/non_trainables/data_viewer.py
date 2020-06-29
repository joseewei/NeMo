# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import numpy as np
from typing import List, Optional

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.configuration_error import ConfigurationError
from nemo.utils.decorators import add_port_docs

__all__ = ['DataViewer']


class DataViewer(NonTrainableNM):
    """
    Class used for viewing the string stream (for now)

    """

    def __init__(self, log_frequency: int = 1, sample_number: int = -1, name: Optional[str] = None):
        """
        Initializes the object.

        Args:
            log_frequency: frequency indicating how often module will log the inputs (DEFAULT: 1)
            sample_number: index of the sample from the batch (DEFAULT: -1 = random)
            name: Name of the module (DEFAULT: None)
        """
        # Call constructors of parent classes.
        NonTrainableNM.__init__(self, name=name)

        self._log_frequency = log_frequency
        self._sample_number = sample_number

    @property
    @add_port_docs()
    def input_ports(self):
        """
        Creates definitions of input ports.
        """
        return {
            "indices": NeuralType(tuple('B'), elements_type=Index()),
            "labels": NeuralType(tuple('B'), elements_type=StringLabel()),  # Labels is string!
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Creates definitions of output ports - no outputs.
        """
        return {}

    def forward(self, indices, labels):
        """
        Logs inputs.

        Args:
            inputs: a tensor [BATCH_SIZE x ...]

        """
        # Get sample number.
        if self._sample_number < 0:
            # Random
            sample_number = np.random.randint(0, len(indices))
        else:
            sample_number = self._sample_number

        # Generate displayed string.
        absent_streams = []
        disp_str = "Showing selected streams for sample {} (index: {}):\n".format(
            sample_number, indices[sample_number]
        )
        disp_str += " '{}': {}\n".format("labels", labels[sample_number])

        # Display.
        logging.info(disp_str)
        print("\n\n {} \n\n".format(disp_str))

        # Return empty
        return {}

