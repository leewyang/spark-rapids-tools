# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Set sample weights based on label values."""

import logging
import numpy as np
import pandas as pd
from spark_rapids_tools.tools.qualx.config import get_config


logger = logging.getLogger(__name__)

expected_model_features = set()  # note: 'weight' is not a true feature


def extract_model_features(features: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Set 'weight' column based on label values.

    Parameters
    ----------
    features: pd.DataFrame
        Input dataframe of model features
    threshold: float
        Threshold for true positives
    positive: float
        Weight to apply to true positives
    """
    # pylint: disable=unused-argument
    threshold = kwargs.get('threshold', 1.0)
    positive_weight = kwargs.get('positive', None)

    cfg = get_config()
    label = f'{cfg.label}_speedup'

    if positive_weight is None:
        # compute the positive weight from the ratio of negatives to positives to balance the classes
        num_positives = (features[label] > threshold).sum()
        num_negatives = (features[label] <= threshold).sum()
        positive_weight = np.max([np.floor(num_negatives / num_positives), 1.0])

    features['weight'] = np.where(features[label] > threshold, positive_weight, 1.0)

    return features
