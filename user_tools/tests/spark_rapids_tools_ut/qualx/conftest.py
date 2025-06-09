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

"""Helper functions for qualx unit tests"""

import random
import numpy as np
import pandas as pd

from spark_rapids_tools.tools.qualx.model import expected_raw_features


def generate_test_data(label_col: str, seed: int = 0) -> pd.DataFrame:
    # Create test data with 200 rows (100 CPU/GPU pairs)
    np.random.seed(seed)
    random.seed(seed)

    # fill in random features and random labels
    fixed_features = set(['appName', 'description', 'runType', 'scaleFactor', 'sqlID'])
    random_features = expected_raw_features() - fixed_features
    features = {}
    for feature in random_features:
        features[feature] = np.random.rand(200)
    df = pd.DataFrame(features)
    df[label_col] = np.random.rand(200)

    # fill in fixed features
    df['appName'] = 'test_app'
    df['description'] = 'testing'
    df['scaleFactor'] = 1
    df.loc[0:99, 'sqlID'] = range(100)
    df.loc[0:99, 'runType'] = 'CPU'
    df.loc[100:199, 'sqlID'] = range(100)
    df.loc[100:199, 'runType'] = 'GPU'

    return df
