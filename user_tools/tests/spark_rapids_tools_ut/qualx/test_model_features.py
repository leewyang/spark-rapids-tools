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

"""Test model features plugins"""

from spark_rapids_tools.tools.qualx.model import extract_model_features
from spark_rapids_tools.tools.qualx.config import get_config
from ..conftest import SparkRapidsToolsUT
from .conftest import generate_test_data


class TestModelFeatures(SparkRapidsToolsUT):
    """Test class for model features plugins"""

    def test_weight_label(self, monkeypatch) -> None:
        """Test training models with different sample weights"""
        # Create test data
        df = generate_test_data('Duration')  # Use default label

        # Mock config without any model features plugins
        default_config = get_config(reload=True)
        default_config.model_features = []
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: default_config)
        features, _, label_col = extract_model_features(df)
        assert 'weight' not in features.columns

        # Check number of positives and negatives @ threshold=1.0
        threshold = 1.0
        num_positives = (features[label_col] > threshold).sum()
        num_negatives = (features[label_col] <= threshold).sum()
        assert num_positives > num_negatives  # 55 > 45

        # Mock config with model features plugin for sample weights @ threshold=1.0
        weighted_config = get_config(reload=True)
        weighted_config.model_features = [{'path': 'weight_label.py', 'args': {'threshold': threshold}}]
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: weighted_config)
        features, feature_cols, label_col = extract_model_features(df)
        assert 'weight' in features.columns
        assert 'weight' not in feature_cols
        assert features.weight.unique() == [1.0]  # 1.0 for all rows, since num_positives > num_negatives

        # Check number of positives and negatives @ threshold=5.0
        threshold = 5.0
        num_positives = (features[label_col] > threshold).sum()
        num_negatives = (features[label_col] <= threshold).sum()
        assert num_positives < num_negatives  # 8 < 92

        # Mock config with model features plugin for sample weights @ threshold=5.0
        weighted_config = get_config(reload=True)
        weighted_config.model_features = [{'path': 'weight_label.py', 'args': {'threshold': threshold}}]
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: weighted_config)
        features, feature_cols, label_col = extract_model_features(df)
        assert 'weight' in features.columns
        assert set(features.weight.unique()) == {1.0, 11.0}
        assert features.loc[features['weight'] > 1.0].shape[0] == num_positives
        assert features.loc[features['weight'] == 1.0].shape[0] == num_negatives
