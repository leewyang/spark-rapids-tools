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

"""Test qualx_model module"""

import pytest  # pylint: disable=import-error
import pandas as pd

from xgboost import Booster

from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.model import (
    extract_model_features,
    predict,
    split_features,
    train,
    compute_shapley_values,
)
from spark_rapids_tools.tools.qualx.util import get_abs_path, load_plugin
from ..conftest import SparkRapidsToolsUT
from .conftest import generate_test_data


class TestModel(SparkRapidsToolsUT):
    """Test class for qualx_model module"""

    @pytest.mark.parametrize('label', ['Duration', 'duration_sum'])
    def test_train_and_predict(self, monkeypatch, label) -> None:
        # mock os.environ
        monkeypatch.setenv('QUALX_LABEL', label)
        get_config(reload=True)  # reload config

        # Create test data
        df = generate_test_data(label)

        # Extract features
        plugin = load_plugin(get_abs_path('split_random.py', 'split_functions'))
        features, feature_cols, label_col = extract_model_features(df)
        features = split_features(features, split_functions={'default': plugin.split_function})

        # Assert label_col column exists
        assert label_col == f'{label}_speedup'
        assert label_col in features.columns

        # Assert label column is in features, but not in list of feature_cols
        assert label in features.columns
        assert label not in feature_cols

        model = train(features, feature_cols, label_col, n_trials=5)

        # Assert model is trained
        assert isinstance(model, Booster)
        assert model.num_boosted_rounds() > 0

        # Make predictions
        results = predict(model, features, feature_cols, label_col)

        # Assert predictions are made
        assert 'y_pred' in results.columns
        assert f'{label}_pred' in results.columns
        assert len(results) == 100

        # Compute SHAP values
        feature_importance, shap_values = compute_shapley_values(model, features)

        # Assert results
        assert isinstance(feature_importance, pd.DataFrame)
        assert isinstance(shap_values, pd.DataFrame)
        assert 'feature' in feature_importance.columns
        assert 'shap_value' in feature_importance.columns
        assert len(shap_values.columns) == len(feature_cols) + 1

    def test_train_with_sample_weight(self, monkeypatch) -> None:
        """Test training models with different sample weights"""
        # Create test data
        df = generate_test_data('Duration')  # Use default label
        plugin = load_plugin(get_abs_path('split_random.py', 'split_functions'))

        # Mock config for default model features plugins
        default_config = get_config(reload=True)
        default_config.model_features = []
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: default_config)

        # Train model without sample weight plugin
        features, feature_cols, label_col = extract_model_features(df)
        features = split_features(features, split_functions={'default': plugin.split_function})
        default_model = train(features, feature_cols, label_col, n_trials=5)

        # Mock config with model features plugin that weights positives
        weighted_config = get_config(reload=True)
        weighted_config.model_features = [{'path': 'weight_label.py', 'args': {'positive': 5.0}}]
        monkeypatch.setattr('spark_rapids_tools.tools.qualx.config.get_config', lambda: weighted_config)

        # Train model with sample weight plugin
        features, feature_cols, label_col = extract_model_features(df)
        features = split_features(features, split_functions={'default': plugin.split_function})
        assert 'weight' in features.columns

        # Train model with sample weights
        weighted_model = train(features, feature_cols, label_col, n_trials=5)

        # Make predictions with both models
        default_results = predict(default_model, features, feature_cols, label_col)
        weighted_results = predict(weighted_model, features, feature_cols, label_col)

        # Compare predictions
        # The weighted model should generally predict higher values for positive samples
        positive_mask = features[label_col] > 1.0
        positive_diff = weighted_results.loc[positive_mask, 'y_pred'] - default_results.loc[positive_mask, 'y_pred']
        assert positive_diff.mean() > 0, 'Weighted model should predict higher values for positive samples'
