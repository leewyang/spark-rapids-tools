import os
import glob
import shutil
from pathlib import Path

import pandas as pd


from spark_rapids_tools.tools.qualx.preprocess import (
    load_datasets,
    expected_raw_features,
    impute
)


def test_load_datasets(get_test_resources_path):
    # Load the datasets
    os.environ['QUALX_DATA_DIR'] = str(get_test_resources_path / 'eventlogs')
    os.environ['QUALX_CACHE_DIR'] = str(get_test_resources_path / 'qualx_cache')
    datasets_dir = str(get_test_resources_path / 'datasets')

    # remove cache if already present
    cache_dir = Path(os.environ['QUALX_CACHE_DIR'])
    if cache_dir.exists():
        # for CI/CD, remove cache if exists
        shutil.rmtree(cache_dir)
        # for development, remove preprocessed files only, keeping profiler CSV files
        # preprocessed_files = glob.glob(str(cache_dir) + '/**/preprocessed.parquet')
        # for f in preprocessed_files:
        #     os.remove(f)

    # TODO: find better way to get the path to the jar
    tools_jar_path = glob.glob(
        str(get_test_resources_path / '../../../../core/target/rapids-4-spark-tools_*.jar')
    )
    tools_jar_path = [p for p in tools_jar_path if not any([s in p for s in ['javadoc.jar', 'sources.jar']])]
    assert len(tools_jar_path) == 1
    os.environ['SPARK_RAPIDS_TOOLS_JAR'] = tools_jar_path[0]

    all_datasets, profile_df = load_datasets(datasets_dir)

    # Basic assertions
    assert isinstance(all_datasets, dict)
    assert 'nds_local' in all_datasets

    assert isinstance(profile_df, pd.DataFrame)
    assert not profile_df.empty
    # assert profile_df.shape == (194, 127)
    assert set(profile_df.columns) == expected_raw_features


def test_impute():
    # Test impute function
    input_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })

    imputed_df = impute(input_df)
    df_columns = set(imputed_df.columns)

    # should not have extra columns
    assert 'col1' not in df_columns
    assert 'col2' not in df_columns

    # should have all expected raw features
    assert df_columns == expected_raw_features
