import pandas as pd
import pytest

from pathlib import Path
from spark_rapids_tools.tools.qualx.util import (
    ensure_directory,
    find_paths,
    find_eventlogs,
    get_dataset_platforms,
    compute_accuracy,
    compute_precision_recall,
    load_plugin,
    random_string,
    print_summary,
    print_speedup_summary,
    create_row_with_default_speedup,
    write_csv_reports,
    log_fallback,
)
from unittest.mock import MagicMock, patch


def test_ensure_directory():
    with patch('os.makedirs') as mock_makedirs:
        ensure_directory('/path/to/dir')
        mock_makedirs.assert_called_once_with(Path('/path/to/dir'), exist_ok=True)


def test_find_paths():
    with patch('os.walk') as mock_walk:
        with patch('os.path.isdir') as mock_isdir:
            mock_walk.return_value = [('/path/to/dir', ['dir1', 'dir2'], ['file1', 'file2'])]
            mock_isdir.return_value = True
            result = find_paths('/path/to/dir')
            assert result == ['/path/to/dir/file1', '/path/to/dir/file2']


def test_find_eventlogs():
    with patch('glob.glob') as mock_glob:
        mock_glob.return_value = ['/eventlog/log1', '/eventlog/log2']
        result = find_eventlogs('/eventlog/*.log')
        assert result == ['/eventlog/log1', '/eventlog/log2']


def test_get_dataset_platforms():
    # supported platforms
    with patch('os.listdir') as mock_listdir:
        mock_listdir.return_value = ['databricks-aws', 'onprem']
        result = get_dataset_platforms('/path/to/datasets')
        assert result == (['databricks-aws', 'onprem'], '/path/to/datasets')

    # unsupported platform
    with pytest.raises(ValueError):
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ['fake_platform']
            result = get_dataset_platforms('/path/to/datasets')


def test_compute_accuracy():
    data = pd.DataFrame({
        'label': [1, 2, 3],
        'pred1': [1, 3, 2],
        'pred2': [2, 1, 3],
        'weight': [1, 2, 3],
    })
    result = compute_accuracy(data, 'label', {'Pred1': 'pred1', 'Pred2': 'pred2'}, 'weight')
    assert result['Pred1']['MAPE'] == pytest.approx(5/18)
    assert result['Pred1']['wMAPE'] == pytest.approx(1/3)
    assert result['Pred1']['dMAPE'] == pytest.approx(1/3)
    assert result['Pred2']['MAPE'] == pytest.approx(0.5)
    assert result['Pred2']['wMAPE'] == pytest.approx(1/3)
    assert result['Pred2']['dMAPE'] == pytest.approx(1/3)


def test_compute_precision_recall():
    data = pd.DataFrame({
        'label': [1, 2, 3],
        'pred1': [1, 3, 2],
        'pred2': [2, 1, 3]
    })
    result = compute_precision_recall(data, 'label', {'Pred1': 'pred1', 'Pred2': 'pred2'}, 1.5)
    expected_result = (
        {'Pred1': 1.0, 'Pred2': 0.5},
        {'Pred1': 1.0, 'Pred2': 0.5}
    )
    assert result == expected_result


def test_load_plugin():
    with pytest.raises(FileNotFoundError):
        load_plugin('/path/to/missing_plugin.py')
    # TODO: load dummy plugin


def test_random_string():
    result = random_string(10)
    assert len(result) == 10


def test_print_summary():
    data = pd.DataFrame({
        'appName': ['app1', 'app2'],
        'appId': ['id1', 'id2'],
        'appDuration': [1, 2],
        'Duration': [1, 2],
        'Duration_pred': [1, 2],
        'Duration_supported': [1, 2],
        'fraction_supported': [1, 2],
        'appDuration_pred': [1, 2],
        'speedup': [1, 2]
    })
    with patch('builtins.print') as mock_print:
        print_summary(data)
        mock_print.assert_called_once()


def test_print_speedup_summary():
    data = pd.DataFrame({
        'appDuration': [1, 2],
        'appDuration_pred': [1, 2]
    })
    with patch('builtins.print') as mock_print:
        print_speedup_summary(data)
        mock_print.assert_called()


def test_create_row_with_default_speedup():
    data = pd.Series({
        'App Name': 'app1',
        'App ID': 'id1',
        'App Duration': 1
    })
    result = create_row_with_default_speedup(data)
    expected_result = pd.Series({
        'appName': 'app1',
        'appId': 'id1',
        'appDuration': 1,
        'Duration': 0,
        'Duration_pred': 0,
        'Duration_supported': 0,
        'fraction_supported': 0.0,
        'appDuration_pred': 1,
        'speedup': 1.0
    })
    pd.testing.assert_series_equal(result, expected_result)


def test_write_csv_reports():
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        write_csv_reports(pd.DataFrame(), pd.DataFrame(), {'perSql': {'path': '/sql/path'}, 'perApp': {'path': '/app/path'}})
        mock_to_csv.assert_any_call('/app/path')
        mock_to_csv.assert_any_call('/sql/path')


def test_log_fallback():
    logger = MagicMock()
    log_fallback(logger, ['app1', 'app2'], 'reason')
    logger.warning.assert_called_once()
