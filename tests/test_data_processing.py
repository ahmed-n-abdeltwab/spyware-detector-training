import pytest
from src.data_processor import DataProcessor
import pandas as pd

def test_data_loading():
    processor = DataProcessor(data_dir='data')
    data = processor.load_data()
    assert isinstance(data, pd.DataFrame)
    assert 'labels' in data.columns
