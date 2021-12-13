import os
import pytest
from utils.build_dataset import Dataset

@pytest.mark.usefixtures("vocab_data_in_gzip")
def test_build_dataset(raw_files, dataset_dest):
    ds = Dataset(raw_files=raw_files, dataset_dest=dataset_dest)
    ds.build_dataset()
    assert os.path.exists(dataset_dest)

@pytest.mark.usefixtures("vocab_data_in_gzip")
def test_read_vocab(raw_files, dataset_dest):
    ds = Dataset(raw_files=raw_files, dataset_dest=dataset_dest)
    ds.build_dataset()
    with open(dataset_dest, "r") as f:
        data = f.read()
    assert data == 'def hello_world(): """this def prints hello world"""<bos> \n print(\'hello world\')<eos>\n'
