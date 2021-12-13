import gzip
import json
import os
import pytest


@pytest.fixture
def raw_files():
    return ["./sample_text.txt.gz"]

@pytest.fixture
def dataset_dest():
    return "./vocab_dataset.txt"

@pytest.fixture(scope='function')
def vocab_data_in_gzip(raw_files, dataset_dest):
    # create data
    # raw_files = ["./sample_text.txt.gz"]
    sample = {
            'code': "def hello_world(): '''this def prints hello world'''\n print('hello world')"
    }
    with gzip.open(raw_files[0], 'wt') as f:
        f.write(json.dumps(sample))
    yield
    # remove data
    os.remove(dataset_dest)
    os.remove(raw_files[0])
