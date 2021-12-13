import gzip
import json
import os
from typing import List

from trainer.constants import SpecialTokens


class Dataset:

    def __init__(self, raw_files: List[str], dataset_dest: str):
        self.raw_files = raw_files
        self.dataset_dest = dataset_dest

    def build_dataset(self):
        if os.path.exists(self.dataset_dest):
            os.remove(self.dataset_dest)
        for filepath in self.raw_files:
            with gzip.open(filepath, 'r') as f:
                lines = [json.loads(line) for line in f]
            with open(self.dataset_dest, "w+") as f:
                for line in lines:
                    code = line['code']
                    if is_valid_docstring(code):
                        processed_code = include_bos_and_eos(code)
                        f.write(processed_code)
                    else:
                        continue

def is_valid_docstring(code: str):
    is_valid = False
    n_sections_make_up_def_docstring_and_code = 3
    if len(code.split("'''")) == n_sections_make_up_def_docstring_and_code or len(code.split('"""')) == n_sections_make_up_def_docstring_and_code:
        is_valid = True
    return is_valid

def include_bos_and_eos(code: str):
    n_sections_make_up_def_docstring_and_code = 3
    split_on_single_quote_docstring = code.split("'''")
    split_on_double_quote_docstring = code.split('"""')

    code_as_sections= split_on_single_quote_docstring if len(split_on_single_quote_docstring) == n_sections_make_up_def_docstring_and_code else split_on_double_quote_docstring
    processed_code = '"""'.join(code_as_sections[:2]) + '"""' + SpecialTokens.BOS + " " + code_as_sections[2] + SpecialTokens.EOS + "\n"
    return processed_code
