from typing import List
from tokenizers import Tokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import json

import os

class Tokenizer:

    def __init__(self, tokenizer: Tokenizer, special_tokens: List[str], transform_output_dir):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.transform_output_dir = transform_output_dir
        self.build_transform_dir_structure()

    def save_tokenizer(self, filepath):
        self.tokenizer.save(filepath)

    def fit(self, files, tokenizer_filename):
        self.tokenizer.train(
            files=files,
            min_frequency=2,
            special_tokens=self.special_tokens
        )
        self.save_tokenizer(os.path.join(self.transform_output_dir, "transform_fn", tokenizer_filename))

    def transform(self, files, processed_data_filename: str):
        path_to_serialize_examples = os.path.join(self.transform_output_dir, "transformed_data", processed_data_filename)
        if os.path.exists(path_to_serialize_examples):
            os.remove(path_to_serialize_examples)
        for file in files:
            with open(file, 'r') as f:
                data = f.read()
                with open(path_to_serialize_examples, "w+") as writer:
                    encoded_data = self.tokenizer.encode(data)
                    writer.write(json.dumps(encoded_data.ids))

    @staticmethod
    def load_tokenizer(filepath):
        return PreTrainedTokenizerFast(tokenizer_file=filepath)

    def build_transform_dir_structure(self):
        [os.makedirs(os.path.join(self.transform_output_dir, dir), exist_ok=True) for dir in ['transform_fn', 'transformed_data']]