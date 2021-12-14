import argparse
import json
import os
from typing import List

from tokenizers import Tokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


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
    def load_tokenizer(filepath) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast(tokenizer_file=filepath)

    def build_transform_dir_structure(self):
        [os.makedirs(os.path.join(self.transform_output_dir, dir), exist_ok=True) for dir in ['transform_fn', 'transformed_data']]

if __name__ == '__main__':

    from trainer.constants import SpecialTokens
    from tokenizers import ByteLevelBPETokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--example-gen', type=str, dest='example_gen')
    parser.add_argument('--transform', type=str, dest='transform', help='destination to write example outputs')
    args = parser.parse_args()

    example_gen = args.example_gen
    transform_dir = args.transform

    files = [os.path.join(example_gen, filename) for filename in os.listdir(example_gen)]
    special_tokens = [SpecialTokens.BOS, SpecialTokens.EOS]

    tokenizer_instance = ByteLevelBPETokenizer()
    tokenizer_filename = 'tokenizer'
    preprocessed_features_filename = "tokenized_data.txt"
    tokenizer = Tokenizer(tokenizer=tokenizer_instance, special_tokens=special_tokens, transform_output_dir=transform_dir)
    tokenizer.fit(files=files, tokenizer_filename=tokenizer_filename)
    tokenizer.transform(files=files, processed_data_filename=preprocessed_features_filename)
