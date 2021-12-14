import os
import json

import argparse

if __name__ == "__main__":
    from tokenizers import ByteLevelBPETokenizer
    with open("./vocab.txt", 'w') as f:
        f.write('hello just tokenize this.')
    token = ByteLevelBPETokenizer()
    token.train(files=['./vocab.txt'], min_frequency=2, special_tokens=['<unk>'])