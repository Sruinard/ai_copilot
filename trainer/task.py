import argparse
from genericpath import exists
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trainer.model import loss_fn


class TextToCodeDataset(Dataset):

    def __init__(self, transform_dir, sequence_length) -> None:
        super().__init__()
        # TODO remove reduced file size
        self.data = self._load_data(transform_dir)[:500]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        word_idx = self.data[idx:idx + self.sequence_length + 1]
        x = torch.tensor(word_idx[:-1], dtype=torch.long)
        y = torch.tensor(word_idx[1:], dtype=torch.long)
        return x, y

    @staticmethod
    def _load_data(transform_dir):
        transformed_data_dir = os.path.join(transform_dir, 'transformed_data')
        data_filename = os.listdir(transformed_data_dir)[0]

        with open(os.path.join(transformed_data_dir, data_filename), 'r') as f:
            data = json.loads(f.read())
        return data

class TrainerConfig:
    max_epochs = 10
    batch_size = 32
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, validation_dataset, config: TrainerConfig) -> None:
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config

        self.epoch = 0
        self.best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_model(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        model_dir = os.path.split(self.config.ckpt_path)[0]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)


    def run_epoch(self, model, optimizer=None, epoch=0, split='train'):
        is_train = split == 'train'
        data = self.train_dataset if is_train else self.validation_dataset
        dataloader = DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        losses = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader)) if is_train else enumerate(dataloader)
        for it, (X, y)  in pbar:
            X = X.to(self.device)
            y = y.to(self.device)

            logits = model(X)
            loss = loss_fn(logits, y)
            loss = loss.mean()
            losses.append(loss.item())


            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                optimizer.step()

                if self.config.lr_decay:
                   lr =  self.decay_learning_rate(y, optimizer)
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                return test_loss

    def decay_learning_rate(self, y, optimizer: torch.optim.Optimizer):
        if self.config.lr_decay:
            self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
            if self.tokens < self.config.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.config.warmup_tokens) / float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.config.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.config.learning_rate

        return lr

    def train(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=self.config.learning_rate, betas=self.config.betas)

        for epoch in range(self.config.max_epochs):

            self.run_epoch(raw_model, optimizer, epoch=epoch, split='train')
            if self.validation_dataset is not None:
                validation_loss = self.run_epoch(raw_model, epoch=epoch, split='validation')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.validation_dataset is None or validation_loss < self.best_loss
            if self.config.ckpt_path is not None and good_model:
                self.best_loss = validation_loss
                self.save_model()


if __name__ == '__main__':
    from trainer.preprocessing import Tokenizer
    from trainer.model import GPT, GPT2Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=str, dest='transform', help='destination to write example outputs')
    parser.add_argument('--serving-dir', type=str, dest='serving_dir', help='destination to write example outputs')

    args = parser.parse_args()

    transform_dir = args.transform
    serving_dir = args.serving_dir
    tokenizer_filepath = os.path.join(transform_dir, 'transform_fn', 'tokenizer')
    ckpt_path = os.path.join(serving_dir, 'GPT.pt')

    tokenizer = Tokenizer.load_tokenizer(filepath=tokenizer_filepath)
    sequence_length = 128
    train_dataset = TextToCodeDataset(transform_dir, sequence_length)
    mconf = GPT2Config(
        tokenizer.vocab_size,
        embedding_size=768,
        n_heads=12,
        n_layers=3,
        max_sequence_length=sequence_length
    )
    model = GPT(mconf)
    tconf = TrainerConfig(
        max_epochs=2,
        batch_size=32,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=2*len(train_dataset)*sequence_length,
        num_workers=4,
        ckpt_path=ckpt_path
    )
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()