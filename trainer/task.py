from torch.utils import data
from trainer import model
import torch
from torch.utils.data import DataLoader
from trainer.model import loss_fn
import math
import numpy as np



class TrainerConfig:
    max_epochs = 10
    batch_size = 64
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


class Trainer:

    def __init__(self, model, train_dataset, validation_dataset, config: TrainerConfig) -> None:
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config
        self.best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay

        device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(device)

    def save_model(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.config.ckpt_path)


    def run_epoch(self, model, optimizer=None, split='train'):
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

        for X, y  in dataloader:
            # Backpropagation
            optimizer.zero_grad()

            X = X.to(self.device)
            y = y.to(self.device)

            logits = model(X)
            loss = loss_fn(logits, y)
            loss = loss.mean()
            losses.append(loss)


            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                optimizer.step()

                if self.config.lr_decay:
                    self.decay_learning_rate(y, optimizer)

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

    def train(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=self.config.learning_rate, betas=self.config.betas)

        for _ in range(self.config.max_epochs):

            self.run_epoch(raw_model, optimizer, 'train')
            if self.validation_dataset is not None:
                validation_loss = self.run_epoch('validation')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.validation_dataset is None or validation_loss < self.best_loss
            if self.config.ckpt_path is not None and good_model:
                self.best_loss = validation_loss
                self.save_model()