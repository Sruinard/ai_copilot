from trainer import model
import torch



class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
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

    def __init__(self, model, train_dataset, validation_dataset, config) -> None:
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config

        device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(device)


    