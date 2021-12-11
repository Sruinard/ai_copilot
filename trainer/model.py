import torch
from torch import nn
from torch.nn import functional as F

class GPTConfig:

    embedding_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1

class GPT2Config(GPTConfig):
    embedding_size = 768
    n_heads = 12
    n_layers = 12
    max_sequence_length = 256

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        assert config.embedding_size % config.n_heads == 0
        self.values = nn.Linear(config.embedding_size, config.embedding_size)
        self.keys = nn.Linear(config.embedding_size, config.embedding_size)
        self.queries = nn.Linear(config.embedding_size, config.embedding_size)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        self.fc = nn.Linear(config.embedding_size, config.embedding_size)

        # triangular lower filled with ones
        self.causal_mask = torch.tril(torch.ones(config.max_sequence_length, config.max_sequence_length)).view(1, 1, config.embedding_size, config.embedding_size)

        self.n_heads = config.n_heads

    def forward(self, x):
        N, sequence_length, embed_size = x.shape

        # B, sequence_length, n_heads, head_size
        values = self.values(x).view(N, sequence_length, self.n_heads, embed_size // self.n_heads)
        keys = self.keys(x).view(N, sequence_length, self.n_heads, embed_size // self.n_heads)
        queries = self.queries(x).view(N, sequence_length, self.n_heads, embed_size // self.n_heads)

        attention = torch.einsum('nqhd,nkhd->nhqk', [queries, keys]) * ( 1 / torch.sqrt(keys.shape[0]))
        attention = attention.masked_fill(self.causal_mask == 0, float('-1e20'))
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        out = torch.einsum('nhqk,nkhd->nqhd', [attention, values]).reshape(N, sequence_length, embed_size)
        out = self.residual_dropout(out)
        return out

class Decoder(nn.Module):

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.embedding_size)
        self.norm_2 = nn.LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size * 4),
            nn.GELU(),
            nn.Linear(config.embedding_size * 4, config.embedding_size),
            nn.Dropout(config.residual_dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.feed_forward(self.norm_2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.positional_embedding = nn.parameter(1, config.max_sequence_length, config.embedding_size)
        self.dropout = nn.Dropout(config.embedding_dropout)

        self.layers = nn.ModuleList(
            [
                Decoder(config) for _ in config.n_layers
            ]
        )

        self.layer_norm = nn.LayerNorm(config.embedding_size)
        self.fc = nn.Linear(config.embedding_size, config.vocab_size, bias=False)


    def forward(self, x):
        _, sequence_length = x.shape

        # word_embedding.shape = batch_size x sequence_length x embedding_dim
        word_embedding = self.word_embedding(x)
        # positional_encoding.shape = 1 x sequence_length x embedding_dim
        positional_encoding = self.positional_embedding[:, :sequence_length, :]
        x = self.dropout(word_embedding + positional_encoding)
        x = self.layers(x)
        x = self.layer_norm(x)
        logits = self.fc(x)
        return logits

def loss_fn(logits, targets):
    # flatten logits and target and compute cross entropy
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
    return loss