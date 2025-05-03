import json

class MiniGPTConfig:
    def __init__(self, path):
        with open(path, "r") as f:
            cfg = json.load(f)
        self.vocab_size      = cfg["vocab_size"]
        self.block_size      = cfg["block_size"]
        self.n_layers        = cfg["n_layers"]
        self.n_heads         = cfg["n_heads"]
        self.n_embd          = cfg["n_embd"]
        self.ffn_hidden_size = cfg["ffn_hidden_size"]
        self.dropout         = cfg["dropout"]
