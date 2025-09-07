import tensorflow as tf
import numpy as np


def reverse_tensor(seq_emb, seq_len):
    """
    genenrate seq, pos and neg based on seq_emb
    seq_emb: float, [B, SL, H], input sequence embeddings
    seq_len: int, [B, SL], input sequence length
    :return:
    """
    pass


if __name__ == "__main__":
    print("in main")
    batch_size = 3
    seq_len = 4
    dim = 2
    seq_emb = np.arange(batch_size * seq_len * dim).reshape((batch_size, seq_len, dim)).astype(np.float32)




