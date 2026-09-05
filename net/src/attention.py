import torch
import torch.nn.functional as F


def attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q@K.transpose(-2 , -1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
    weights = F.softmax(scores , dim = -1)
    output = weights @ V
    
    return output