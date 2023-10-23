import random
# import torch

class Uniform:

  def __init__(self, config, Q=None):
    pass

  def estimate(self, graphs):
    # return [2 * adj.sum() / torch.numel(adj) for adj, nl in graphs], [0] * len(graphs)
    return [random.random()*0.2+0.8 for _ in graphs], [0] * len(graphs)

