import numpy as np
import torch

from utils.nas_helper import is_valid

class RandomNB101:

  def __init__(self, config):
    self.config = config
    assert config.generator.search_space == "NB101"

    self.device = self.config.generator.device


  def sample(self, batch_size):
    graphs = []
    while len(graphs) < batch_size:
      adj = np.random.choice(
        [0, 1], size=(7, 7))
      adj = np.triu(adj, 1)
      nl = np.random.choice([1, 2, 3], size=7).tolist()
      nl[0] = 0
      nl[-1] = 4
      adj = torch.tensor(adj, device=self.device)
      nl = torch.tensor(nl, device=self.device)

      if is_valid("NB101", (adj, nl)):
        graphs.append((adj, nl))
    return graphs
