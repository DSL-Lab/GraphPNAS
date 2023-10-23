import random

import torch

from utils.nas_helper import is_valid, search_space_params

class ErdosRenyi:
  # erdos renyi with rejection sampling

  def __init__(self, config):
    self.config = config
    self.max_num_nodes, self.num_node_labels, self.num_edge_labels = \
        search_space_params(config)
    self.erdos_renyi_p = config.generator.erdos_renyi_p
    self.search_space = config.generator.search_space

  def _smalldarts_random(self):
    adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
    nl = torch.ones(self.max_num_nodes)
    for dst in range(2, 6):
      srcs = random.sample(range(dst), 2)
      adj[srcs[0], dst] = random.randint(1, self.num_edge_labels - 1)
      adj[srcs[1], dst] = random.randint(1, self.num_edge_labels - 1)
    return adj, nl

  def _nb201_random(self):
    adj = torch.zeros(self.max_num_nodes, self.max_num_nodes)
    nl = torch.zeros(self.max_num_nodes)
    adj[0][1] = random.randint(1, self.num_edge_labels - 1)
    adj[0][2] = random.randint(1, self.num_edge_labels - 1)
    adj[1][2] = random.randint(1, self.num_edge_labels - 1)
    adj[0][3] = random.randint(1, self.num_edge_labels - 1)
    adj[1][3] = random.randint(1, self.num_edge_labels - 1)
    adj[2][3] = random.randint(1, self.num_edge_labels - 1)
    return adj, nl

  def sample(self, batch_size):
    graphs = []

    N = random.randint(3, self.max_num_nodes)
    while len(graphs) < batch_size:
      # provide optimized version of random generator for the more restrictive
      # search spaces, such as darts
      if self.search_space == 'smalldarts':
        graph = self._smalldarts_random()
      elif self.search_space == 'darts':
        graph = self._smalldarts_random(), self._smalldarts_random()
      elif self.search_space == 'NB201':
        graph = self._nb201_random()
      else:
        # default random generator (may be very slow)
        adj = torch.zeros(N, N)
        for src in range(N):
          for dst in range(src+1, N):
            adj[src, dst] = random.random() < self.erdos_renyi_p
            if self.num_edge_labels > 1:
              adj[src, dst] *= random.randint(1, self.num_edge_labels - 1)

        nl = torch.zeros(N)
        if self.num_node_labels > 1:
          nl[0] = 0
          nl[-1] = self.num_node_labels - 1
          for i in range(1, N-1):
            nl[i] = random.randint(1, self.num_node_labels - 1)
        graph = adj, nl

      if is_valid(self.search_space, graph):
        graphs.append(graph)
        N = random.randint(3, self.max_num_nodes)

    return graphs

class SkipExplorer:

  def __init__(self, config):
    self.config = config
    self.max_num_nodes, self.num_node_labels, self.num_edge_labels = \
        search_space_params(config)
    self.erdos_renyi_p = config.generator.erdos_renyi_p
    self.search_space = config.generator.search_space

  def sample(self, batch_size):
    graphs = []
    N = self.max_num_nodes
    for _ in range(batch_size):
      A = torch.zeros(N, N)

      for src in range(N - 1):
        A[src, src + 1] = 1.
        for dst in range(src + 2, N):
          A[src, dst] = random.random() < self.erdos_renyi_p
      nl = torch.zeros(N)

      graphs.append((A, nl))

    return graphs


