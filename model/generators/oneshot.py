from model.misc.gnn import GNN

from utils.nas_helper import hier_space_params

import torch
import torch.nn as nn
import torch.nn.functional as F

class OneShotGenerator(nn.Module):
  """
  Beginning with a fixed seed graph (eg. a line graph), models a probability
  distribution of the remaining edges to be filled in (eg. the skip connections).
  """

  def __init__(self, config, Q=None):
    super().__init__()
    self.config = config
    self.Q = Q
    self.device = config.generator.device

    self.max_num_nodes = hier_space_params(config)

    self.hidden_dim = config.generator.hidden_dim
    self.num_GNN_prop = config.generator.num_GNN_prop
    self.num_GNN_layers = config.generator.num_GNN_layers
    self.has_attention = config.generator.has_attention

    self.seed_edges = torch.stack([
      torch.arange(self.max_num_nodes - 1),
      torch.arange(1, self.max_num_nodes),
    ], dim=1)
    # pairs of idxs of all the new potential edges we want to add
    # ie., the edges we want to sample from the gnn output
    self.node_idx_gnn = torch.nonzero(torch.triu(
      # diagonal=2 to avoid generating edges already existing in line graph
      torch.ones(self.max_num_nodes, self.max_num_nodes), diagonal=2,
    ))

    self.decoder = GNN(
      msg_dim=self.hidden_dim,
      node_state_dim=self.hidden_dim,
      edge_feat_dim=self.hidden_dim,
      num_prop=self.num_GNN_prop,
      num_layer=self.num_GNN_layers,
      has_attention=self.has_attention
    )

    # we only want 1 output, the logit prob
    self.output_dim = 1
    # logit probability to generate edge
    self.output_theta = nn.Sequential( # edges
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.output_dim))

    # initial node features for each node is 1 hot encoding of node id
    # we may need to project it down to fit in the hidden dim
    # (note this is different from gran where the row of adj matrix is passed in)
    self.dimension_reduce = self.max_num_nodes > self.hidden_dim
    if self.dimension_reduce:
      self.decoder_input = nn.Linear(self.max_num_nodes, self.hidden_dim)
    else:
      padding = self.hidden_dim - self.max_num_nodes
      self.decoder_input = nn.ConstantPad1d(padding=(0, padding), value=0)

    pos_weight = torch.ones([1])
    self.adj_loss_func = nn.BCEWithLogitsLoss(
      pos_weight=pos_weight, reduction='none'
    )


  def _inference(self):
    """
    Run gnn inference to get hidden states

    Currently doesn't allow node feats to be passed in
    """

    N = self.max_num_nodes
    edges = self.seed_edges
    M = edges.shape[0]

    node_feat = torch.zeros(N, N, device=self.device)
    node_feat.diagonal().fill_(1.)
    node_feat = self.decoder_input(node_feat)

    node_id_edge_feat = torch.zeros(M, N, device=self.device)
    if len(edges) > 0:
      node_id_edge_feat.scatter_(1, edges[:, 0].view(-1, 1), -1)
      node_id_edge_feat.scatter_(1, edges[:, 1].view(-1, 1), 1)

    if self.hidden_dim >= N:
      att_edge_feat = torch.zeros(M, self.hidden_dim - N, device=self.device)
    else:
      raise "need to project the edge hidden dim down as well"

    att_edge_feat = torch.cat([att_edge_feat, node_id_edge_feat], dim=1)

    node_state = self.decoder(node_feat, edges, edge_feat=att_edge_feat)

    diff = node_state[self.node_idx_gnn[:, 0], :] \
           - node_state[self.node_idx_gnn[:, 1], :]

    log_theta = self.output_theta(diff)

    return log_theta

  def _sampling(self, B, debug=0):
    with torch.no_grad():
      N = self.max_num_nodes

      log_theta = self._inference()

      theta = log_theta.sigmoid().view(-1)
      logits = torch.zeros(N, N)
      logits[self.node_idx_gnn[:, 0], self.node_idx_gnn[:, 1]] = theta.view(-1)
      print(torch.round(logits * 1000) / 10.)

      A = torch.zeros(B, N, N)
      nl = torch.ones(B, N)
      for b in range(B):
        A[b, self.node_idx_gnn[:, 0], self.node_idx_gnn[:, 1]] = \
            torch.bernoulli(theta)

      A[:, self.seed_edges[:, 0], self.seed_edges[:, 1]] = 1.
      return A, nl

  def forward(self, input_dict):
    """
    Args:
      is_sampling: True if sampling else False when training
    Sampling Args:
      batch_size: number of samples
    Training Args:
      adj: training batch (B, C, N, N)
      rewards: (B,)
      baseline: float
    """
    is_sampling    = input_dict.get('is_sampling', False)
    batch_size     = input_dict.get('batch_size', 0)
    A_pad          = input_dict.get('adj', None)
    rewards        = input_dict.get('rewards', 1)
    baseline       = input_dict.get('baseline', 0)

    if not is_sampling:
      log_theta = self._inference()
      A_pad = A_pad.transpose(2, 3)
      adj_loss = self.bce_loss(log_theta, A_pad, rewards, baseline)
      return adj_loss
    else:
      adj, nl = self._sampling(batch_size)
      return zip(adj, nl)


  def sample(self, batch_size, dataset=None, debug=0, search_space=None, sample_method='random'):
    graphs = []
    num_tries = 0
    max_num_tries = 20
    while len(graphs) < batch_size and num_tries < max_num_tries:
      candidates = self({
        'is_sampling': True,
        'batch_size': batch_size,
      })
      # check validity of candidates here?
      # if needed
      if dataset is not None:
        candidates = [graph for graph in candidates if graph not in dataset]
      graphs.extend(candidates)
      num_tries += 1

    if len(graphs) < batch_size:
      return None
    return graphs[:batch_size]


  def bce_loss(self, log_theta, A_pad, rewards, baseline):
    B = A_pad.shape[0]
    C = A_pad.shape[1]
    assert C == 1, "multiple canon orders not supported"
    N = A_pad.shape[-1]

    label = A_pad.reshape(B, N * N)

    logits = torch.zeros(N, N)
    logits[self.node_idx_gnn[:, 0], self.node_idx_gnn[:, 1]] = log_theta.view(-1)
    logits = logits.view(N * N)

    adj_loss = torch.stack([self.adj_loss_func(logits, label[b]) for b in range(B)], dim=1)
    adj_loss *= rewards - baseline
    adj_loss = adj_loss.mean()
    return adj_loss

