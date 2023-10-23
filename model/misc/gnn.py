import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GNN(nn.Module):

  def __init__(self,
               msg_dim,
               node_state_dim,
               edge_feat_dim,
               num_prop=1,
               num_layer=1,
               has_attention=True,
               att_hidden_dim=128,
               has_residual=False,
               has_graph_output=False,
               output_hidden_dim=128,
               graph_output_dim=None):
    super(GNN, self).__init__()
    self.msg_dim = msg_dim
    self.node_state_dim = node_state_dim
    self.edge_feat_dim = edge_feat_dim
    self.num_prop = num_prop
    self.num_layer = num_layer
    self.has_attention = has_attention
    self.has_residual = has_residual
    self.att_hidden_dim = att_hidden_dim
    self.has_graph_output = has_graph_output
    self.output_hidden_dim = output_hidden_dim
    self.graph_output_dim = graph_output_dim

    self.update_func = nn.ModuleList([
        nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
        for _ in range(self.num_layer)
    ])

    self.msg_func = nn.ModuleList([
        nn.Sequential(
            *[
                nn.Linear(self.node_state_dim + self.edge_feat_dim,
                          self.msg_dim),
                nn.ReLU(),
                nn.Linear(self.msg_dim, self.msg_dim)
            ]) for _ in range(self.num_layer)
    ])

    if self.has_attention:
      self.att_head = nn.ModuleList([
          nn.Sequential(
              *[
                  nn.Linear(self.node_state_dim + self.edge_feat_dim,
                            self.att_hidden_dim),
                  nn.ReLU(),
                  nn.Linear(self.att_hidden_dim, self.msg_dim),
                  nn.Sigmoid()
              ]) for _ in range(self.num_layer)
      ])

    if self.has_graph_output:
      self.graph_output_head_att = nn.Sequential(*[
          nn.Linear(self.node_state_dim, self.output_hidden_dim),
          nn.ReLU(),
          nn.Linear(self.output_hidden_dim, 1),
          nn.Sigmoid()
      ])

      self.graph_output_head = nn.Sequential(
          *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

  def _prop(self, state, edge, edge_feat, layer_idx=0):
    ### compute message
    state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
    if self.edge_feat_dim > 0:
      edge_input = torch.cat([state_diff, edge_feat], dim=1)
    else:
      edge_input = state_diff

    msg = self.msg_func[layer_idx](edge_input)

    ### attention on messages
    if self.has_attention:
      att_weight = self.att_head[layer_idx](edge_input)
      msg = msg * att_weight

    ### aggregate message by sum
    state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
    scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
    state_msg = state_msg.scatter_add(0, scatter_idx, msg)

    ### state update
    state = self.update_func[layer_idx](state_msg, state)
    return state

  def forward(self, node_feat, edge, edge_feat, graph_idx=None):
    """
      N.B.: merge a batch of graphs as a single graph

      node_feat: N X D, node feature
      edge: M X 2, edge indices
      edge_feat: M X D', edge feature
      graph_idx: N X 1, graph indices
    """

    state = node_feat
    prev_state = state
    for ii in range(self.num_layer):
      if ii > 0:
        state = F.relu(state)

      for jj in range(self.num_prop):
        state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

    if self.has_residual:
      state = state + prev_state

    if self.has_graph_output:
      num_graph = graph_idx.max() + 1
      node_att_weight = self.graph_output_head_att(state)
      node_output = self.graph_output_head(state)

      # weighted average
      reduce_output = torch.zeros(num_graph,
                                  node_output.shape[1]).to(node_feat.device)
      reduce_output = reduce_output.scatter_add(0,
                                                graph_idx.unsqueeze(1).expand(
                                                    -1, node_output.shape[1]),
                                                node_output * node_att_weight)

      const = torch.zeros(num_graph).to(node_feat.device)
      const = const.scatter_add(
          0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

      reduce_output = reduce_output / const.view(-1, 1)

      return reduce_output
    else:
      return state


