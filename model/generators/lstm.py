
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.nas_helper import is_valid, search_space_params

def to_device(d: dict, device):
  if isinstance(d, dict):
    for k, obj in d.items():
      if torch.is_tensor(obj):
        d[k] = obj.to(device)
  elif isinstance(d, list):
    for i, obj in enumerate(d):
      d[i] = obj.to(device)

class StackedLSTMCell(nn.Module):
  def __init__(self, layers, size, bias):
    super().__init__()
    self.lstm_num_layers = layers
    self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias)
                       for _ in range(self.lstm_num_layers)])

  def forward(self, inputs, hidden):
    prev_h, prev_c = hidden
    next_h, next_c = [], []
    for i, m in enumerate(self.lstm_modules):
      curr_h, curr_c = m(inputs, (prev_h[i], prev_c[i]))
      next_c.append(curr_c)
      next_h.append(curr_h)
      # current implementation only supports batch size equals 1,
      # but the algorithm does not necessarily have this limitation
      inputs = curr_h
    return next_h, next_c

class ReinforceField:
  """
  A field with ``name``, with ``total`` choices. ``choose_one`` is true if one and only one is meant to be
  selected. Otherwise, any number of choices can be chosen.
  """

  def __init__(self, name, total, choose_one):
    self.name = name
    self.total = total
    self.choose_one = choose_one

  def __repr__(self):
    return f'ReinforceField(name={self.name}, total={self.total}, choose_one={self.choose_one})'


class LstmRL(nn.Module):
  """
  A controller that mutates the graph with RL.

  Parameters
  ----------
  fields : list of ReinforceField
    List of fields to choose.
  lstm_size : int
    Controller LSTM hidden units.
  lstm_num_layers : int
    Number of layers for stacked LSTM.
  tanh_constant : float
    Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
  skip_target : float
    Target probability that skipconnect will appear.
  temperature : float
    Temperature constant that divides the logits.
  entropy_reduction : str
    Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
  """

  def __init__(self, fields, device, lstm_size=128, lstm_num_layers=2, tanh_constant=2.5,
         skip_target=0.4, temperature=5):
    super(LstmRL, self).__init__()
    self.fields = fields
    self.device = device
    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.skip_target = skip_target
    # HACK for gran controller
    self.max_num_nodes = len(fields) + 1
    self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)
    self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
    self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
    self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1)
    self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]),  # pylint: disable=not-callable
                     requires_grad=False)
    self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    self.soft = nn.ModuleDict({
      field.name: nn.Linear(self.lstm_size, field.total, bias=False) for field in fields
    })
    self.embedding = nn.ModuleDict({
      field.name: nn.Embedding(field.total, self.lstm_size) for field in fields
    })

  def resample(self):
    self._initialize()
    result = dict()
    for field in self.fields:
      result[field.name] = self._sample_single(field)
    return result

  def _initialize(self, batch_size=1):
    self._inputs = self.g_emb.data.expand(batch_size, -1)
    self._c = [torch.zeros(
                (batch_size, self.lstm_size),
                 dtype=self._inputs.dtype,
                 device=self._inputs.device
              ) for _ in range(self.lstm_num_layers)]
    self._h = [torch.zeros(
                (batch_size, self.lstm_size),
                dtype=self._inputs.dtype,
                device=self._inputs.device
              ) for _ in range(self.lstm_num_layers)]
    self.sample_log_prob = 0
    self.sample_entropy = 0
    self.sample_skip_penalty = 0
    self.inference_log_prob = []

  def _lstm_next_step(self):
    self._h, self._c = self.lstm(self._inputs, (self._h, self._c))

  # TODO
  def _inference_batch(self, field, decision):
    B = decision.shape[0]
    self._lstm_next_step()
    logit = self.soft[field.name](self._h[-1])
    if self.temperature is not None:
      logit /= self.temperature
    if self.tanh_constant is not None:
      logit = self.tanh_constant * torch.tanh(logit)

    if field.choose_one:
      logit = F.log_softmax(logit, dim=-1)
      log_prob = self.cross_entropy_loss(logit, decision)
      self._inputs = self.embedding[field.name](decision)
    else:
      # assume that multi-choose field is the default one
      logit = logit.sigmoid()
      log_prob = self.cross_entropy_loss(logit, decision)

      next_inputs = []
      for i in range(B):
        di = decision[i].nonzero().view(-1)
        if di.shape[0] != 0:
          di = (torch.sum(self.embedding[field.name](di), 0) / (1. + torch.sum(di))).unsqueeze(0)
        else:
          di = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)
        next_inputs.append(di)
      self._inputs = torch.cat(next_inputs, 0)
    self.inference_log_prob.append(log_prob)

  # TODO to be debugged
  def _sample_single(self, field):
    with torch.no_grad():
      self._lstm_next_step()
    logit = self.soft[field.name](self._h[-1])
    if self.temperature is not None:
      logit /= self.temperature
    if self.tanh_constant is not None:
      logit = self.tanh_constant * torch.tanh(logit)

    if field.choose_one:
      sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
      self._inputs = self.embedding[field.name](sampled)
    else:
      logit = logit.view(-1, 1)
      logit = torch.cat([-logit, logit], 1)  # pylint: disable=invalid-unary-operand-type
      sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
      sampled = sampled.nonzero().view(-1)
      if sampled.sum().item():
        self._inputs = (torch.sum(self.embedding[field.name](sampled.view(-1)), 0) / (1. + torch.sum(sampled))).unsqueeze(0)
      else:
        self._inputs = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)

    sampled = sampled.detach().cpu().numpy().tolist()
    if len(sampled) == 1 and field.choose_one:
      sampled = sampled[0]
    return sampled

  # TODO to be debugged
  # def _sample_single(self, field):
  #     with torch.no_grad():
  #         self._lstm_next_step()
  #     logit = self.soft[field.name](self._h[-1])
  #     if self.temperature is not None:
  #         logit /= self.temperature
  #     if self.tanh_constant is not None:
  #         logit = self.tanh_constant * torch.tanh(logit)

  #     logit = logit.view(-1, 1)
  #     logit = torch.cat([-logit, logit], 1)  # pylint: disable=invalid-unary-operand-type
  #     sampled = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
  #     sampled = sampled.nonzero().view(-1)
  #     if sampled.sum().item():
  #         self._inputs = (torch.sum(self.embedding[field.name](sampled.view(-1)), 0) / (1. + torch.sum(sampled))).unsqueeze(0)
  #     else:
  #         self._inputs = torch.zeros(1, self.lstm_size, device=self.embedding[field.name].weight.device)

  #     sampled = sampled.detach().cpu().numpy().tolist()
  #     return sampleda

  def adj_to_decisions(self, batch_adj, batch_node_label):
    decisions = []
    n = batch_adj.shape[-1]
    for i in range(1, n):
      decisions.append(batch_node_label[:, i])
      decisions.append(batch_adj[:, i, :i])
    return decisions

  # TODO check
  def decisions_to_adj(self, decisions):
    n = len(self.fields) // 2 + 1
    adj = torch.zeros((n, n))
    node_label = torch.zeros(n)
    for i in range(1, n):
      node_label[i] = decisions[2 * i - 2]
      decision = decisions[2 * i - 1]
      for choose in decision:
        assert choose < i
        adj[i, choose] = 1
    return adj, node_label

  def sample(self, batch_size, dataset=None):
    num_tries = 0
    graphs = []

    while num_tries < 100:
      candidates = []
      for i in range(batch_size):
        self._initialize()
        decisions = []
        for field in self.fields:
          decisions.append(self._sample_single(field))
          # TODO fit samples to an adj matrix
        adj, nl = self.decisions_to_adj(decisions)
        end_node = torch.nonzero(nl == 4)
        num_node = end_node.min() + 1 if end_node.nelement() > 0 else 0
        candidates.append((adj.transpose(0, 1)[:num_node, :num_node], nl[:num_node]))

      candidates = [
          (adj, nl) for adj, nl in candidates
          if is_valid("NB101", (adj, nl))
      ]

      if dataset is not None:
        candidates = [graph for graph in candidates if graph not in dataset]
      graphs.extend(candidates)
      num_tries += 1
      if len(graphs) >= batch_size:
        break
    return graphs[:batch_size]

  def forward(self, input_dict, dataset=None):
    """
      input share be batch of adjs
    """
    to_device(input_dict, self.device)
    is_sampling = input_dict[
        'is_sampling'] if 'is_sampling' in input_dict else False
    batch_size = input_dict[
        'batch_size'] if 'batch_size' in input_dict else None
    A_pad = input_dict['adj'] if 'adj' in input_dict else None
    node_label = input_dict['node_label'] if 'node_label' in input_dict else None

    if is_sampling:
      return self.sample(batch_size, dataset)
    else:
      self._initialize(A_pad.shape[0])
      A_pad = A_pad[:, 0, ...]
      node_label = node_label[:, 0, ...]
      # print("A_pad.shape", A_pad.shape)
      # print("A_pad", A_pad)
      # print("node_label.shape", node_label.shape)
      # print("node_label", node_label)
      if len(A_pad.shape) == 4:
        A_pad = A_pad[:, 0, :, :]
      decisions = self.adj_to_decisions(A_pad, node_label)
      # print("decisions", decisions)
      for field, decision in zip(self.fields, decisions):
        # print(field, decision)
        self._inference_batch(field, decision)

      loss = 0
      cnt = 0
      for log_prob_row in self.inference_log_prob:
        loss = loss + log_prob_row.sum(axis=-1)
        cnt = cnt + log_prob_row.shape[-1]
      loss = loss / cnt
      # TODO hack here
      return loss.mean()

def lstm_nb101(config, Q=None):
  n_layers = 7
  n_label = 5
  rl_fields = []
  for i in range(1, n_layers):
    rl_fields.append(
      ReinforceField(f'node_{i}', n_label, choose_one=True)
    )
    rl_fields.append(
      ReinforceField(f'row_{i}', i, choose_one=False)
    )
  return LstmRL(rl_fields, 'cuda')

  self.controller = LstmRL(rl_fields, self.device).to(self.device)
  self.ctrl_optim = torch.optim.Adam(self.controller.parameters(), lr=config.solver.search.ctrl_lr)

  if config.resume:
    if os.path.exists(config.ckpt_path):
      self.Q.info(f"Auto resume from {config.ckpt_path}")
      self.resume(config.ckpt_path)
    else:
      self.Q.info(f"No checkpoint found {config.ckpt_path}")
      if config.search_engine.rl.pretrain:
        self.pretrain()
  elif config.search_engine.rl.pretrain:
    self.pretrain()

