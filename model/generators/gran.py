import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.misc.gnn import GNN
from utils.nas_helper import is_valid, search_space_params
from utils.torch_helper import to_base_b

EPS = np.finfo(np.float32).eps

__all__ = ['GRAN']

class GRAN(nn.Module):
  """ Graph Recurrent Attention Networks """

  def __init__(self, config, Q=None):
    super(GRAN, self).__init__()
    self.config = config
    self.Q = Q
    self.device = config.generator.device
    self.search_space = config.generator.search_space
    self.max_num_nodes, self.num_node_labels, self.num_edge_labels = \
        search_space_params(config)
    self.hidden_dim = config.generator.hidden_dim
    self.is_sym = False # config.generator.is_sym # (we want DAGs)
    self.block_size = config.generator.block_size
    self.sample_stride = config.generator.sample_stride
    self.num_GNN_prop = config.generator.num_GNN_prop
    self.num_GNN_layers = config.generator.num_GNN_layers
    self.edge_weight = config.generator.edge_weight if hasattr(
        config.generator, 'edge_weight') else 1.0
    self.dimension_reduce = config.generator.dimension_reduce
    self.has_attention = config.generator.has_attention
    self.num_canonical_order = config.generator.num_canonical_order
    self.output_dim = 1
    self.num_mix_component = config.generator.num_mix_component
    self.has_rand_feat = False # use random feature instead of 1-of-K encoding
    self.att_edge_dim = self.block_size

    # these heads output logit(prob)
    self.output_theta = nn.Sequential( # edges
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component))

    self.output_alpha = nn.Sequential( # mixture
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.num_mix_component))

    self.output_phi = nn.Sequential( # node label
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.num_node_labels),
    )

    self.output_psi = nn.Sequential( # edge label
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.num_edge_labels),
    )

    if self.dimension_reduce:
      self.embedding_dim = config.generator.hidden_dim - self.max_num_nodes
      self.decoder_input = nn.Sequential(
          nn.Linear(self.max_num_nodes, self.embedding_dim))
    else:
      self.embedding_dim = self.max_num_nodes

    self.decoder = GNN(
        msg_dim=self.hidden_dim,
        node_state_dim=self.hidden_dim,
        edge_feat_dim=self.hidden_dim,
        num_prop=self.num_GNN_prop,
        num_layer=self.num_GNN_layers,
        has_attention=self.has_attention)

    ### Loss functions
    pos_weight = torch.ones([1]) * self.edge_weight
    self.adj_loss_func = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction='none')


  def _inference(self,
                 A_pad=None,
                 edges=None,
                 node_ids=None,
                 node_idx_gnn=None,
                 node_idx_feat=None,
                 node_feat=None,
                 edge_feat=None,
                 att_idx=None,
                 debug=0):
    """
    forward pass generate adj in row-wise auto-regressive fashion.
    given input features, compute logit probabilities

    node_idx_gnn: M X 2, node indices of augmented edges
    node_idx_feat: N X 1, node indices of subgraphs for indexing from feature
                  (0 indicates indexing from 0-th row of feature which is
                    always zero and corresponds to newly generated nodes)
    att_idx: N X 1, one-hot encoding of newly generated nodes
                  (0 indicates existing nodes, 1-D indicates new nodes in
                    the to-be-generated block)
    subgraph_idx: E X 1, indices corresponding to augmented edges
                  (representing which subgraph in mini-batch the augmented
                  edge belongs to)
    node_feat: initial node features
    A_pad: padded adjacency matrix only used to calculate node features
    """

    H = self.hidden_dim
    K = self.block_size

    if A_pad is None:
      assert node_feat is not None
    if node_feat is None:
      assert A_pad is not None
      B, C, N_max, _ = A_pad.shape
      A_pad = A_pad.to(self.device)
      A_pad = (A_pad > 0).float()
      A_pad = A_pad.view(B * C * N_max, -1)
      # we allow this to be passed in as well, to cache
      node_feat = torch.zeros(B * C * N_max, self.hidden_dim, device=self.device)
      if self.dimension_reduce:
        node_feat[:, :self.embedding_dim] = self.decoder_input(A_pad)  # BCN_max X H
      else:
        node_feat[:, :self.embedding_dim] = A_pad  # BCN_max X N_max

      node_feat = node_feat.reshape(B * C, N_max, self.hidden_dim)
      # node id as feature
      node_feat[:, range(0, N_max), range(self.embedding_dim, self.embedding_dim + N_max)] = 1
      node_feat = node_feat.reshape(B * C * N_max, self.hidden_dim)

      ### GNN inference
      # pad zero as node feature for newly generated nodes (1st row)
      node_feat = F.pad(
          node_feat, (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) X N_max
    if debug:
      print("idx: ", debug-1)
      print("A_pad", A_pad)
      print("edges.T", edges.transpose(0, 1))
      print("node_ids", node_ids)
      print("node_idx_gnn.T", node_idx_gnn.transpose(0, 1))
      print("node_idx_feat", node_idx_feat)
      print("node_feat", node_feat)
      print("edge_feat", edge_feat)
      print("att_idx", att_idx)
      print("#"*40)

    # create symmetry-breaking edge feature for the newly generated nodes
    att_idx = att_idx.view(-1, 1)

    # att_edge_dim is the largest even number less than combined_att_edge_dim
    combined_att_edge_dim = self.hidden_dim - self.num_edge_labels - self.max_num_nodes
    assert combined_att_edge_dim > 0
    if self.has_rand_feat:
      # create random feature
      att_edge_feat = torch.zeros(edges.shape[0],
                                  combined_att_edge_dim).to(node_feat.device)
      idx_new_node = (att_idx[[edges[:, 0]]] >
                      0).long() + (att_idx[[edges[:, 1]]] > 0).long()
      idx_new_node = idx_new_node.byte().squeeze()
      att_edge_feat[idx_new_node, :] = torch.randn(
          idx_new_node.long().sum(),
          att_edge_feat.shape[1]).to(node_feat.device)
    else:
      # create one-hot feature
      att_edge_feat = torch.zeros(edges.shape[0],
                                  combined_att_edge_dim).to(node_feat.device)
      # scatter with empty index seems to cause problem on CPU but not on GPU
      if len(edges) > 0:
        att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
        att_edge_feat = att_edge_feat.scatter(
            1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

    if edge_feat is None:
      edge_feat = torch.empty(edges.shape[0], 0).to(node_feat.device)
    else:
      assert self.num_edge_labels > 0

    # node id as feature
    node_id_edge_feat = torch.zeros(edges.shape[0], self.max_num_nodes, device=self.device)
    if len(edges) > 0:
      node_id_edge_feat.scatter_(1, node_ids[edges[:, 0]].view(-1, 1), -1)
      node_id_edge_feat.scatter_(1, node_ids[edges[:, 1]].view(-1, 1), 1)

    att_edge_feat = torch.cat([att_edge_feat, edge_feat, node_id_edge_feat], dim=1)

    # GNN inference
    # N.B.: node_feat is shared by multiple subgraphs within the same batch
    node_state = self.decoder(
        node_feat[node_idx_feat], edges, edge_feat=att_edge_feat)

    ### Pairwise predict edges
    diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]

    log_theta = self.output_theta(diff)  # B X (tt+K)K
    log_theta = log_theta.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
    log_alpha = self.output_alpha(diff)  # B X (tt+K)K
    log_alpha = log_alpha.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K

    new_node_idxs = torch.nonzero(att_idx.view(-1)).view(-1)
    log_phi = self.output_phi(node_state[new_node_idxs, :]) if self.num_node_labels > 0 else None
    log_psi = self.output_psi(diff) if self.num_edge_labels > 0 else None

    return log_theta, log_alpha, log_phi, log_psi

  def _sampling(self, B, debug=0, temperature=1, sample_method="random"):
    """ generate adj in row-wise auto-regressive fashion """

    beam_search = sample_method == "beam" or sample_method == "beam_return_prob"
    with torch.no_grad():

        K = self.block_size
        S = self.sample_stride
        H = self.hidden_dim
        N = self.max_num_nodes

        # increase size of A so we don't exceed array dimensions
        # (only applicable if we need a partial block size for the last generation
        # stride)
        mod_val = (N - K) % S
        if mod_val > 0:
          N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
        else:
          N_pad = N

        A = torch.zeros(B, N_pad, N_pad).to(self.device)
        if beam_search:
          A_loglikelihood = torch.full((B,), float('-inf'))
          A_loglikelihood[0] = 0
          candidates = torch.zeros(B, N_pad, N_pad).to(self.device)
        if self.num_node_labels > 0:
          nl = torch.zeros(B, N_pad, dtype=torch.long, device=self.device)
        else:
          # useful so we don't have to keep checking null
          nl = torch.zeros(B, N_pad, device=self.device)
        # L = labeled adjacency matrix
        if self.num_edge_labels > 0:
          L = torch.zeros(B, N_pad, N_pad, device=self.device)
        else:
          L = None
        dim_input = self.hidden_dim

        # initial node state
        # cache node state for speed up
        node_state = torch.zeros(B, N_pad, dim_input).to(self.device)

        for ii in range(0, N_pad, S):
          # ii is beginning of current block
          # jj is end of current block
          jj = ii + K
          if jj > N_pad:
            break

          # reset to discard overlap generation
          A[:, ii:, :] = .0
          A = torch.tril(A, diagonal=-1)
          if self.num_edge_labels > 0:
            L[:, ii:, :] = 0.
            L = torch.tril(L, diagonal=-1)

          i_start = ii - K if ii >= K else 0
          if self.dimension_reduce:
            node_state[:, i_start:ii, :self.embedding_dim] = self.decoder_input((A[:, i_start:ii, :N] > 0).float())
          else:
            node_state[:, i_start:ii, :self.embedding_dim] = A[:, i_start:ii, :N]

          node_state[:, range(i_start, ii), range(self.embedding_dim + i_start, self.embedding_dim + ii)] = 1

          node_feat = F.pad(
              node_state[:, :ii, :], (0, 0, 0, K), 'constant', value=.0)
          node_feat = node_feat.view(B * (ii + K), dim_input)
          node_idx_feat = torch.arange(B * (ii + K))

          ### GNN propagation
          adj = F.pad(
              A[:, :ii, :ii], (0, K, 0, K), 'constant', value=1.0)  # B X jj X jj
          adj = torch.tril(adj, diagonal=-1)
          adj = adj + adj.transpose(1, 2)
          edges = [
              adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
              for bb in range(B)
          ]
          edges = torch.cat(edges, dim=1).t()

          att_idx = torch.cat([torch.zeros(ii).long(),
                               torch.arange(1, K + 1)]).to(self.device)
          att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

          if self.num_edge_labels > 0:
            edge_feat = torch.zeros((edges.shape[0], self.num_edge_labels)).to(self.device)
            if edges.shape[0] > 0:
              # recover the edge labels generated so far
              batch_idx = edges[:, 0] // adj.shape[1] # recover bb from above
              src_idx = edges[:, 0] % adj.shape[1]
              dst_idx = edges[:, 1] % adj.shape[1]
              edge_label = (L + L.transpose(1, 2))[batch_idx, src_idx, dst_idx].long()
              edge_feat[range(edges.shape[0]), edge_label] = 1
          else:
            edge_feat = None

          idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
          idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
          idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)
          node_idx_gnn = torch.stack([idx_row, idx_col], dim=1)
          node_idx_gnn = torch.cat([node_idx_gnn + bb * adj.shape[1] for bb in range(B)])

          node_ids = torch.arange(adj.shape[1]).repeat(B)

          cols = [
            torch.tensor([5, 5]),
            torch.tensor([1, 5, 0]),
            torch.tensor([4, 4, 0, 0]),
            torch.tensor([0, 5, 0, 0, 4]),
          ]
          debug = False
          for xd in range(B):
            aa = A[xd]
            # if ii == 2:
              # debug = xd + 1
            # if ii == 3 and (aa[2, :2] == cols[0]).all():
              # debug = xd + 1
            # if ii == 4 and (aa[2, :2] == cols[0]).all()\
                # and (aa[3, :3] == cols[1]).all():
              # debug = xd + 1
            # if ii == 5 and (aa[2, :2] == cols[0]).all()\
                # and (aa[3, :3] == cols[1]).all()\
                # and (aa[4, :4] == cols[2]).all():
              # debug = xd + 1
          log_theta, log_alpha, log_phi, log_psi = self._inference(
             edges=edges.to(self.device),
             node_ids=node_ids.to(self.device),
             node_idx_gnn=node_idx_gnn.to(self.device),
             node_idx_feat=node_idx_feat.to(self.device),
             node_feat=node_feat.to(self.device),
             edge_feat=edge_feat.to(self.device),
             att_idx=att_idx.to(self.device),
             debug=debug,
          )
          debug = False # REMOVE
          log_theta /= temperature
          log_alpha /= temperature
          if log_phi is not None:
            log_phi /= temperature
          if log_psi is not None:
            log_psi /= temperature

          if beam_search:
            candidates *= 0
            cand_idxs = [[i, float("-inf")] for i in range(B)] # pairs (idx, log_prob)
            if ii == 1 and "darts" in self.search_space:
              # in darts we don't generate the first row
              continue
            log_theta = log_theta.view(B, -1, self.num_mix_component)
            log_theta = F.logsigmoid(log_theta)
            log_alpha = log_alpha.view(B, -1, self.num_mix_component)
            log_alpha = log_alpha.mean(dim=1)
            log_alpha = torch.log_softmax(log_alpha, dim=1)
            log_alpha = log_alpha.view(B, 1, self.num_mix_component)
            edge_log_prob = (log_theta + log_alpha).logsumexp(dim=-1)
            edge_log_prob = edge_log_prob.view(B, ii+K, K)
            edge_log_prob = edge_log_prob.transpose(1, 2)
            # import pdb; pdb.set_trace()
            if self.num_edge_labels > 0:
              log_psi = log_psi.view(B, -1, self.num_edge_labels)
              log_psi = torch.log_softmax(log_psi, dim=-1)
            # compute num choices
            # num choices for the edges = (ii) + (ii+1) + ... + (jj-1)
            num_edges = (ii + jj - 1) * (jj - ii) // 2
            if num_edges < 0:
              num_edges = 0
              cand_idxs[0][1] = 0
            choices_per_edge = max(self.num_edge_labels, 2)

            for b, (adj, p) in enumerate(zip(A, A_loglikelihood)):
              if p == float("-inf"): continue
              for mask in range(choices_per_edge ** num_edges):
                bits = to_base_b(mask, choices_per_edge, num_edges).long()
                edges = bits > 0
                if "darts" in self.search_space:
                  if ii >= 2 and (bits > 0).sum() != 2:
                    # in darts we want in degree = 2
                    continue
                # if len(bits) >= 2:
                  # import pdb; pdb.set_trace()
                edges = edges.float()
                log_prob = p.item()
                lo = 0
                # mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           # subgraph_idx, subgraph_idx_base, num_canonical_order,
                           # node_label=None, log_phi=None, edge_label=None, log_psi=None,
                           # sum_order_log_prob=False, return_neg_log_prob=False, reduction="mean",
                           # rewards=None, baseline=0,
                           # debug=0):
                for i in range(ii, jj):
                  if i == 0: continue
                  delta = ii - i
                  log_prob += torch.dot(edge_log_prob[b, delta, :i], edges[lo:lo+i])
                  log_prob += torch.dot(
                    torch.log(1 - edge_log_prob[b, delta, :i].exp() + 1e-6), 1-edges[lo:lo+i]
                  )
                  if self.num_edge_labels > 0:
                    edge_lab_ps = log_psi[b, range(i), bits[lo:lo+i]]
                    log_prob += torch.dot(edge_lab_ps, edges[lo:lo+i])
                  should_print = False
                  cols = [
                    torch.tensor([5, 5]),
                    torch.tensor([1, 5, 0]),
                    torch.tensor([4, 4, 0, 0]),
                    torch.tensor([0, 5, 0, 0, 4]),
                  ]
                  if len(bits) == 2 and (bits == cols[0]).all():
                    print(adj)
                    should_print = True
                  if len(bits) == 3 and (bits == cols[1]).all()\
                      and (adj[2, :2] == cols[0]).all():
                    should_print = True
                  if len(bits) == 4 and (bits == cols[2]).all()\
                      and (adj[2, :2] == cols[0]).all()\
                      and (adj[3, :3] == cols[1]).all():
                    should_print = True
                  if len(bits) == 5 and (bits == cols[3]).all()\
                      and (adj[2, :2] == cols[0]).all()\
                      and (adj[3, :3] == cols[1]).all()\
                      and (adj[4, :4] == cols[2]).all():
                    should_print = True
                  if False and should_print:
                    print(bits)
                    print("\t", edge_log_prob[b, delta, :i])
                    tmp = torch.dot(edge_log_prob[b, delta, :i], edges[lo:lo+i])
                    tmp += torch.dot(
                      torch.log(1 - edge_log_prob[b, delta, :i].exp() + 1e-6), 1-edges[lo:lo+i]
                    )
                    print("\t", tmp)
                    print("\t", log_psi[b, range(i), bits[lo:lo+i]] * edges[lo:lo+i],)
                    # torch.dot(edge_lab_ps, edges[lo:lo+i]))
                  lo += i

                # print("lp", log_prob, bits)
                if log_prob > cand_idxs[-1][1]:
                  cand_idxs[-1][1] = log_prob
                  new_idx = cand_idxs[-1][0]
                  # sort by decreasing likelihood
                  cand_idxs.sort(key=lambda x:-x[1])
                  candidates[new_idx, :ii, :ii] = adj[:ii, :ii]
                  lo = 0
                  for i in range(ii, jj):
                    candidates[new_idx, i, :i] = edges[lo:lo+i]
                    if self.num_edge_labels > 0:
                      candidates[new_idx, i, :i] *= bits[lo:lo+i]
                    lo += i
              # print("#"*14)
              # print(cand_idxs)

            A[:, :, :] = candidates
            for i in range(B):
              A_loglikelihood[cand_idxs[i][0]] = cand_idxs[i][1]
            # print("*"*20)
            # print(ii, "end", A, A_loglikelihood)
            print(A_loglikelihood)
            L = A
          else: # random sampling
            log_theta = log_theta.view(B, -1, K, self.num_mix_component)  # B X (ii+K) X K X L
            log_theta = log_theta.transpose(1, 2)  # B X K X (ii+K) X L

            log_alpha = log_alpha.view(B, -1, self.num_mix_component)  # B X (K * (ii+K)) X L
            prob_alpha = F.softmax(log_alpha.mean(dim=1), -1)
            alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()

            edge_prob = []
            for bb in range(B):
              theta = torch.sigmoid(log_theta[bb, :, :, alpha[bb]])
              edge_prob += [theta]

            edge_prob = torch.stack(edge_prob, dim=0)

            generated_edges = torch.bernoulli(edge_prob[:, :jj - ii, :])
            A[:, ii:jj, :jj] = generated_edges

            if self.num_node_labels > 0:
              phi = torch.softmax(log_phi, dim=1)
              phi = phi.view(B * K, self.num_node_labels)
              new_labels = torch.multinomial(phi, num_samples=1).view(B * K)
              new_node_idx_0 = torch.arange(B).repeat(K)
              new_node_idx_1 = torch.arange(ii, ii + K).repeat(B)
              nl[new_node_idx_0, new_node_idx_1] = new_labels

            if self.num_edge_labels > 0:
              psi = torch.softmax(log_psi, dim=1)
              # only works if stride/block size is 1
              labels = torch.multinomial(psi, num_samples=1).view(B, 1, ii+K)
              # generated_edges = generated_edges.view(B, ii + K)
              L[:, ii:jj, :jj] = generated_edges * labels.float()

        if self.num_edge_labels > 0:
          A = L

        ### make it symmetric
        if self.is_sym:
          A = torch.tril(A, diagonal=-1)
          A = A + A.transpose(1, 2)

        if sample_method == "beam_return_prob":
          return A, nl, A_loglikelihood

        return A, nl

  def forward(self, input_dict):
    """
      B: batch size
      N: number of rows/columns in mini-batch
      N_max: number of max number of rows/columns
      M: number of augmented edges in mini-batch
      H: input dimension of GNN
      K: block size
      E: number of edges in mini-batch
      S: stride
      C: number of canonical orderings
      D: number of mixture Bernoulli

      Args:
        A_pad: B X C X N_max X N_max, padded adjacency matrix
        node_label: B X C X N_max, padded node labels
        node_idx_gnn: M X 2, node indices of augmented edges
        node_idx_feat: N X 1, node indices of subgraphs for indexing from feature
                      (0 indicates indexing from 0-th row of feature which is
                        always zero and corresponds to newly generated nodes)
        att_idx: N X 1, one-hot encoding of newly generated nodes
                      (0 indicates existing nodes, 1-D indicates new nodes in
                        the to-be-generated block)
        subgraph_idx: E X 1, indices corresponding to augmented edges
                      (representing which subgraph in mini-batch the augmented
                      edge belongs to)
        edges: E X 2, edge as [incoming node index, outgoing node index]
        label: E X 1, binary label of augmented edges
        num_nodes_pmf: N_max, empirical probability mass function of number of nodes

      Returns:
        loss                        if training
        list of adjacency matrices  else
    """
    is_sampling    = input_dict.get('is_sampling', False)
    batch_size     = input_dict.get('batch_size', None)
    sample_method  = input_dict.get('sample_method', 'random')
    temperature    = input_dict.get('temperature', 1)
    A_pad          = input_dict.get('adj', None)
    node_idx_gnn   = input_dict.get('node_idx_gnn', None)
    node_idx_feat  = input_dict.get('node_idx_feat', None)
    node_ids       = input_dict.get('node_ids', None)
    att_idx        = input_dict.get('att_idx', None)
    subgraph_idx   = input_dict.get('subgraph_idx', None)
    edges          = input_dict.get('edges', None)
    label          = input_dict.get('label', None)
    node_label     = input_dict.get('node_label', None)
    edge_label     = input_dict.get('edge_label', None)
    edge_feat      = input_dict.get('edge_feat', None)
    num_nodes_pmf  = input_dict.get('num_nodes_pmf', None)
    subgraph_idx_base = input_dict.get('subgraph_idx_base', None)
    rewards        = input_dict.get('rewards', None)
    baseline       = input_dict.get('baseline', 0)
    debug          = input_dict.get('debug', 0)
    return_neg_log_prob = input_dict.get('return_neg_log_prob', False)

    if not self.config.nas.baseline:
      baseline = 0

    N_max = self.max_num_nodes

    if not is_sampling:
      edge_feat = edge_feat.to(self.device) if edge_feat is not None else None
      att_idx = att_idx.to(self.device)
      node_ids = node_ids.to(self.device)
      node_idx_gnn = node_idx_gnn.to(self.device)
      node_idx_feat = node_idx_feat.to(self.device)
      edges = edges.to(self.device)
      subgraph_idx = subgraph_idx.to(self.device)
      subgraph_idx_base = subgraph_idx_base.to(self.device)
      label = label.to(self.device)
      node_label = node_label.to(self.device) if node_label is not None else None
      edge_label = edge_label.to(self.device) if edge_label is not None else None
      rewards = rewards.to(self.device) if rewards is not None else None


      B, _, N, _ = A_pad.shape

      ### compute adj loss
      log_theta, log_alpha, log_phi, log_psi = self._inference(
          A_pad=A_pad,
          edges=edges,
          node_idx_gnn=node_idx_gnn,
          node_idx_feat=node_idx_feat,
          node_ids=node_ids,
          edge_feat=edge_feat,
          att_idx=att_idx,
          debug=debug
      )

      num_edges = log_theta.shape[0]

      adj_loss = mixture_bernoulli_loss(label, log_theta, log_alpha,
                                        self.adj_loss_func, subgraph_idx, subgraph_idx_base,
                                        self.num_canonical_order,
                                        node_label=node_label, log_phi=log_phi,
                                        edge_label=edge_label, log_psi=log_psi,
                                        rewards=rewards, baseline=baseline,
                                        debug=debug, return_neg_log_prob=return_neg_log_prob)

      return adj_loss
    else:
      graphs = self._sampling(batch_size, debug=debug, temperature=temperature,
                             sample_method=sample_method)

      if 'return_prob' in sample_method:
        A, nl, prob = graphs
      else:
        A, nl = graphs
        prob = None

      ### sample number of nodes
      # num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.device)
      # num_nodes = torch.multinomial(
          # num_nodes_pmf, batch_size, replacement=True) + 1  # shape B X 1

      # A_list = [
          # A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      # ]

      # sanity check, mask out self loops
      A[:, range(self.max_num_nodes), range(self.max_num_nodes)] = 0
      nl[:, 0] = 0

      A_list = [A[ii, :, :].transpose(0, 1) for ii in range(batch_size)]
      ret = [A_list]
      if nl is not None:
        nl_list = [nl[ii, :] for ii in range(batch_size)]
        ret.append(nl_list)
      if prob is not None:
        ret.append([p for p in prob])

      return list(zip(*ret))

  def sample(self, batch_size, dataset=None, debug=0, search_space=None, sample_method='random'):
    """
      Sample `batch_size` graphs with rejection sampling. We do rejection
      sampling based on the `is_valid` function.
      - If a dataset is provided we also do rejection sampling based on
          `if graph in dataset`
      - If a search space is provided then we call is_valid with that search
          space
      @sample_method - ['random', 'beam', 'beam_return_prob']
    """

    if search_space is None:
      search_space = self.search_space

    if self.config.experimental.use_temperature == 'ramping':
      temperature = 1 + (self.config.experimental.temperature - 1) * num_tries
      if debug:
        print("ramping temp", temperature)
      sample_bs = batch_size
    elif self.config.experimental.use_temperature == 'fixed':
      temperature = self.config.experimental.temperature
    else:
      temperature = 1

    if "beam" in sample_method:
      # trust that beam search sampling will only sample valid archs
      return self({
        'is_sampling': True,
        'batch_size': batch_size,
        'sample_method': sample_method,
        'temperature': temperature,
        'debug': debug,
      })
    elif sample_method == 'random':
      num_tries, max_num_tries = 0, 20
      num_samples = 0
      graphs = []
      while len(graphs) < batch_size and num_tries < max_num_tries:
        # ramp batch size
        sample_bs = batch_size * (num_tries * 5 + 1)

        # ramp temperature
        if self.config.experimental.use_temperature == 'ramping':
          temperature = 1 + (self.config.experimental.temperature - 1) * num_tries
          if debug: print("ramping temp", temperature)
          sample_bs = batch_size

        candidates = self({
          'is_sampling': True,
          'batch_size': sample_bs,
          'sample_method': sample_method,
          'temperature': temperature,
          'debug': debug
        })
        num_samples += len(candidates)

        if self.num_node_labels > 0:
          num_nodes = [
            torch.nonzero(graph[1] == self.num_node_labels - 1)
            for graph in candidates
          ]
          num_nodes = [n.min() + 1 if n.nelement() != 0 else 0 for n in num_nodes]
        else:
          num_nodes = [self.max_num_nodes] * len(candidates)

        candidates = [
          (adj[:N, :N].cpu(), nl[:N].cpu())
          for N, (adj, nl) in zip(num_nodes, candidates)
          if is_valid(search_space, (adj[:N, :N].cpu(), nl[:N].cpu()))
        ]
        if dataset is not None:
          candidates = [graph for graph in candidates if graph not in dataset]
        graphs.extend(candidates)
        num_tries += 1

      eff_str = f"GRAN Effectiveness: {len(graphs)} in {num_samples} for {int(len(graphs)/num_samples*100)}%"
      if self.Q is not None:
        self.Q.log(eff_str)
      elif debug:
        print(eff_str)


      if len(graphs) < batch_size:
        return None
      return graphs[:batch_size]

    else:
      raise "invalid sample method"


def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx, subgraph_idx_base, num_canonical_order,
                           node_label=None, log_phi=None, edge_label=None, log_psi=None,
                           sum_order_log_prob=False, return_neg_log_prob=False, reduction="mean",
                           rewards=None, baseline=0,
                           debug=0):
  """
    Compute likelihood for mixture of Bernoulli model

    Args:
      label: E X 1, see comments above
      log_theta: E X D, see comments above
      log_alpha: E X D, see comments above
      adj_loss_func: BCE loss
      subgraph_idx: E X 1, see comments above
      subgraph_idx_base: B+1, cumulative # of edges in the subgraphs associated with each batch
      num_canonical_order: int, number of node orderings considered
      node_label: 0 or N X 1, a value in [0, num_node_labels - 1] (only works for block_size=1 !!)
      log_phi: 0 or N X num_node_labels,
      edge_label: 0 or E X 1, a value in [0, num_edge_labels - 1]
      log_psi: 0 or E X num_edge_labels,
      sum_order_log_prob: boolean, if True sum the log prob of orderings instead of taking logsumexp
        i.e. log p(G, pi_1) + log p(G, pi_2) instead of log [p(G, pi_1) + p(G, pi_2)]
        This is equivalent to the original GRAN loss.
      return_neg_log_prob: boolean, if True also return neg log prob
      reduction: string, type of reduction on batch dimension ("mean", "sum", "none")

    Returns:
      loss (and potentially neg log prob)
  """

  num_subgraph = subgraph_idx_base[-1] # == subgraph_idx.max() + 1
  B = subgraph_idx_base.shape[0] - 1
  C = num_canonical_order
  E = log_theta.shape[0]
  K = log_theta.shape[1]
  assert E % C == 0
  # neg log likelihood of each edge generation
  adj_nll = torch.stack( # E * K
      [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1
  )
  # mask out self loops
  mask = torch.tensor([0 if i==len(subgraph_idx) - 1 or subgraph_idx[i] != subgraph_idx[i+1] else 1 for i in range(len(subgraph_idx))])
  mask = mask.view(-1, 1).to(adj_nll.device)
  adj_nll = adj_nll * mask
  if debug:
    print(torch.round(torch.sigmoid(log_theta).mean(dim=1)*10**2)/10**2, "\n", label)

  num_edges_per_subgraph = torch.zeros(num_subgraph).to(label.device) # S
  num_edges_per_subgraph = num_edges_per_subgraph.scatter_add(
      0, subgraph_idx, torch.ones_like(subgraph_idx).float().to(label.device)
  )

  # calculate nll per subgraph
  reduce_adj_nll = torch.zeros(num_subgraph, K).to(label.device)
  reduce_adj_nll = reduce_adj_nll.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_nll)

  # log alpha per subgraph
  reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
  reduce_log_alpha = reduce_log_alpha.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
  # mean pool log alpha across all edges
  reduce_log_alpha = reduce_log_alpha / num_edges_per_subgraph.view(-1, 1)
  reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

  # reduce across all mixture components
  log_prob = -reduce_adj_nll + reduce_log_alpha
  log_prob = torch.logsumexp(log_prob, dim=1) # S, K

  bc_log_prob = torch.zeros([B*C]).to(label.device) # B*C
  bc_idx = torch.arange(B*C).to(label.device) # B*C
  bc_const = torch.zeros(B*C).to(label.device)
  bc_size = (subgraph_idx_base[1:] - subgraph_idx_base[:-1]) // C # B
  bc_size = torch.repeat_interleave(bc_size, C) # B*C
  bc_idx = torch.repeat_interleave(bc_idx, bc_size) # S
  bc_log_prob = bc_log_prob.scatter_add(0, bc_idx, log_prob)
  # loss must be normalized for numerical stability
  bc_const = bc_const.scatter_add(0, bc_idx, num_edges_per_subgraph)
  bc_loss = (bc_log_prob / bc_const)

  bc_log_prob = bc_log_prob.reshape(B,C)
  bc_loss = bc_loss.reshape(B,C)
  if sum_order_log_prob:
    b_log_prob = torch.sum(bc_log_prob, dim=1)
    b_loss = torch.sum(bc_loss, dim=1)
  else:
    b_log_prob = torch.logsumexp(bc_log_prob, dim=1)
    b_loss = torch.logsumexp(bc_loss, dim=1)

  if log_phi is not None:
    # warning again! only implemented for block_size = 1 (our use case)
    log_phi = torch.log_softmax(log_phi, dim=1)

    N = node_label.shape[2]
    node_label = node_label.view(B * C, N)
    num_node_labels = log_phi.shape[1]
    num_nodes = [
      torch.nonzero(nl == num_node_labels - 1)
      for nl in node_label
    ]
    num_nodes = torch.tensor([n.min() + 1 if n.nelement() != 0 else 0 for n in num_nodes])

    nl_flat = torch.cat([node_label[i, :num_nodes[i]] for i in range(B * C)])
    node_subgraph_idx = torch.cat([
      torch.full((num_nodes[i].item(),), i // C, dtype=torch.long)
      for i in range(B * C)
    ])

    phi_loglikelihood_pernode = - log_phi[range(len(nl_flat)), nl_flat.long()]

    b_phi_nll = torch.zeros(B, device=label.device)
    b_phi_nll.scatter_add_(0, node_subgraph_idx, phi_loglikelihood_pernode)
    b_phi_loss = b_phi_nll / num_nodes.view(B, C)[:, 0]
  else:
    b_phi_nll = 0
    b_phi_loss = 0

  if log_psi is not None:
    log_psi = torch.log_softmax(log_psi, dim=1)
    # import pdb; pdb.set_trace()

    psi_loglikelihood_peredge = log_psi[range(len(edge_label)), edge_label.long()]
    # we don't want the 0 labels to contribute to the loss
    psi_loglikelihood_peredge = psi_loglikelihood_peredge * (edge_label > 0).float()

    subgraph_psi_nll = torch.zeros(num_subgraph, device=label.device) # TODO fix, this should be size (B,)
    subgraph_psi_nll.scatter_add_(0, subgraph_idx, psi_loglikelihood_peredge)

    # we want to normalize by the number of actually generated edges in each subgraph
    num_gen_edges = torch.zeros(num_subgraph, device=label.device)
    num_gen_edges.scatter_add_(0, subgraph_idx, label)
    num_gen_edges[num_gen_edges != 0] = 1. / num_gen_edges[num_gen_edges != 0]

    subgraph_psi_loss = - subgraph_psi_nll * num_gen_edges

    bc_psi_loss = torch.zeros(B * C, device=label.device)
    bc_psi_loss = bc_psi_loss.scatter_add(0, bc_idx, subgraph_psi_loss)
    b_psi_loss = bc_psi_loss.view(B, C).sum(dim=1)

    bc_psi_nll = torch.zeros(B * C, device=label.device)
    bc_psi_nll = bc_psi_nll.scatter_add(0, bc_idx, -subgraph_psi_nll)
    b_psi_nll = bc_psi_nll.view(B, C).sum(dim=1)
  else:
    b_psi_nll = 0
    b_psi_loss = 0

  if rewards is None:
    rewards = torch.ones(B, device=label.device)

  # probability calculation was for lower-triangular edges
  # must be squared to get probability for entire graph
  # b_neg_log_prob = -2*b_log_prob
  b_neg_log_prob = - b_log_prob + b_phi_nll + b_psi_nll
  # print("iloss", -b_log_prob, b_psi_nll)
  # print("log prob", -b_neg_log_prob)
  b_loss = -b_loss

  if debug:
    print(b_loss, 1./8 * b_psi_loss)
  loss = b_loss + 0.5 * b_phi_loss + 1./8 * b_psi_loss
  # loss = b_loss
  rewards = rewards - baseline
  pos_loss = loss * rewards * (rewards > 0).float()
  neg_loss = torch.log(1-torch.exp(-loss)+1e-6) * rewards * (rewards < 0).float()

  loss = neg_loss + pos_loss

  if reduction == "mean":
    neg_log_prob = b_neg_log_prob.mean()
    loss = loss.mean()
  elif reduction == "sum":
    neg_log_prob = b_neg_log_prob.sum()
    loss = loss.sum()
  else:
    assert reduction == "none"
    neg_log_prob = b_neg_log_prob
    loss = loss

  if return_neg_log_prob:
    return loss, neg_log_prob
  else:
    return loss
