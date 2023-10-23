import torch
import time
import os
import pickle
import glob
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
from utils.data_helper import *
from utils.nas_helper import search_space_params


class NASData(object):
  """
  Modified version of GRANData

  Changes:
  - Maintains and processes node and edge features
  - Keeps track of reward for each graph, as well as an ewma baseline
  - No longer loads and stores graphs from the disk (our dataset is small enough
    to store entirely in ram)
  - When num_canonical_order > 1, considers different topological orders instead
    of bfs, dfs, etc., as in the original
  """

  def __init__(self, config):
    self.config = config
    self.max_num_nodes, self.num_node_labels, self.num_edge_labels = \
        search_space_params(config)
    self.block_size = config.generator.block_size
    self.stride = config.generator.sample_stride

    self.data = [] # contains pairs (list of graph orderings, reward)
    self.raw_data = [] # contains tuples (adj, nl, accuracy)
    self.data_dict = {} # map between hash(adj, nl) -> (adj, nl, reward)

    self.ewma = 0
    self.ewma_alpha = config.nas.ewma_alpha
    self.reward_calc = config.nas.reward
    assert self.reward_calc in ['acc', 'cdf']

    self.npr = np.random.RandomState(config.seed)
    self.num_canonical_order = config.generator.num_canonical_order


  def append(self, new_data):
    mean_reward = sum(x[1] for x in new_data) / len(new_data)
    self.ewma = self.ewma_alpha * mean_reward + (1 - self.ewma_alpha) * self.ewma

    for (adj_torch, nl_torch), reward in new_data:
      adj = adj_torch.transpose(0, 1).detach().cpu().numpy()
      nl = nl_torch.detach().cpu().numpy()
      self.data_dict[(adj.tobytes(), nl.tobytes())] = (adj, nl, reward)

    data_nodupes = list(self.data_dict.values())

    # sort by reward
    data_nodupes.sort(key=lambda x: x[2])

    # keep top
    keep_top_n = 1000000000
    if isinstance(self.config.nas.keep_top, str):
      # progressive keep_top
      print(self.config.nas.keep_top)
      for limit, val in eval(self.config.nas.keep_top):
        print(limit, "|", val)
        if len(data_nodupes) > limit:
          keep_top_n = val
    else:
      # otherwise its an int
      keep_top_n = self.config.nas.keep_top
    print(f"keeping top {keep_top_n}")
    if len(data_nodupes) > keep_top_n:
      data_nodupes = data_nodupes[-keep_top_n:]

    self.raw_data = data_nodupes
    # only one node ordering for now
    # but here is where we could add more topo orderings
    if self.reward_calc == 'acc':
      self.data = [([(adj, nl)], reward) for adj, nl, reward in data_nodupes]
    elif self.reward_calc == 'cdf':
      self.data = [([(adj, nl)], (i + 1) / len(data_nodupes)) for i, (adj, nl, reward) in enumerate(data_nodupes)]


  def __contains__(self, graph):
    adj, nl = graph
    if isinstance(adj, torch.Tensor):
      adj = adj.transpose(0, 1).detach().cpu().numpy()
    if isinstance(nl, torch.Tensor):
      nl = nl.detach().cpu().numpy()
    return (adj.tobytes(), nl.tobytes()) in self.data_dict


  def __getitem__(self, index):
    # load graph
    graph_list, reward = self.data[index]
    L_list = [L for L, _ in graph_list]
    nl_list = [nl for _, nl in graph_list]

    return self.adj_batch(L_list, nl_list, reward)


  def adj_batch(self, L_list, nl_list, reward):
    K = self.block_size
    N = self.max_num_nodes
    S = self.stride

    adj_list = [(L > 0).astype(L.dtype) for L in L_list]

    num_nodes = adj_list[0].shape[0]
    num_subgraphs = int(np.floor((num_nodes - K) / S) + 1)

    start_time = time.time()

    edges = []
    node_idx_gnn = []
    node_idx_feat = []
    node_ids = []
    label = []
    edge_label = []
    edge_feat = []
    subgraph_size = []
    subgraph_idx = []
    att_idx = []
    subgraph_count = 0

    prev = time.time()
    for ii in range(len(adj_list)):
      # loop over different orderings
      adj_full = adj_list[ii]
      L_full = L_list[ii]
      # adj_tril = np.tril(adj_full, k=-1)

      idx = -1
      for jj in range(0, num_nodes, S):
        # loop over different subgraphs
        idx += 1

        ### for each size-(jj+K) subgraph, we generate edges for the new block of K nodes
        if jj + K > num_nodes:
          break

        ### get graph for GNN propagation
        adj_block = np.pad(
            adj_full[:jj, :jj], ((0, K), (0, K)),
            'constant',
            constant_values=1.0)  # assuming fully connected for the new block
        adj_block = np.tril(adj_block, k=-1)
        adj_block = adj_block + adj_block.transpose()
        adj_block = torch.from_numpy(adj_block).to_sparse()
        edges += [adj_block.coalesce().indices().long()]

        ### get attention index
        # exist node: 0
        # newly added node: 1, ..., K
        if jj == 0:
          att_idx += [np.arange(1, K + 1).astype(np.uint8)]
        else:
          att_idx += [
              np.concatenate([
                  np.zeros(jj).astype(np.uint8),
                  np.arange(1, K + 1).astype(np.uint8)
              ])
          ]

        ### get node feature index for GNN input
        # use inf to indicate the newly added nodes where input feature is zero
        if jj == 0:
          node_idx_feat += [np.ones(K) * np.inf]
        else:
          node_idx_feat += [
              np.concatenate([np.arange(jj) + ii * N,
                              np.ones(K) * np.inf])
          ]

        ### get node index for GNN output
        idx_row_gnn, idx_col_gnn = np.meshgrid(
            np.arange(jj, jj + K), np.arange(jj + K))
        idx_row_gnn = idx_row_gnn.reshape(-1, 1)
        idx_col_gnn = idx_col_gnn.reshape(-1, 1)
        node_idx_gnn += [
            np.concatenate([idx_row_gnn, idx_col_gnn],
                           axis=1).astype(np.int64)
        ]

        ### get predict label
        label += [
            adj_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
        ]
        edge_label += [
            L_full[idx_row_gnn, idx_col_gnn].flatten().astype(np.uint8)
        ]

        num_edges = edges[-1].shape[1]
        one_hot_edge_feat = np.zeros((num_edges, self.num_edge_labels))
        if self.num_edge_labels > 0:
          L_block = np.pad(
              L_full[:jj, :jj], ((0, K), (0, K)),
              'constant',
              constant_values=0.0)
          for i in range(num_edges):
            u, v = edges[-1][:, i]
            if u < v:
              u, v = v, u
            one_hot_edge_feat[i, int(L_block[u, v])] = 1
        edge_feat += [
            one_hot_edge_feat
        ]

        subgraph_size += [jj + K]
        node_ids += [np.arange(subgraph_size[-1])]
        subgraph_idx += [
            np.ones_like(label[-1]).astype(np.int64) * subgraph_count
        ]
        subgraph_count += 1

    ### adjust index basis for the selected subgraphs
    cum_size = np.cumsum([0] + subgraph_size).astype(np.int64)
    for ii in range(len(edges)):
      edges[ii] = edges[ii] + cum_size[ii]
      node_idx_gnn[ii] = node_idx_gnn[ii] + cum_size[ii]

    # print("graph idx time", time.time() - prev)
    ### pack tensors
    data = {}
    data['rewards'] = reward
    data['adj'] = np.tril(np.stack(adj_list, axis=0), k=-1)
    data['node_label'] = np.stack(nl_list, axis=0)
    data['edges'] = torch.cat(edges, dim=1).t().long()
    data['node_idx_gnn'] = np.concatenate(node_idx_gnn)
    data['node_idx_feat'] = np.concatenate(node_idx_feat)
    data['node_ids'] = np.concatenate(node_ids)
    data['label'] = np.concatenate(label)
    data['edge_label'] = np.concatenate(edge_label)
    data['edge_feat'] = np.concatenate(edge_feat)
    data['att_idx'] = np.concatenate(att_idx)
    data['subgraph_idx'] = np.concatenate(subgraph_idx)
    data['subgraph_count'] = subgraph_count
    data['num_nodes'] = num_nodes
    data['subgraph_size'] = subgraph_size
    data['num_count'] = sum(subgraph_size)

    end_time = time.time()

    return data

  def __len__(self):
    return len(self.data)

  def collate_fn(self, batch):
    assert isinstance(batch, list)
    start_time = time.time()
    batch_size = len(batch)
    N = self.max_num_nodes
    C = self.num_canonical_order

    data = {}

    pad_size = [self.max_num_nodes - bb['num_nodes'] for bb in batch]
    subgraph_idx_base = np.array([0] +
                                 [bb['subgraph_count'] for bb in batch])
    subgraph_idx_base = np.cumsum(subgraph_idx_base)

    data['is_sampling'] = False
    data['rewards'] = torch.tensor([bb['rewards'] for bb in batch], dtype=torch.float) # B X N
    data['baseline'] = self.ewma

    data['subgraph_idx_base'] = torch.from_numpy(
      subgraph_idx_base)

    data['num_nodes_gt'] = torch.from_numpy(
        np.array([bb['num_nodes'] for bb in batch])).long().view(-1)

    data['adj'] = torch.from_numpy(
      np.stack(
        [
          np.pad(
            bb['adj'], ((0, 0), (0, pad_size[ii]), (0, pad_size[ii])),
            'constant', constant_values=0.0
          ) for ii, bb in enumerate(batch)
        ],
      axis=0)
    ).float()  # B X C X N X N

    data['node_label'] = torch.from_numpy(
      np.stack(
        [
          np.pad(
            bb['node_label'], ((0, 0), (0, pad_size[ii])),
            'constant', constant_values=0.0
          ) for ii, bb in enumerate(batch)
        ],
      axis=0)
    ).long() # B X C X N

    idx_base = np.array([0] + [bb['num_count'] for bb in batch])
    idx_base = np.cumsum(idx_base)

    data['edges'] = torch.cat(
        [bb['edges'] + idx_base[ii] for ii, bb in enumerate(batch)],
        dim=0).long()

    data['node_idx_gnn'] = torch.from_numpy(
        np.concatenate(
            [
                bb['node_idx_gnn'] + idx_base[ii]
                for ii, bb in enumerate(batch)
            ],
            axis=0)).long()

    data['node_ids'] = torch.from_numpy(
        np.concatenate([bb['node_ids'] for bb in batch], axis=0)).long()

    data['att_idx'] = torch.from_numpy(
        np.concatenate([bb['att_idx'] for bb in batch], axis=0)).long()

    # shift one position for padding 0-th row feature in the model
    node_idx_feat = np.concatenate(
        [
            bb['node_idx_feat'] + ii * C * N
            for ii, bb in enumerate(batch)
        ],
        axis=0) + 1
    node_idx_feat[np.isinf(node_idx_feat)] = 0
    node_idx_feat = node_idx_feat.astype(np.int64)
    data['node_idx_feat'] = torch.from_numpy(node_idx_feat).long()

    data['label'] = torch.from_numpy(
        np.concatenate([bb['label'] for bb in batch])).float()
    data['edge_label'] = torch.from_numpy(
        np.concatenate([bb['edge_label'] for bb in batch])).float()
    if self.num_edge_labels > 0:
      data['edge_feat'] = torch.from_numpy(
          np.concatenate([bb['edge_feat'] for bb in batch])).float()

    data['subgraph_idx'] = torch.from_numpy(
        np.concatenate([
            bb['subgraph_idx'] + subgraph_idx_base[ii]
            for ii, bb in enumerate(batch)
        ])).long()


    end_time = time.time()
    # print('collate time = {}'.format(end_time - start_time))

    return data
