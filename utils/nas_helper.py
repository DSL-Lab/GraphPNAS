import torch
from utils.darts.genotypes import Genotype, darts_ops, darts_ops_inv

def search_space_params(config):
  """
  returns (max_num_nodes, num_node_labels, num_edge_labels)
  """
  search_space = config.generator.search_space
  assert search_space in ["NB101", "NB201", "darts", "smalldarts", "custom"]

  if search_space == "NB101":
    return 7, 5, 0

  if search_space == "NB201":
    return 4, 0, 5

  if search_space == "darts" or search_space == "smalldarts":
    return 6, 0, 8

  if search_space == "custom":
    return (
      config.generator.max_num_nodes,
      config.generator.num_node_labels,
      config.generator.num_edge_labels,
    )


  assert False, "not implemented"

def hier_space_params(config):
  """
  returns numcells
  """
  search_space = config.generator.search_space
  assert search_space in ["NB101", "darts", "smalldarts", "custom"]

  if search_space == "NB101":
    return 9

  if search_space == "darts" or search_space == "smalldarts":
    return 20

  if search_space == "custom":
    return config.generator.max_num_cells

  assert False, "not implemented"


def is_valid(search_space, graph):
  """
  returns true iff the given graph is contained the specified search space
  """
  assert search_space in ["NB101", "NB201", "darts", "smalldarts", "custom"]

  if search_space == "NB101":
    adj, nl = graph
    # verify dimensions, at most 7 nodes
    w, h = adj.shape
    if w != h or w > 7 or h > 7:
      return False
    if w == 0:
      return False
    if nl.shape != (w,):
      return False

    # at most 9 edges
    if adj.sum() > 9:
      return False

    # check upper triangular (DAG)
    if not torch.allclose(adj, adj.triu(diagonal=1)):
      return False

    # check connected
    in_degree = adj.sum(dim=0)
    if not (in_degree[1:] > 0).all():
      return False
    out_degree = adj.sum(dim=1)
    if not (out_degree[:-1] > 0).all():
      return False

    # first and last node are input/output respectively
    if nl[0] != 0 or nl[-1] != 4:
      return False
    # none of the interior nodes are input/output
    for i in range(1, w-1):
      if nl[i] < 1 or nl[i] > 3:
        return False

    return True
  elif search_space == "NB201":
    from model.evaluators.nas_bench import nasbench201_is_valid
    return nasbench201_is_valid(graph)
  elif search_space == "darts":
    normal, reduce = graph
    return is_valid("smalldarts", normal) and is_valid("smalldarts", reduce)
  elif search_space == "smalldarts":
    adj, nl = graph
    w, h = adj.shape
    if w != 6 or h != 6:
      return False
    for dst in range(1, 6):
      in_deg = 0
      for src in range(dst):
        if adj[src, dst] > 7:
          return False
        if adj[src, dst] > 0:
          in_deg += 1
      if dst == 1 and in_deg > 0:
        return False
      elif dst > 1 and in_deg != 2:
        return False
    return True
  elif search_space == "custom":
    return True


def calculate_explore_p(config, nas_iter):
  method = config.nas.explore_method
  # assert method in ["none", "harmonic", "exponential", "two", "three", "constant", "30", "20", "10", "15"]

  if method == "none":
    return 0.
  if method == "harmonic":
    return 2 / (nas_iter + 2)
  if method == "exponential":
    return 2 ** (-nas_iter / 2)
  if method == "constant":
    return config.nas.explore_p
  if method.isnumeric():
    return 1. if nas_iter < int(method) else 0.
  if method == "two":
    return 1. if nas_iter < 2 else 0.
  if method == "three":
    return 1. if nas_iter < 3 else 0.


def _adj_to_op_list(adj):
  op_list = []
  for dst in range(2, 6):
    for src in range(dst):
      if adj[src, dst] > 0:
        op_id = adj[src, dst].int().item()
        op_list.append((darts_ops[op_id], src))
  return op_list


def adj_to_genotype(adj):
  assert is_valid("smalldarts", (adj, None))
  fixed_reduce = [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)]
  # adj to genotype
  normal = _adj_to_op_list(adj)
  return Genotype(normal=normal, normal_concat=[2, 3, 4, 5], reduce=fixed_reduce, reduce_concat=[2, 3, 4, 5])


def adjs_to_genotype(normal, reduce):
  # print(normal, reduce)
  assert is_valid("darts", ((normal, None), (reduce, None)))
  normal_op_list = _adj_to_op_list(normal)
  reduce_op_list = _adj_to_op_list(reduce)
  return Genotype(normal=normal_op_list, normal_concat=[2, 3, 4, 5], reduce=reduce_op_list, reduce_concat=[2, 3, 4, 5])


def genotype_to_adjs(genotype):
  nor = torch.zeros((6, 6))
  for i, (op_name, src) in enumerate(genotype.normal):
    dst = 2 + i // 2
    op = darts_ops_inv[op_name]
    nor[src, dst] = op
  red = torch.zeros((6, 6))
  for i, (op_name, src) in enumerate(genotype.reduce):
    dst = 2 + i // 2
    op = darts_ops_inv[op_name]
    red[src, dst] = op

  nl = torch.ones(6)
  return (nor, nl), (red, nl)



# def ls_neighbours(config, adj, nl):
  # """
  # Returns a list of all graphs of edit distance 1
  # (one edge added or removed or changed, OR one label changed)
  # """

  # ret = []
  # N = nl.shape[0]

  # for dst in range(N):
    # for src in range(dst):
      # if adj[src, dst]:
        # # one edge removed
        # new_adj = adj.detach().clone()
        # new_adj[src, dst] = 0
        # if config.
        # # edge label change

# fmt in ["default", "adj"]
def fmt_graphs(config, graphs, fmt="default"):
  search_space = config.generator.search_space
  ret = ""
  if search_space == 'darts':
    for normal, reduce in graphs:
      nor_adj, _ = normal
      red_adj, _ = reduce
      if fmt == "default":
        ret += f"{adjs_to_genotype(nor_adj, red_adj)}\n"
      elif fmt == "adj":
        ret += f"Normal:\n{nor_adj}\nReduce:\n{red_adj}\n"
  elif search_space == 'NB201':
    for graph in graphs:
      ret += f"{graph}\n"
  else:
    for graph in graphs:
      adj, nl = graph
      ret += f"{adj}, {nl}\n"

  return ret



