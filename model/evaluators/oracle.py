import time

import torch.multiprocessing as mp

from utils.darts.model import NetworkCIFAR
from utils.nas_helper import is_valid, adj_to_genotype, adjs_to_genotype
from utils.sgd_helper import train

__all__ = ['Oracle']

class Oracle:

  def __init__(self, config, Q):
    self.config = config
    self.Q = Q
    self.num_process = config.oracle.num_process
    self.eval_iter = 0

  def estimate(self, samples):
    ans_Q = mp.Queue()
    proc_queues = []
    processes = []

    for i in range(self.num_process):
      process_Q = mp.Queue()
      processes.append(mp.Process(
        target=train_target, name=f"ora_train_process_{i}",
        args=(i, self.eval_iter, self.config, process_Q, ans_Q),
      ))
      proc_queues.append(process_Q)

    for process in processes:
      process.start()

    scores = [-1] * len(samples)
    num_done = 0

    for graph_idx, graph in enumerate(samples):
      if graph_idx < self.num_process:
        pid = graph_idx
      else:
        # no process available, so we wait
        fin_graph_idx, fin_score, fin_pid = ans_Q.get()
        scores[fin_graph_idx] = fin_score
        pid = fin_pid
        num_done += 1

      graph = samples[graph_idx]
      proc_queues[pid].put((graph_idx, graph))

    for pq in proc_queues:
      pq.put("__exit__")

    for process in processes:
      process.join()

    while num_done < len(samples):
      fin_graph_idx, fin_score, fin_pid = ans_Q.get()
      scores[fin_graph_idx] = fin_score
      pid = fin_pid
      num_done += 1

    self.eval_iter += 1
    return scores, [0] * len(scores)


def train_target(pid, eval_iter, config, in_Q, out_Q):
  while True:
    message = in_Q.get()
    if message == "__exit__":
      break
    graph_idx, graph = message
    model = graph_to_model(graph, config)
    _, val_acc = train(
      config, model, device_id=pid,
      save_name=f"ora_{eval_iter:02}:{graph_idx:02}_weights",
    )
    print(f"done training {val_acc}")
    out_Q.put((graph_idx, val_acc, pid))


def graph_to_model(graph, config):
  search_space = config.generator.search_space
  num_classes = 100 if config.oracle.dataset == "CIFAR100" else 10

  # sanity check
  assert is_valid(search_space, graph)

  if search_space == "smalldarts":
    adj, nl = graph
    genotype = adj_to_genotype(adj)
    return NetworkCIFAR(36, num_classes, 20, config.oracle.auxiliary, genotype)
  elif search_space == "darts":
    normal, reduce = graph
    normal, nl = normal
    reduce, nl = reduce
    genotype = adjs_to_genotype(normal, reduce)
    return NetworkCIFAR(36, num_classes, 20, config.oracle.auxiliary, genotype)
  else:
    # TODO
    return None

