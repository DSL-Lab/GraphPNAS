from utils.nas_helper import is_valid, adj_to_genotype, adjs_to_genotype
NB201_API = None

class NASBench101:

  def __init__(self, config, Q):
    # this is quite slow so we don't want to do it more often than we need to
    from nasbench.api import NASBench101
    self.config = config
    self.max_oracle_evaluations = config.nas.max_oracle_evaluations
    self.Q = Q
    self.api = NASBench101(config.dataset.NB101_path)

    self.cache = set()

  def estimate(self, samples):
    scores = []
    scores2 = []
    for adj, nl in samples:
      run0 = self.api.query_perf(adj, nl, epochs=108, stop_halfway=False, run=0, test=False)
      run1 = self.api.query_perf(adj, nl, epochs=108, stop_halfway=False, run=1, test=False)
      run2 = self.api.query_perf(adj, nl, epochs=108, stop_halfway=False, run=2, test=False)
      val_acc = (run0 + run1 + run2) / 3
      scores.append(val_acc)
      test0 = self.api.query_perf(adj, nl, epochs=108, stop_halfway=False, run=0, test=True)
      test1 = self.api.query_perf(adj, nl, epochs=108, stop_halfway=False, run=1, test=True)
      test2 = self.api.query_perf(adj, nl, epochs=108, stop_halfway=False, run=2, test=True)
      tst_acc = (test0 + test1 + test2) / 3
      scores2.append(tst_acc)

      hash = self.api.get_hash(adj, nl)
      if hash not in self.cache:
        self.cache.add(hash)
        self.Q.log(f"Evaluation {len(self.cache)} of {self.max_oracle_evaluations}: val={val_acc}, tst={tst_acc}")
    if len(self.cache) >= self.max_oracle_evaluations:
      self.Q.log(f"Reached {self.max_oracle_evaluations}, sending exit signal")
      self.Q.exit()
    return scores, scores2

class NASBench201:
  def __init__(self, config, Q=None):
    global NB201_API
    # if 'NB201_API' not in globals():
    if NB201_API is None:
      from nasbench201 import create
      print("Creating NASBench201 API")
      NB201_API = create("./data/nasbench_201", 'tss', fast_mode=True, verbose=False)
    self.api = NB201_API
    # self.api = create("./data/NATS-tss-v1_0-3ffb9.pickle.pbz2", 'tss', verbose=False)
    self.config = config
    self.Q = Q
    self.max_oracle_evaluations = config.nas.max_oracle_evaluations
    self.time = 0
    self.step = 0
    self.cache = set()
    self.search_hist = []
    self.dataset = config.oracle.dataset
    self.budget = 100000000
    self.max_step = 100

  def estimate(self, samples):
    for adj, nl in samples:
      assert adj.shape[0] == 4
    arch_strs = [get_nb201_arch_str(graph) for graph in samples]
    est_val = []
    est_test = []
    for arch_str in arch_strs:
      self.cache.add(arch_str)
      if self.dataset == 'cifar10':
        est_12_ep = self.api.get_more_info(arch_str, 'cifar10-valid', hp="12", is_random=True)
        time_cost = est_12_ep['train-all-time'] + est_12_ep['valid-per-time']
        valid_acc = est_12_ep['valid-accuracy']
      elif self.dataset == 'ImageNet16-120':
        est_12_ep = self.api.get_more_info(arch_str, 'ImageNet16-120', hp="200", is_random=True)
        time_cost = est_12_ep['train-all-time'] + est_12_ep['valid-per-time']
        valid_acc = est_12_ep['valid-accuracy']
        # valid_acc, _, time_cost, _ = self.api.simulate_train_eval(arch_str, self.dataset, hp='12')
      else:
        est_12_ep = self.api.get_more_info(arch_str, 'cifar100', hp="200", is_random=True)
        time_cost = est_12_ep['train-all-time'] + est_12_ep['valid-per-time']
        valid_acc = est_12_ep['valid-accuracy']
        # valid_acc, _, time_cost, _ = self.api.simulate_train_eval(arch_str, self.dataset, hp='12')
      est_val.append(valid_acc)
      est_200_ep = self.api.get_more_info(arch_str, self.dataset, hp="200", is_random=False)
      print(est_200_ep)
      est_test.append(est_200_ep['test-accuracy'])
      self.time = self.time + time_cost
      self.step = self.step + 1
      self.search_hist.append((arch_str, est_val[-1], est_test[-1]))

    print(f"Budget: {self.time}/{self.budget}, step: {self.step}/{self.max_step}")
    if self.time >= self.budget or self.step >= self.max_step:
      self.Q.log(f"Reached budget limit, sending exit signal")
      self.search_hist.sort(key=lambda x: x[1], reverse=True)
      print("Search best:", self.search_hist[:3])
      self.Q.push_log(("result", self.get_best()))
      self.Q.exit()
    return est_val, est_test

  def get_best(self):
      self.search_hist.sort(key=lambda x: x[1], reverse=True)
      print("Search best:", self.search_hist[:3])
      arch_str = self.search_hist[0][0]
      if self.dataset == 'cifar10':
        c10v = self.api.get_more_info(arch_str, 'cifar10-valid', hp="200", is_random=False)
        c10 = self.api.get_more_info(arch_str, 'cifar10', hp="200", is_random=False)
      else:
        c10 = c10v = self.api.get_more_info(arch_str, self.dataset, hp="200", is_random=False)
      return c10v, c10
  
  def get_all_archs(self):
    all_archs = self.api.meta_archs
    return [get_nb201_arch_from_str(arch_str) for arch_str in all_archs]

def get_nb201_arch_str(sample):
  adj, nl = sample
  adj = adj.numpy().astype(int)
  search_space = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
  ]
  arch_str = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
    search_space[adj[0][1]],
    search_space[adj[0][2]],
    search_space[adj[1][2]],
    search_space[adj[0][3]],
    search_space[adj[1][3]],
    search_space[adj[2][3]],
  )
  return arch_str

def get_nb201_arch_from_str(arch_str):
  adj = np.zeros((4, 4), dtype=np.int)
  nl = np.zeros((4,), dtype=np.int)
  search_space = [
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
  ]
  for i, node_str in enumerate(arch_str.split('+')):
    node_str = node_str.strip('|')
    for j, op_str in enumerate(node_str.split('|')):
      op_str = op_str.split('~')[0]
      op = search_space.index(op_str)
      adj[i, j] = op
  return adj, nl

def nasbench201_is_valid(sample):
  global NB201_API
  # if 'NB201_API' not in globals():
  if NB201_API is None:
    from nasbench201 import create
    print("Creating NASBench201 API")
    NB201_API = create("./data/nasbench_201", 'tss', fast_mode=True, verbose=False)
  if isinstance(sample, str):
    arch_str = sample
  else:
    arch_str = get_nb201_arch_str(sample)
  return arch_str in NB201_API.archstr2index


class NASBench301:

  def __init__(self, config, Q=None):
    import nasbench301 as nb
    self.config = config
    self.search_space = config.generator.search_space
    self.max_oracle_evaluations = config.nas.max_oracle_evaluations
    self.noisy = config.experimental.nb301_noise
    self.Q = Q

    print("Loading NB301...")
    self.performance_model = nb.load_ensemble(config.dataset.NB301_path)
    print("Done loading NB301!")

    self.cache = set()
    self.best_so_far = 0


  def estimate(self, samples):
    scores = []
    scores2 = []
    for graph in samples:
      if self.search_space == 'darts':
        normal, reduce = graph
        genotype = adjs_to_genotype(normal[0], reduce[0]) # zeroth index is adj
      elif self.search_space == "smalldarts":
        adj, _ = graph
        genotype = adj_to_genotype(adj)
      else:
        raise "bad search space"
      pred = self.performance_model.predict(config=genotype, representation="genotype", with_noise=self.noisy)
      actu = self.performance_model.predict(config=genotype, representation="genotype", with_noise=False)
      scores.append(pred)
      scores2.append(actu)

      hash = str(genotype)
      if hash not in self.cache:
        self.cache.add(hash)
        if actu > self.best_so_far:
          self.best_so_far = actu
        if self.Q is not None:
          self.Q.log(f"Evaluation {len(self.cache)} of {self.max_oracle_evaluations}: val={pred}, test={actu}, best={self.best_so_far}")
    if self.Q is not None and len(self.cache) >= self.max_oracle_evaluations:
      self.Q.log(f"Reached {self.max_oracle_evaluations}, sending exit signal")
      self.Q.exit()
    return scores, scores2

