import torch

from dataset.nas_data import NASData
from utils.nas_helper import adjs_to_genotype

class DARTSData:
  """
  Two copies of NASData for both normal and reduction cell.
  """

  def __init__(self, config):
    self.config = config
    self.nas_data = NASData(config)
    self.data_dict = {}
    self.data = []

    # TODO remove
    self.ewma = 0
    self.ewma_alpha = config.nas.ewma_alpha
    self.reward_calc = config.nas.reward

  def append(self, new_data):
    mean_reward = sum(x[1] for x in new_data) / len(new_data)
    self.ewma = self.ewma_alpha * mean_reward + (1 - self.ewma_alpha) * self.ewma
    self.nas_data.ewma = self.ewma

    normal_graphs = []
    reduce_graphs = []
    rewards = []
    for (normal, reduce), reward in new_data:
      nadj, nnl = normal
      radj, rnl = reduce
      nadj = nadj.transpose(0, 1).detach().cpu().numpy()
      radj = radj.transpose(0, 1).detach().cpu().numpy()
      nnl = nnl.detach().cpu().numpy()
      rnl = rnl.detach().cpu().numpy()
      genotype = adjs_to_genotype(normal[0], reduce[0])
      if str(genotype) not in self.data_dict:
        self.data_dict[str(genotype)] = (((nadj, nnl), (radj, rnl)), reward)

    data_nodupes = list(self.data_dict.values())
    data_nodupes.sort(key=lambda x: x[1])

    # keep top
    # TODO reduce code duplication between this and nasdata
    # DARTSData should really be a subclass of NASData
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

    if self.reward_calc == 'cdf':
      data_nodupes = [(graph, (i + 1) / len(data_nodupes)) for i, (graph, _) in enumerate(data_nodupes)]

    self.data = data_nodupes


  def __contains__(self, graph):
    normal, reduce = graph
    genotype = adjs_to_genotype(normal[0], reduce[0])
    return str(genotype) in self.data_dict


  def __getitem__(self, index):
    (normal, reduce), reward = self.data[index]
    nor_data = self.nas_data.adj_batch([normal[0]], [normal[1]], reward)
    red_data = self.nas_data.adj_batch([reduce[0]], [reduce[1]], reward)

    return nor_data, red_data

  def __len__(self):
    return len(self.data)

  def collate_fn(self, batch):
    normal_batch = [normal for normal, _ in batch]
    reduce_batch = [reduce for _, reduce in batch]
    return (
      self.nas_data.collate_fn(normal_batch),
      self.nas_data.collate_fn(reduce_batch)
    )

  def rewards(self):
    return torch.tensor([r for _, r in self.data_dict.values()])



