import torch.nn as nn

from model.generators.gran import GRAN

class DartsGenerator(nn.Module):

  def __init__(self, config, Q=None):
    super().__init__()
    self.config = config
    self.Q = Q
    self.normal_gen = GRAN(config, Q);
    self.reduce_gen = GRAN(config, Q);


  def forward(self, batch):
    normal_batch, reduce_batch = batch
    is_sampling = normal_batch.get('is_sampling', False)
    if is_sampling:
      normal_cells = self.normal_gen(input_dict)
      reduce_cells = self.reduce_gen(input_dict)
      return list(zip(normal_cells, reduce_cells))
    else:
      return_neg_log_prob = normal_batch.get('return_neg_log_prob', False)

      if return_neg_log_prob:
        nor_ret = self.normal_gen(normal_batch)
        red_ret = self.reduce_gen(reduce_batch)
        return nor_ret[0] + red_ret[0], nor_ret[1] + red_ret[1]
      else:
        loss = self.normal_gen(normal_batch) + self.reduce_gen(reduce_batch)
        return loss

  def sample(self, batch_size, dataset=None, debug=0, sample_method='random'):
    if sample_method == 'beam' or sample_method == 'beam_return_prob':
      normal_cells = self.normal_gen.sample(
        batch_size, dataset=None, debug=debug, search_space="smalldarts",
        sample_method='beam_return_prob',
      )
      reduce_cells = self.reduce_gen.sample(
        batch_size, dataset=None, debug=debug, search_space="smalldarts",
        sample_method='beam_return_prob',
      )
      candidates = []
      for nor_adj, nor_nl, nor_p in normal_cells:
        for red_adj, red_nl, red_p in reduce_cells:
          candidates.append(((nor_adj, nor_nl), (red_adj, red_nl), nor_p + red_p))
      candidates.sort(key=lambda x: x[2])
      candidates = candidates[-batch_size:]
      return [(nor, red) for nor, red, _ in candidates]
    elif sample_method == 'random':
      ret = []
      while len(ret) < batch_size:
        # we sample with smalldarts search space, as we can only check validity
        # of one cell at a time in the gran
        normal_cells = self.normal_gen.sample(
          batch_size, dataset=None, debug=debug, search_space="smalldarts",
          sample_method=sample_method,
        )
        reduce_cells = self.reduce_gen.sample(
          batch_size, dataset=None, debug=debug, search_space="smalldarts",
          sample_method=sample_method,
        )
        if normal_cells is None or reduce_cells is None:
          return None
        candidates = list(zip(normal_cells, reduce_cells))
        if dataset is not None:
          candidates = [
            (nor, red) for nor, red in candidates
            if (nor, red) not in dataset
          ]
        ret += candidates
      return ret[:batch_size]
    else:
      raise "invalid sample method"


