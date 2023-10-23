import random
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from dataset.nas_data import NASData
from dataset.darts_data import DARTSData

from model.evaluators.oracle import Oracle
from model.evaluators.nas_bench import NASBench101, NASBench301, NASBench201
from model.evaluators.uniform import Uniform
from model.generators.erdos_renyi import ErdosRenyi, SkipExplorer
from model.generators.random_nb101 import RandomNB101
from model.generators.gran import GRAN
from model.generators.lstm import lstm_nb101
from model.generators.darts import DartsGenerator
from model.generators.oneshot import OneShotGenerator

from utils.logger import get_logger
from utils.mp_helper import Queues
from utils.nas_helper import calculate_explore_p, fmt_graphs

def pretrain(config):
  generator_cls = eval(config.generator.cls)
  generator = generator_cls(config)

  explorer_cls = eval(config.explorer.cls)
  explorer = explorer_cls(config)

  dataset = eval(config.dataset.cls)(config)

  nas_iter = 0
  evaluator = eval(config.evaluator.cls)(config, None)
  import ipdb; ipdb.set_trace()
  all_archs = evaluator.get_all_data()
   
  data = [(arch, 1) for arch in all_archs] 
  dataset.append(data)

  baseline = dataset.ewma if config.nas.baseline else 0
  Q.log(f"Sample mean={sum(x[1] for x in data)/len(data)}, baseline={baseline}")

  # reset generator
  generator = generator_cls(config, Q=Q)
  generator.to(config.generator.device)

  train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.train.batch_size,
    shuffle=config.train.shuffle,
    num_workers=config.train.num_workers,
    collate_fn=dataset.collate_fn,
    drop_last=False
  )

  params = filter(lambda p: p.requires_grad, generator.parameters())

  if config.train.optimizer == 'SGD':
    optimizer = optim.SGD(
    params,
    lr=config.train.lr,
    momentum=config.train.momentum,
    weight_decay=config.train.wd
    )
  elif config.train.optimizer == 'Adam':
    optimizer = optim.Adam(
    params, lr=config.train.lr, weight_decay=config.train.wd
    )
  else:
    raise ValueError("Non-supported optimizer!")

  lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=config.train.lr_decay_epoch,
    gamma=config.train.lr_decay
  )

  for epoch in range(config.train.max_epoch):
    generator.train()

    train_loss = 0.
    for batch in train_loader:
    optimizer.zero_grad()

    loss = generator(batch)
    loss.backward()

    optimizer.step()

    train_loss += loss.item()

    if epoch == 0 or epoch == config.train.max_epoch - 1:
    Q.log(f"Epoch={epoch}, Train loss={train_loss/len(train_loader)}")

    lr_scheduler.step()

  generator.eval()

  #torch.save(generator.state_dict(), os.path.join(config.save_dir, f"gen_snapshot_iter{nas_iter:02}.pt"))
  nas_iter = nas_iter + 1
  if nas_iter >= config.nas.max_nas_iterations:
    break
  else:
    Q.push_gen(("sample", config.nas.sample_batch_size))