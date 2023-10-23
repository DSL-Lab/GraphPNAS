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


__all__ = ['NAS']


def NAS(config):
  """
  Kicks off the NAS process. Spawns a generator and evaluator process and
  maintains the message queues between them.
  """

  assert config.generator.cls in [
      "GRAN", "DartsGenerator", "OneShotGenerator", 'lstm_nb101',
  ], "invalid generator name"
  assert config.evaluator.cls in [
      "Oracle", "NASBench101", "NASBench201", "NASBench301", "Uniform",
  ], "invalid evaluator name"

  Q = Queues()
  logger = get_logger('exp_logger')

  gen_process = mp.Process(target=gen_target, name="NAS_gen_process", args=(Q, config))
  eva_process = mp.Process(target=eva_target, name="NAS_eva_process", args=(Q, config))

  gen_process.start()
  eva_process.start()
  result = 0
  try:

    Q.push_gen(("sample", config.nas.sample_batch_size))

    while True:
      # handle loggin
      # have centralized logging since it behaves pretty poorly in
      # multiprocessing environmentsg
      message = Q.pop_log()
      if message == "__exit__":
        break

      cmd, args = message
      if cmd == "log":
        logger.info(args)
      if cmd == 'result':
        print(args)
        result = args
  finally:
    Q.push("__exit__")
  return result


def gen_target(Q, config):
  try:
    random.seed(config.seed * 2)
    torch.manual_seed(config.seed * 3)
    np.random.seed(config.seed * 4)

    generator_cls = eval(config.generator.cls)
    generator = generator_cls(config)

    explorer_cls = eval(config.explorer.cls)
    explorer = explorer_cls(config)

    dataset = eval(config.dataset.cls)(config)

    nas_iter = 0
    hist_log = []

    while True:
      message = Q.pop_gen()
      if message == "__exit__":
        break

      cmd, args = message

      if cmd == "sample":
        generator.eval()
        batch_size = args

        explore_p = calculate_explore_p(config, nas_iter)
        if random.random() < explore_p:
          Q.log("Randomly exploring...")
          samples = explorer.sample(batch_size)
        else:
          Q.log("Sampling...")
          samples = generator.sample(batch_size, dataset)
          if samples is None:
            Q.log("Generator converged badly. Randomly exploring...")
            samples = explorer.sample(batch_size)

        # we are assuming here that samples is not on gpu memory
        # obtain rewards for our sample
        Q.log(f"Sampled:\n{fmt_graphs(config, samples)}")
        Q.push_eva(samples)

      if cmd == "train":
        data = args
        hist_log.append((nas_iter, data))
        torch.save(hist_log, os.path.join(config.save_dir, 'hist_log.pt'))
        if len(data) > 0:
          dataset.append(data)
          if len(dataset) > config.nas.max_oracle_evaluations:
            Q.log(f"Reached maximum oracle evaluations, exiting...")
            break

          baseline = dataset.ewma if config.nas.baseline else 0
          Q.log(f"Sample mean={sum(x[1] for x in data)/len(data)}, baseline={baseline}")
        else:
          Q.log(f"No data sampled")

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
          # if epoch % config.train.print_every == config.train.print_every - 1:
          #   Q.log(f"Epoch={epoch}, Train loss={train_loss/len(train_loader)}")

          lr_scheduler.step()

        generator.eval()

        #torch.save(generator.state_dict(), os.path.join(config.save_dir, f"gen_snapshot_iter{nas_iter:02}.pt"))
        nas_iter = nas_iter + 1
        if nas_iter >= config.nas.max_nas_iterations:
            break
        else:
            Q.push_gen(("sample", config.nas.sample_batch_size))

      if nas_iter >= config.nas.max_nas_iterations:
        Q.log(f"Reached maximum nas iterations, exiting...")
        break
  finally:
    # notify exit no matter how we exit, to try kill all the processes
    Q.push("__exit__")


def eva_target(Q, config):
  try:
    random.seed(config.seed * 2 + 1)
    torch.manual_seed(config.seed * 3 + 1)
    np.random.seed(config.seed * 4 + 1)

    evaluator = eval(config.evaluator.cls)(config, Q)

    while True:
      message = Q.pop_eva()
      if message == "__exit__":
        break

      samples = message

      rewards, rewards2 = evaluator.estimate(samples)
      for graph, reward, reward2 in zip(message, rewards, rewards2):
        Q.log(f"Performance Estimate:\n{fmt_graphs(config, [graph])}, val acc={reward} tst acc={reward2}")
      Q.push_gen(("train", list(zip(message, rewards))))

  finally:
    Q.push("__exit__")

