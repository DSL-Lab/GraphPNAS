import os
import sys
import time
import glob
import numpy as np
import torch
from utils.logger import get_logger
from utils.torch_helper import FastDataLoader
import utils.darts.utils as utils
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


logging = get_logger('exp_logger')

def train(config, model, device_id=None, is_test=False, quiet=True, save_name="weights"):
  """
  device_id - if device_id is None then use dataparallel (all devices)
  """
  if config.debug and model == None:
    # debugging purposes
    return 0., 0.

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  if device_id is not None:
    device = f"cuda:{device_id}"
  else:
    device = "cuda:0"

  torch.cuda.set_device(device)
  cudnn.benchmark = True
  cudnn.enabled=True

  # model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  # model = model.cuda()
  if device_id is None:
    if not quiet:
      logging.info("Using DataParallel")
    model = nn.DataParallel(model)
  model.to(device)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      config.oracle.lr,
      momentum=config.oracle.momentum,
      weight_decay=config.oracle.weight_decay
      )

  # TODO cache this
  if config.oracle.dataset == "CIFAR10":
    if not quiet:
      logging.info("Loading CIFAR10")
    train_transform, valid_transform = utils._data_transforms_cifar10(config)
    train_data = dset.CIFAR10(root=config.dataset.CIFAR_path, train=True, download=True, transform=train_transform)
    if is_test:
      valid_data = dset.CIFAR10(root=config.dataset.CIFAR_path, train=False, download=True, transform=valid_transform)
    else:
      valid_data = dset.CIFAR10(root=config.dataset.CIFAR_path, train=True, download=True, transform=valid_transform)
  elif config.oracle.dataset == "CIFAR100":
    if not quiet:
      logging.info("Loading CIFAR100")
    train_transform, valid_transform = utils._data_transforms_cifar100(config)
    train_data = dset.CIFAR100(root=config.dataset.CIFAR_path, train=True, download=True, transform=train_transform)
    if is_test:
      valid_data = dset.CIFAR100(root=config.dataset.CIFAR_path, train=False, download=True, transform=valid_transform)
    else:
      valid_data = dset.CIFAR100(root=config.dataset.CIFAR_path, train=True, download=True, transform=valid_transform)


  if is_test:
    if not quiet:
      logging.info("Loading test set")
    train_queue = FastDataLoader(
        train_data, batch_size=config.oracle.batch_size, shuffle=True, pin_memory=True, num_workers=config.oracle.num_workers)
    valid_queue = FastDataLoader(
        valid_data, batch_size=config.oracle.batch_size, shuffle=False, pin_memory=False, num_workers=config.oracle.num_workers)
  else:
    idxs = list(range(50000))
    train_idx, valid_idx = idxs[:40000], idxs[40000:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_queue = FastDataLoader(
        train_data, batch_size=config.oracle.batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=config.oracle.num_workers,
    )
    valid_queue = FastDataLoader(
        valid_data, batch_size=config.oracle.batch_size, sampler=valid_sampler,
        pin_memory=True, num_workers=0,
    )

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.oracle.epochs))

  for epoch in range(config.oracle.epochs):
    scheduler.step()
    if not quiet:
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    if device_id is None:
      if not quiet:
        logging.info(f'dataparallel drop_path_prob {config.oracle.drop_path_prob * epoch / config.oracle.epochs}')
      model.module.set_drop_path_prob(config.oracle.drop_path_prob * epoch / config.oracle.epochs)
    else:
      if not quiet:
        logging.info(f'not dataparallel drop_path_prob {config.oracle.drop_path_prob * epoch / config.oracle.epochs}')
      model.set_drop_path_prob(config.oracle.drop_path_prob * epoch / config.oracle.epochs)

    train_acc, train_obj = train_iter(train_queue, model, criterion, optimizer, config)
    if not quiet:
      logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer_iter(valid_queue, model, criterion, config)
    if not quiet:
      logging.info('valid_acc %f', valid_acc)

    if config.oracle.save_snapshots:
      utils.save(model, os.path.join(config.save_dir, f'{save_name}.pt'))

  return train_acc, valid_acc


def train_iter(train_queue, model, criterion, optimizer, config):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if config.oracle.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += config.oracle.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), config.oracle.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if config.debug:
      # just one forward and backward pass is enough code coverage
      print("[DEBUG]: breaking from training early")
      break

    if step % config.oracle.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer_iter(valid_queue, model, criterion, config):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(non_blocking=True)

    with torch.no_grad():
      logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % config.oracle.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


