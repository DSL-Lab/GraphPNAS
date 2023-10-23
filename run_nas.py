import os
import sys
import torch
import torch.multiprocessing as mp
import logging
import traceback
import numpy as np
from pprint import pformat

from model.nas import NAS
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config
torch.set_printoptions(profile='full')


def main():
  args = parse_arguments()
  config = get_config(args.config_file, is_test=args.test, args=args)
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()

  # log info
  log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
  logger = setup_logging(args.log_level, log_file)
  logger.info("Writing log file to {}".format(log_file))
  logger.info("Exp instance id = {}".format(config.run_id))
  logger.info("Exp comment = {}".format(args.comment))
  logger.info("Config =")
  logger.info(">" * 80)
  logger.info(pformat(config))
  logger.info("<" * 80)

  if config.debug:
    logger.info("Entering debug mode, continue with caution...")
    logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    logger.info("@         DEBUG MODE         @")
    logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

  # Run the experiment
  try:
    NAS(config)
  except:
    logger.error(traceback.format_exc())

  sys.exit(0)


if __name__ == "__main__":
  # cuda complains if you use fork
  mp.set_start_method('spawn')

  main()
