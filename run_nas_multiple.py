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
  config = get_config(args.config_file, is_test=args.test)
  log_file = os.path.join(config.save_dir, f"log_exp_all.txt".format(config.run_id))
  logger = setup_logging(args.log_level, log_file)
  results = []
  for seed in range(21, 26):
    config.seed = seed

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    config.use_gpu = config.use_gpu and torch.cuda.is_available()

    # log info
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
      result = NAS(config)
    except:
      logger.error(traceback.format_exc())
     
    logger.info(str(result))
    results.append(result)
    logger.info("Test - All")
    logger.info("Mean: {}".format(np.mean([x[1]['test-accuracy'] for x in results])))
    logger.info("Std: {}".format(np.std([x[1]['test-accuracy'] for x in results])))

  logger.info("Valid")
  if 'valid-accuracy' in results[0][0]:
    logger.info("Mean: {}".format(np.mean([x[0]['valid-accuracy'] for x in results])))
    logger.info("Std: {}".format(np.std([x[0]['valid-accuracy'] for x in results])))
  elif 'valtest-accuracy' in results[0][0]:
    logger.info("Mean {}:".format(np.mean([x[0]['valtest-accuracy'] for x in results])))
    logger.info("Std: {}".format(np.std([x[0]['valtest-accuracy'] for x in results])))
  logger.info("Test")
  logger.info("Mean: {}".format(np.mean([x[0]['test-accuracy'] for x in results])))
  logger.info("Std: {}".format(np.std([x[0]['test-accuracy'] for x in results])))
  torch.save(results, os.path.join(config.save_dir, "result.pth"))
  # print("Cifar10")
  # print("Test")
  # print("Mean:", np.mean([x[1]['test-accuracy'] for x in results]))
  # print("Std:", np.std([x[1]['test-accuracy'] for x in results]))

  # print("Cifar10-Valid")
  # print("Valid")
  # print("Mean:", np.mean([x[0]['valid-accuracy'] for x in results]))
  # print("Std:", np.std([x[0]['valid-accuracy'] for x in results]))
  # print("Test")
  # print("Mean:", np.mean([x[0]['test-accuracy'] for x in results]))
  # print("Std:", np.std([x[0]['test-accuracy'] for x in results]))

  sys.exit(0)


if __name__ == "__main__":
  # cuda complains if you use fork
  mp.set_start_method('spawn')

  main()
