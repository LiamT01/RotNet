import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from loguru import logger


def get_logger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def reduce_losses(loss_list):
    sum_losses = defaultdict(float)
    sum_relative_losses = defaultdict(float)
    for loss_per_batch in loss_list:
        for loss in loss_per_batch:
            sum_losses[loss['name']] += loss['loss'].item()
            sum_relative_losses[loss['name']] += loss['relative loss'].item()
    return {
        'count': len(loss_list),
        'sum loss': sum_losses,
        'sum relative loss': sum_relative_losses,
    }


def get_num_digits(number):
    return int(np.ceil(np.log(number) / np.log(10) + 1))
