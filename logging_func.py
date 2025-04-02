
# general logging function  2024.08.23

import sys
import logging


def init_logger(log_file, logger_level):
    '''
    log_file: log information output address
    '''

    logger = logging.getLogger()
    logger.setLevel(logger_level)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    error_handler = logging.StreamHandler(sys.stdout)
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(error_handler)

    return logger


if __name__ == '__main__':
    logger = init_logger('test.log', logging.INFO)
    logger.info('hi llo')
    logger.info('hi llo')
    logger.info('hi llo')


