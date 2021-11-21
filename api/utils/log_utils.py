
import logging
import os


def create_logger_module(module_name):
    """
    Creat logger module
    :param module_name:
    :return:
    """
    logger_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(format=logger_format, datefmt='%d-%m-%Y:%H:%M:%S',
                        level=os.environ.get('LOGGING_LEVEL', logging.INFO))
    logger = logging.getLogger(module_name)
    return logger
