import logging
from typing import Dict, Optional, Union

from .logger import Logger


def print_log(msg,
              logger: Optional[Union[Logger, str]] = None,
              level='INFO') -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (str): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(msg, level)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = Logger.get_current_instance()
        logger_instance.log(msg, level)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if Logger.check_instance_created(logger):
            logger_instance = Logger.get_instance(logger)
            logger_instance.log(msg, level)
        else:
            raise ValueError(f'MMLogger: {logger} has not been created!')
    else:
        raise TypeError(
            '`logger` should be either a logging.Logger object, str, '
            f'"silent", "current" or None, but got {type(logger)}')
