import sys

sys.path.insert(0, ".")

import logging

from LPR.utils.logger import Logger

logger = Logger(level=logging.DEBUG, path_file="test.log").logger


def test_logger():
    logger.info("Log info")
    logger.debug("Log debug")
    logger.warning("Log warning")
    logger.critical("Log critical")
    logger.error("Log error")


if __name__ == "__main__":
    test_logger()
