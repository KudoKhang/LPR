# Reference: https://codelearn.io/sharing/logging-python-system-talk-module
import logging
import os
from typing import Optional

from LPR.utils import COLOR


class CustomFormatter(logging.Formatter):
    def __init__(self, format_log, format_log_timer):
        super().__init__()
        self.format_log = format_log
        self.format_log_timer = format_log_timer

    def select_format(self, format_type):
        return {
            logging.DEBUG: COLOR.LIGHTBLUE + format_type + COLOR.NOCOLOR,
            logging.INFO: COLOR.LIGHTCYAN + format_type + COLOR.NOCOLOR,
            logging.WARNING: COLOR.YELLOW + format_type + COLOR.NOCOLOR,
            logging.ERROR: COLOR.RED + format_type + COLOR.NOCOLOR,
            logging.CRITICAL: COLOR.CYAN + format_type + COLOR.NOCOLOR,
        }

    def format(self, record):
        format_type = self.format_log_timer if "_timer.py" in record.pathname else self.format_log
        FORMATS = self.select_format(format_type)
        log_fmt = FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    def __init__(
        self,
        name: str = "faurd",
        level: int = logging.DEBUG,
        path_file: Optional[str] = None,
        root_log: str = "logs",
        format_log: str = "%(asctime)s - %(levelname)s - [in %(pathname)s:%(lineno)d] - %(message)s",
        format_log_timer: str = "%(asctime)s - %(levelname)s - [Timer] - %(message)s",
    ):
        """Logger in json format that writes to a file and console.

        Args:
            name (str): Name of the logger.
            level (str): Level of the logger.
            path_file (str): Path to the log file.
            format_log (str): Format of the log record.

        Attributes:
            logger (logging.Logger): Logger object.

        """
        self.name = name
        self.level = level
        self.path_file = path_file
        self.root_log = root_log

        if not os.path.exists(self.root_log):
            os.makedirs(self.root_log, exist_ok=True)
        self.format_log = format_log
        self.format_log_timer = format_log_timer
        self.format = CustomFormatter(self.format_log, self.format_log_timer)

        self.logger = logging.getLogger(self.name)
        self.configure()

    def configure(self):
        """Configures the logger."""
        if self.logger.level == 0 or self.level < self.logger.level:
            self.logger.setLevel(self.level)

        if len(self.logger.handlers) == 0:
            handler = logging.StreamHandler()
            handler.setLevel(self.level)
            handler.setFormatter(self.format)
            self.logger.addHandler(handler)

        if self.path_file is not None:
            self.path_file = os.path.join(self.root_log, self.path_file)
            path_file_handler = logging.FileHandler(self.path_file)
            path_file_handler.setLevel(self.level)
            path_file_handler.setFormatter(logging.Formatter(self.format_log))
            self.logger.addHandler(path_file_handler)

        self.logger.propagate = False
