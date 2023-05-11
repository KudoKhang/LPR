import sys

sys.path.insert(0, ".")
import os

from LPR.utils.config import cfg
from LPR.utils.downloader import download_from_GDrive
from LPR.utils.logger import Logger

cfg = cfg()
logger = Logger().logger


def download_checkpoints(list_checkpoints):
    for checkpoints in list_checkpoints:
        if not os.path.exists(checkpoints.path_local):
            download_from_GDrive(file_id=checkpoints.file_id, path_local=checkpoints.path_local)
        else:
            logger.warning(f"{checkpoints.path_local} already exist!")


if __name__ == "__main__":
    list_checkpoints = [cfg.plate, cfg.character, cfg.classify, cfg.corner, cfg.classify_tf]
    download_checkpoints(list_checkpoints)
