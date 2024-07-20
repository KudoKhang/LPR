import os

from lpr.utils.config import cfg
from lpr.utils.downloader import download_from_GDrive
from lpr.utils.logger import logger


def download_checkpoints(list_checkpoints):
    for checkpoints in list_checkpoints:
        if not os.path.exists(checkpoints.path_local):
            download_from_GDrive(file_id=checkpoints.file_id, path_local=checkpoints.path_local)
        else:
            logger.warning(f"{checkpoints.path_local} already exist!")


if __name__ == "__main__":
    list_checkpoints = [cfg.plate, cfg.character, cfg.classify, cfg.corner, cfg.classify_tf]
    download_checkpoints(list_checkpoints)
