import os
import subprocess

import gdown
from codetiming import Timer
from FAURD.utils.logger import Logger

logger = Logger().logger


@Timer("Download file from GDrive", "{name}: {milliseconds:.2f} ms", logger=logger.info)
def download_from_GDrive(file_id: str, path_local: str):
    """Downloads a file from Google Drive."""
    os.makedirs(os.path.dirname(path_local), exist_ok=True)
    url = f"https://drive.google.com/uc?&id={file_id}&confirm=t"
    gdown.download(url, output=path_local, quiet=False)


@Timer("Download file from Wget", "{name}: {milliseconds:.2f} ms", logger=logger.info)
def download_from_wget(file_id: str, path_local: str):
    """Downloads a file from Wget."""
    cmd = f"wget {file_id} -O {path_local}"
    os.makedirs(os.path.dirname(path_local), exist_ok=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    std_out, std_err = process.communicate()
    logger.error(std_err)
