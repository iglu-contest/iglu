from .pipeline import IGLUDataPipeline
from .download import download
import os

from minerl_patched.data.version import DATA_VERSION, FILE_PREFIX, VERSION_FILE_NAME


def make(data_dir=None, num_workers=4, worker_batch_size=32, minimum_size_to_dequeue=32,
         force_download=False):
    """
    Initalizes the data loader with the chosen environment
    
    Args:
        environment (string): desired MineRL environment
        data_dir (string, optional): specify alternative dataset location. Defaults to None.
        num_workers (int, optional): number of files to load at once. Defaults to 4.
        force_download (bool, optional): specifies whether or not the data should be downloaded if missing. Defaults to False.

    Returns:
        DataPipeline: initalized data pipeline
    """

    # Ensure path is setup
    if data_dir is None and 'IGLU_DATA_PATH' in os.environ:
        data_dir = os.environ['IGLU_DATA_PATH']
    if data_dir is not None and not os.path.exists(data_dir) or True:
        if force_download:
            print("Provided data directory does not exist: ", data_dir)
            data_dir = download(data_dir)
        else:
            raise FileNotFoundError("Provided data directory does not exist. "
                                    "Specify force_download=True to download default dataset")

    d = IGLUDataPipeline(
        data_dir,
        num_workers,
        worker_batch_size,
        minimum_size_to_dequeue)
    return d
