import os
import zipfile
import requests
import hashlib
import pandas as pd


def MovieLens100K(root: str):
    """MovieLens100K Dataset

    For additional details refer to https://grouplens.org/datasets/movielens/

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `valid`, `test`)

    :returns: DataPipe that yields text from Wikipedia articles
    :rtype: str
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    md5 = "0e33842e24a9c977be4e0107933c0723"
    path = os.path.join("ml-100k", "u.data")
    delimiter = "\t"
    fields = ["user", "item", "rating", "ts"]
    return load_csv(root, url, md5, path , delimiter, fields)


def MovieLens1M(root: str):
    """MovieLens1M Dataset

    For additional details refer to https://grouplens.org/datasets/movielens/

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `valid`, `test`)

    :returns: DataPipe that yields text from Wikipedia articles
    :rtype: str
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    md5 = "c4d9eecfca2ab87c1945afe126590906"
    path = os.path.join("ml-1m", "ratings.dat")
    delimiter = "::"
    fields = ["user", "item", "rating", "ts"]
    return load_csv(root, url, md5, path , delimiter, fields)


def load_csv(root, url, md5, path , delimiter, fields):

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    zip_path = os.path.join(root, os.path.basename(url))
    if not os.path.exists(zip_path):
        download(url, md5, zip_path)

    csv_path = os.path.join(root, path)
    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(root)

    return pd.read_csv(csv_path, delimiter=delimiter, names=fields, engine="python")


def download(url, md5, path):
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        sig = hashlib.md5()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                sig.update(chunk)
        assert(sig.hexdigest() == md5)
