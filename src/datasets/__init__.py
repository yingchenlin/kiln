from .image import *
from .text import *
from .event import *


def get_dataset(config):
    type_ = config["type"]
    if type_ == "image":
        return ImageDataset(config)
    if type_ == "text":
        return TextDataset(config)
    if type_ == "event":
        return EventDataset(config)
    raise Exception(f"unknown dataset type '{type_}'")

