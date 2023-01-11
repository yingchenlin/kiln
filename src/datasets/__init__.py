from .image import *
#from .text import *
from .event import *


def get_dataset(config, path):
    type_ = config["type"]
    if type_ == "image":
        return ImageDataset(config, path)
    #if type_ == "text":
    #    return TextDataset(config, path)
    if type_ == "event":
        return EventDataset(config, path)
    raise Exception(f"unknown dataset type '{type_}'")

