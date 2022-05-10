import os

def path_type(path_str: str):
    return os.path.normpath(os.path.abspath(path_str))
