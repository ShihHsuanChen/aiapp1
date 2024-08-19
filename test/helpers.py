import os


DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def get_data_path(path):
    return os.path.join(DATADIR, path)
