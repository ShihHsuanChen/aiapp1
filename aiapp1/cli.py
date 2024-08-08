from . import __version__


def cli():
    print('hello')
    print('version:', __version__)
    from . import main
