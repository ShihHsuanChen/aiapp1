from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from . import __version__


def cli():
    parser = ArgumentParser(
        'myaiapp',
        description='Doing image classification using MobileNet',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-v', '--version',
        action='store_true',
        help='show version',
    )
    parser.add_argument(
        '-k', '--topk',
        type=int,
        default=5,
        help='list predict classes of top-k highest probabilities',
    )
    parser.add_argument(
        'image_path',
        nargs='?',
        help='image path or url',
    )
    # parse
    args = parser.parse_args()

    # execute
    if args.version:
        print(__version__)
        return

    if args.image_path is None:
        print('Requires an image path')
        return

    _inference(args)


def _inference(args):
    from .main import inference
    result = inference(args.image_path, topk=args.topk)
    # print result
    for label, prob in result:
        print(label, prob)
