import os
from aiapp1.cli import cli


if __name__ == '__main__':
    os.environ['INFER_MODEL_NAME'] = 'mobilenetv4_conv_small.e2400_r224_in1k'
    cli()
