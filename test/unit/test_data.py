import pytest
from PIL import Image

from helpers import get_data_path

import aiapp1.data


@pytest.mark.parametrize(
    'image_file, islocal, expect_success',
    [
        ('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png', False, True),
        ('beignets-task-guide.png', True, True),
        ('cloth.jpg', True, True),
        ('notanimage.csv', True, False),
    ],
    ids=[
        'url_png',
        'local_png',
        'local_jpg',
        'local_csv',
    ],
)
def test_read_image(image_file, islocal, expect_success):
    # prepare
    if islocal:
        image_file = get_data_path(image_file)

    # execute
    if expect_success:
        res = aiapp1.data.read_image(image_file)
        # test
        assert isinstance(res, Image.Image)
    else:
        try:
            res = aiapp1.data.read_image(image_file)
        except:
            pass
        else:
            raise AssertionError('Should not a valid image file')
