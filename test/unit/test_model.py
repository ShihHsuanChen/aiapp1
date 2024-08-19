import pytest

from PIL import Image
from helpers import get_data_path

import aiapp1.model


@pytest.fixture()
def get_image():
    return Image.open(get_data_path('beignets-task-guide.png'))


@pytest.mark.parametrize(
    'model_name, islocal',
    [
        ('mobilenetv4_conv_small.e2400_r224_in1k', False),
        ('./local_mobilenetv4_conv_small.e2400_r224_in1k', True), # should FAIL
    ],
    ids=[
        'huggingface',
        'local',
    ],
)
@pytest.mark.parametrize(
    'topk',
    [5, 10],
)
def test_local_model(model_name, islocal, get_image, topk):
    # prepare
    if islocal:
        model_name = get_data_path(model_name)
    print(model_name)
    image = get_image

    # execute load model
    model = aiapp1.model.load_model(model_name)

    # test
    assert isinstance(model, aiapp1.model.InferModel)

    # execute inference
    res = model.inference(image, topk=topk)

    # test
    assert isinstance(res, list)
    assert len(res) == topk

    for v in res:
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert isinstance(v[0], str)
        assert isinstance(v[1], float)
