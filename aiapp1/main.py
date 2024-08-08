# pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
from .model import load_model
from .data import read_image
from .configs import cfg


# configs: .env .ini .cfg <-- pydantic
model = load_model(cfg.infer_model_name)


def inference(image_path: str, topk: int = 5):
    img = read_image(image_path)

    # runtime interface: inputs -> (inference) -> output
    result = model.inference(img, topk=topk)
    return result
