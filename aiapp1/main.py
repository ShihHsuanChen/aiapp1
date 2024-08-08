# pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
from .model import load_model
from .data import read_image
from .configs import cfg

# cli argument: myaiapp ./beignets-task-guide.png
img = read_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png')

# configs: .env .ini .cfg <-- pydantic
model = load_model(cfg.infer_model_name)

# runtime interface: inputs -> (inference) -> output
result = model.inference(img, topk=5)

# print
for label, prob in result:
    print(label, prob)
