# pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
from .model import load_model
from .data import read_image

# cli argument: myaiapp ./beignets-task-guide.png
img = read_image('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png')

# configs: .env .ini .cfg <-- pydantic
model = load_model('mobilenetv4_conv_small.e2400_r224_in1k')

# runtime interface: inputs -> (inference) -> output
result = model.inference(img, topk=5)

# print
for label, prob in result:
    print(label, prob)
