# pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
from urllib.request import urlopen
from PIL import Image
import torch
import timm
from imagenet_stubs.imagenet_2012_labels import label_to_name

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

with torch.no_grad():
    output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1), k=5)
# batch_size = 1
top5_class_indices = top5_class_indices.cpu().numpy().tolist()[0]
top5_probabilities = top5_probabilities.cpu().numpy().tolist()[0]

for idx, prob in zip(top5_class_indices, top5_probabilities):
    print(label_to_name(idx), prob)
