from urllib.request import urlopen
from PIL import Image


def read_image(image_path: str) -> Image.Image:
    """
    image_path: (str) path or url (example: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
    """
    if image_path.startswith('http://') or image_path.startswith('https://'):
        image = Image.open(urlopen(image_path))
    else:
        image = Image.open(image_path)
    return image.convert('RGB')
