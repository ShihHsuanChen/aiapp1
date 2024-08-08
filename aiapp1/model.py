from typing import List, Tuple

import torch
import timm
from PIL import Image
from imagenet_stubs.imagenet_2012_labels import label_to_name


class InferModel:
    def __init__(self, model_name: str):
        model = timm.create_model(model_name, pretrained=True)
        self.model = model.eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def inference(self,
            image: Image.Image,
            topk: int = 5,
            ) -> List[Tuple[str, float]]:
        """
        Do model inference

        Arguments:
            - image: (PIL.Image.Image)
            - topk: (int) filter top-k results (default: 5)
        Return:
            - result: (List[Tuple]) [(<label>, <probability>)]
        """
        with torch.no_grad():
            output = self.model(self.transforms(image).unsqueeze(0))  # unsqueeze single image into batch of 1
        topk_probabilities, topk_class_indices = torch.topk(output.softmax(dim=1), k=topk)
        # batch_size = 1
        topk_class_indices = topk_class_indices.cpu().numpy().tolist()[0]
        topk_probabilities = topk_probabilities.cpu().numpy().tolist()[0]

        result = [(label_to_name(idx), prob) for idx, prob in zip(topk_class_indices, topk_probabilities)]
        return result


def load_model(model_name: str) -> InferModel:
    return InferModel(model_name)
