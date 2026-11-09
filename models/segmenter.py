import numpy as np
import torch
from transformers import SamModel, SamProcessor


class LesionSegmenter:
    def __init__(self, model_id, device):
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def __call__(self, image):
        image = image.convert('RGB')
        width, height = image.size
        inputs = self.processor(image, input_points=[[[width / 2.0, height / 2.0]]], return_tensors='pt').to(self.device)
        outputs = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0][0]
        scores = outputs.iou_scores.cpu()[0, 0]
        best = int(torch.argmax(scores))
        return masks[best].numpy().astype(bool), float(scores[best])
