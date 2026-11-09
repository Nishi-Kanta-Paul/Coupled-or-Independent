import os

import cv2
import numpy as np
import torch


class CraftScorer:
    def __init__(self, device, model_dir):
        import easyocr
        os.makedirs(model_dir, exist_ok=True)
        reader = easyocr.Reader(['en'], gpu=device.type == 'cuda', model_storage_directory=model_dir,
                                user_network_directory=model_dir, recognizer=False, verbose=False)
        self.net = reader.detector
        self.device = reader.device

    @torch.no_grad()
    def __call__(self, image, canvas_size=0):
        from easyocr import imgproc
        array = np.asarray(image.convert('RGB'))
        canvas = canvas_size if canvas_size and canvas_size > 0 else max(array.shape[:2])
        resized, _, _ = imgproc.resize_aspect_ratio(array, canvas, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0)
        x = torch.from_numpy(imgproc.normalizeMeanVariance(resized)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        y, _ = self.net(x)
        return float(y[0, :, :, 0].max().item())
