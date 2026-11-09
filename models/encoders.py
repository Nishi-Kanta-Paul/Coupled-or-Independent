import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ENCODER_REGISTRY = {
    'biomedclip': ('open_clip', 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'),
    'pubmedclip': ('hf_clip', 'flaviagiammarino/pubmed-clip-vit-base-patch32'),
    'clip': ('hf_clip', 'openai/clip-vit-base-patch32'),
}


class ContrastiveEncoder:
    name = ''
    resolution = 224
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    def prepare(self, image):
        image = image.convert('RGB').resize((self.resolution, self.resolution), Image.BICUBIC)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(self.mean, dtype=torch.float32)[:, None, None]
        std = torch.tensor(self.std, dtype=torch.float32)[:, None, None]
        return (tensor - mean) / std

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        features = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.prepare(im) for im in images[i:i + batch_size]]).to(self.device)
            features.append(F.normalize(self.image_features(batch).float(), dim=-1).cpu())
        return torch.cat(features).numpy() if features else np.zeros((0, 1), dtype=np.float32)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=64):
        features = []
        for i in range(0, len(texts), batch_size):
            features.append(F.normalize(self.text_features(list(texts[i:i + batch_size])).float(), dim=-1).cpu())
        return torch.cat(features).numpy()


class HFCLIPEncoder(ContrastiveEncoder):
    def __init__(self, name, model_id, device):
        from transformers import CLIPModel, AutoTokenizer, CLIPImageProcessor
        self.name = name
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = CLIPImageProcessor.from_pretrained(model_id)
        self.mean, self.std = tuple(processor.image_mean), tuple(processor.image_std)
        self.resolution = int(self.model.config.vision_config.image_size)

    def image_features(self, pixels):
        return self.model.get_image_features(pixel_values=pixels)

    def text_features(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        return self.model.get_text_features(**tokens)


class OpenCLIPEncoder(ContrastiveEncoder):
    def __init__(self, name, model_id, device):
        import open_clip
        from torchvision.transforms import Normalize
        self.name = name
        self.device = device
        self.model, preprocess = open_clip.create_model_from_pretrained(model_id)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)
        normalize = [t for t in preprocess.transforms if isinstance(t, Normalize)][0]
        self.mean, self.std = tuple(normalize.mean), tuple(normalize.std)
        size = self.model.visual.image_size
        self.resolution = int(size[0] if isinstance(size, (tuple, list)) else size)

    def image_features(self, pixels):
        return self.model.encode_image(pixels)

    def text_features(self, texts):
        return self.model.encode_text(self.tokenizer(texts, context_length=256).to(self.device))


def build_encoder(name, device):
    kind, model_id = ENCODER_REGISTRY[name]
    if kind == 'open_clip':
        return OpenCLIPEncoder(name, model_id, device)
    return HFCLIPEncoder(name, model_id, device)
