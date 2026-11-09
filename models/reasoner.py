import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from util.utils import set_seed
from .llava_med import load_llava_med, expand_to_square


class VisionReasoner:
    def __init__(self, model_id, device, dtype, max_new_tokens, temperature, top_p, image_max_side=0):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.image_max_side = image_max_side
        self.is_llava_med = 'llava-med' in model_id.lower()
        if self.is_llava_med:
            self.model, self.processor, self.suppress = load_llava_med(model_id, dtype, device)
        else:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
            self.suppress = None
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

    def prepare_image(self, image):
        image = image.convert('RGB')
        if self.image_max_side and max(image.size) > self.image_max_side:
            image = image.copy()
            image.thumbnail((self.image_max_side, self.image_max_side), Image.BICUBIC)
        if self.is_llava_med:
            image = expand_to_square(image, self.processor.image_processor.image_mean)
        return image

    def messages(self, turns):
        out = []
        for i, (role, text) in enumerate(turns):
            content = [{'type': 'text', 'text': text}]
            if i == 0:
                content = [{'type': 'image'}] + content
            out.append({'role': role, 'content': content})
        return out

    @torch.no_grad()
    def generate(self, turns, image, seed, stop_strings=None):
        prompt = self.processor.apply_chat_template(self.messages(turns), add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors='pt').to(self.device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(self.dtype)
        set_seed(seed)
        kwargs = {'do_sample': True, 'temperature': self.temperature, 'top_p': self.top_p, 'top_k': 0,
                  'repetition_penalty': 1.0, 'max_new_tokens': self.max_new_tokens, 'pad_token_id': self.pad_token_id}
        if stop_strings:
            kwargs.update({'stop_strings': stop_strings, 'tokenizer': self.tokenizer})
        if self.suppress:
            kwargs['suppress_tokens'] = self.suppress
        output = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
