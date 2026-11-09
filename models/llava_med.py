import json
import math

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import (AddedToken, AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor, MistralConfig)


LLAVA_MED_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' }}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}{{ '<image>\n' }}{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}"
    "{{ ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' }}"
    "{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}"
    "{{ ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)


def expand_to_square(image, mean):
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    background = tuple(int(round(m * 255)) for m in mean)
    canvas = Image.new(image.mode, (side, side), background)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def map_key(key, targets):
    if key.startswith('model.vision_tower.vision_tower.'):
        rest = key[len('model.vision_tower.vision_tower.'):]
        candidates = ['vision_tower.' + rest, 'model.vision_tower.' + rest]
    elif key.startswith('model.mm_projector.0.'):
        rest = key[len('model.mm_projector.0.'):]
        candidates = ['multi_modal_projector.linear_1.' + rest, 'model.multi_modal_projector.linear_1.' + rest]
    elif key.startswith('model.mm_projector.2.'):
        rest = key[len('model.mm_projector.2.'):]
        candidates = ['multi_modal_projector.linear_2.' + rest, 'model.multi_modal_projector.linear_2.' + rest]
    elif key.startswith('model.'):
        rest = key[len('model.'):]
        candidates = ['language_model.model.' + rest, 'model.language_model.' + rest]
    elif key == 'lm_head.weight':
        candidates = ['language_model.lm_head.weight', 'lm_head.weight']
    else:
        candidates = [key]
    for candidate in candidates:
        if candidate in targets:
            return candidate
    raise KeyError(f'No target for LLaVA-Med parameter {key}')


def build_llava_med(model_id):
    source = json.load(open(hf_hub_download(model_id, 'config.json')))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_tokens(AddedToken('<image>', special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    vision_config = CLIPVisionConfig.from_pretrained(source['mm_vision_tower'])
    padded_vocab = int(math.ceil(len(tokenizer) / 64.0) * 64)
    text_config = MistralConfig(
        vocab_size=padded_vocab, hidden_size=source['hidden_size'], intermediate_size=source['intermediate_size'],
        num_hidden_layers=source['num_hidden_layers'], num_attention_heads=source['num_attention_heads'],
        num_key_value_heads=source['num_key_value_heads'], max_position_embeddings=source['max_position_embeddings'],
        rms_norm_eps=source['rms_norm_eps'], rope_theta=source['rope_theta'], sliding_window=source.get('sliding_window'),
        hidden_act=source['hidden_act'], tie_word_embeddings=False, bos_token_id=source['bos_token_id'],
        eos_token_id=source['eos_token_id'], pad_token_id=tokenizer.pad_token_id)
    strategy = 'default' if source.get('mm_vision_select_feature', 'patch') == 'patch' else 'full'
    config = LlavaConfig(
        vision_config=vision_config, text_config=text_config, image_token_index=tokenizer.convert_tokens_to_ids('<image>'),
        projector_hidden_act='gelu', vision_feature_select_strategy=strategy,
        vision_feature_layer=source['mm_vision_select_layer'], pad_token_id=tokenizer.pad_token_id,
        image_seq_length=(vision_config.image_size // vision_config.patch_size) ** 2)
    return source, tokenizer, config


def load_llava_med(model_id, dtype, device):
    source, tokenizer, config = build_llava_med(model_id)
    with init_empty_weights(include_buffers=False):
        model = LlavaForConditionalGeneration(config)
    targets = set(model.state_dict().keys())
    index = json.load(open(hf_hub_download(model_id, 'model.safetensors.index.json')))['weight_map']
    state = {}
    for shard in sorted(set(index.values())):
        with safe_open(hf_hub_download(model_id, shard), framework='pt') as f:
            for key in f.keys():
                state[map_key(key, targets)] = f.get_tensor(key).to(dtype)
    old_vocab = source['vocab_size']
    new_vocab = config.text_config.vocab_size
    for key in [k for k in state if k.endswith('embed_tokens.weight') or k.endswith('lm_head.weight')]:
        weight = state[key]
        extra = weight.float().mean(dim=0, keepdim=True).expand(new_vocab - old_vocab, -1).to(dtype)
        state[key] = torch.cat([weight, extra], dim=0)
    missing = targets - set(state)
    unexpected = set(state) - targets
    if missing or unexpected:
        raise RuntimeError(f'LLaVA-Med conversion mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}')
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(device).eval()
    image_processor = CLIPImageProcessor.from_pretrained(source['mm_vision_tower'])
    processor = LlavaProcessor(
        image_processor=image_processor, tokenizer=tokenizer, patch_size=config.vision_config.patch_size,
        vision_feature_select_strategy=config.vision_feature_select_strategy, chat_template=LLAVA_MED_TEMPLATE,
        num_additional_image_tokens=1)
    suppress = list(range(old_vocab, new_vocab))
    return model, processor, suppress
