"""captioner.py – BLIP auto-caption in one call"""
from __future__ import annotations
from functools import lru_cache
from typing import Union

from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)

@lru_cache(maxsize=1)
def _load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=torch.float32
    )
    model.eval()
    return processor, model

def caption(pil_img: Union[str, Image.Image]) -> str:
    """Return a short caption for an RGB PIL image or image-path."""
    if isinstance(pil_img, (str, bytes, bytearray)):
        pil_img = Image.open(pil_img).convert("RGB")

    processor, model = _load_model()
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(max_length=20, **inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
