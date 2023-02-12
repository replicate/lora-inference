#!/usr/bin/env python
import torch
import os
import shutil
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPFeatureExtractor


import dotenv

dotenv.load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", None)
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = os.environ.get("SAFETY_MODEL_ID", None)
IS_FP16 = 1

assert MODEL_ID is not None, "MODEL_ID must be set"
assert SAFETY_MODEL_ID is not None, "SAFETY_MODEL_ID must be set"
assert IS_FP16 is not None, "IS_FP16 must be set"

if os.path.exists(MODEL_CACHE):
    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

torch_dtype = torch.float16 if IS_FP16 == 1 else torch.float32

safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    SAFETY_MODEL_ID, torch_dtype=torch_dtype
)

feature_extractor = CLIPFeatureExtractor.from_dict(
    {
        "crop_size": {"height": 224, "width": 224},
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "feature_extractor_type": "CLIPFeatureExtractor",
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "CLIPFeatureExtractor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": 224},
    }
)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,
    torch_dtype=torch_dtype,
)

pipe.save_pretrained(MODEL_CACHE)
