#!/usr/bin/env python

import os
import shutil
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

import dotenv

dotenv.load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", None)
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = os.environ.get("SAFETY_MODEL_ID", None)
IS_FP16 = os.environ.get("IS_FP16", None)

assert MODEL_ID is not None, "MODEL_ID must be set"
assert SAFETY_MODEL_ID is not None, "SAFETY_MODEL_ID must be set"
assert IS_FP16 is not None, "IS_FP16 must be set"

if os.path.exists(MODEL_CACHE):
    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

torch_dtype = "torch.float16" if IS_FP16 == 1 else "torch.float32"

saftey_checker = StableDiffusionSafetyChecker.from_pretrained(
    SAFETY_MODEL_ID, cache_dir=MODEL_CACHE, torch_dtype=torch_dtype
)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_CACHE,
    torch_dtype=torch_dtype,
)
