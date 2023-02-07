import sys

sys.path.append(".")
from predict import Predictor
from inspect import signature

p = Predictor()
p.setup()
sig = signature(Predictor.predict)

defaults = {}
for param_name, param in sig.parameters.items():
    if param.default != param.empty and param.default.default != Ellipsis:
        defaults[param_name] = param.default.default

print(defaults)
del defaults["prompt"]
del defaults["lora_urls"]
del defaults["lora_scales"]

out = p.predict(
    lora_urls="https://storage.googleapis.com/replicant-misc/lora/bfirsh-2.safetensors|https://storage.googleapis.com/replicant-misc/lora/lora_illust.safetensors",
    lora_scales="0.7|0.1",
    prompt="a photo of <1> in style of <2>",
    **defaults
)
