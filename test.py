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
    lora_urls="https://replicate.delivery/pbxt/f3qw3sRceIuzGUR11igMilMJscSeS32NGQTgBeREwjmMQfQDC/tmpm1lnsk1xelon-musk.safetensors|https://replicate.delivery/pbxt/6CAOif8vSfkSBEQk3YB4qqcs3ZEkOay22pKLEVfe7e9FjehGE/tmp3901xx0nold-disney.safetensors",
    # lora_urls="",
    lora_scales="0.7|0.1",
    prompt="a photo of <1> in style of <2> avatarart style",
    **defaults
)

from PIL import Image

Image.open("/tmp/out-0.png").save("./out.png")
