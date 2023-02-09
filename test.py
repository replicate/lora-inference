#!/usr/bin/env python

from predict import Predictor
from inspect import signature
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="photo of <1> on the moon village, fantasy house, detailed faces, highres, RAW photo 8k uhd, dslr",
    )
    parser.add_argument("--sd_version", type=str, default="v1")

    sd_version = parser.parse_args().sd_version
    prompt = parser.parse_args().prompt

    EXAMPLE_LORAS = {
        "v1": "https://replicate.delivery/pbxt/IzbeguwVsW3PcC1gbiLy5SeALwk4sGgWroHagcYIn9I960bQA/tmpjlodd7vazekezip.safetensors",
        "v2": "",
        "andreas": "https://replicate.delivery/pbxt/tLNfiG3fK2jZo0CrBG4cNTJNhEi7r117ANUBjWrLTkQRMraQA/tmpg9tq4is5me.safetensors"
        # "krk" :
    }

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
    del defaults["guidance_scale"]

    out = p.predict(
        lora_urls=EXAMPLE_LORAS[sd_version],
        lora_scales="0.5",
        prompt=prompt,
        guidance_scale=3.0,
        **defaults,
    )

    from PIL import Image

    Image.open("/tmp/out-0.png").save("./out.png")
