#!/usr/bin/env python

from predict import Predictor
from inspect import signature
import argparse
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="photo of <1> on the moon village in style of <2>, fantasy house, detailed faces, highres, RAW photo 8k uhd, dslr",
    )
    parser.add_argument("--sd_version", type=str, default="v1")
    parser.add_argument("--test_text2img", action="store_true")
    parser.add_argument("--test_img2img", action="store_true")
    parser.add_argument("--test_adapter", action="store_true")
    
    arg = parser.parse_args()
    
    sd_version = arg.sd_version
    prompt = arg.prompt
    test_text2img = arg.test_text2img
    test_img2img = arg.test_img2img
    test_adapter = arg.test_adapter

    EXAMPLE_LORAS = {
        "v1": "https://replicate.delivery/pbxt/IzbeguwVsW3PcC1gbiLy5SeALwk4sGgWroHagcYIn9I960bQA/tmpjlodd7vazekezip.safetensors|https://raw.githubusercontent.com/cloneofsimo/lora/master/example_loras/lora_popart.safetensors",
        "v2": "",
    }

    EXAMPLE_LORAS_2 = {
        "and": "https://replicate.delivery/pbxt/tLNfiG3fK2jZo0CrBG4cNTJNhEi7r117ANUBjWrLTkQRMraQA/tmpg9tq4is5me.safetensors",
        "v1": "https://raw.githubusercontent.com/cloneofsimo/lora/master/example_loras/lora_krk.safetensors|https://raw.githubusercontent.com/cloneofsimo/lora/master/example_loras/lora_illust.safetensors",
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
    del defaults["seed"]
    
    if test_adapter:
        defaults["adapter_condition_image"] = "./test_data/keypose_0.png"
        defaults["adapter_type"] = "keypose"

    if test_text2img:
        # Test text2img
        out = p.predict(
            lora_urls=EXAMPLE_LORAS[sd_version],
            lora_scales="0.5|0.9",
            prompt=prompt,
            guidance_scale=3.0,
            seed=0,
            **defaults,
        )

        Image.open("/tmp/out-0.png").save("./out-0.png")

        # Also test null lora
        out = p.predict(
            lora_urls="",
            lora_scales="",
            prompt=prompt,
            guidance_scale=3.0,
            seed=0,
            **defaults,
        )

        Image.open("/tmp/out-0.png").save("./out-1.png")

        # Also test reloading lora, with null scale
        out = p.predict(
            lora_urls=EXAMPLE_LORAS[sd_version],
            lora_scales="0.0|0.0",
            prompt=prompt,
            guidance_scale=3.0,
            seed=0,
            **defaults,
        )

        Image.open("/tmp/out-0.png").save("./out-2.png")

        # 2 and 3 should be the same.

        out = p.predict(
            lora_urls=EXAMPLE_LORAS_2[sd_version],
            lora_scales="0.5|0.7",
            prompt=prompt,
            guidance_scale=3.0,
            seed=0,
            **defaults,
        )

        Image.open("/tmp/out-0.png").save("./out-3.png")

        # Test img2img
    if test_img2img:
        del defaults["image"]
        del defaults["prompt_strength"]
        
        out = p.predict(
            lora_urls=EXAMPLE_LORAS_2[sd_version],
            lora_scales="0.5|0.7",
            prompt=prompt,
            guidance_scale=3.0,
            seed=0,
            image="./out-0.png",
            prompt_strength=0.7,
            **defaults,
        )
        
        Image.open("/tmp/out-0.png").save("./out-img2img.png")
        