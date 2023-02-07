from hashlib import sha512
import os
from typing import List
import time
import requests

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPFeatureExtractor


from lora_diffusion import patch_pipe, tune_lora_scale, set_lora_diag

from safetensors.torch import safe_open, save_file

import dotenv
import os

dotenv.load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", None)
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = os.environ.get("SAFETY_MODEL_ID", None)


def lora_join(lora_safetenors: list):
    metadatas = [dict(safelora.metadata()) for safelora in lora_safetenors]
    total_metadata = {}
    total_tensor = {}
    total_rank = 0
    ranklist = []
    for _metadata in metadatas:
        rankset = []
        for k, v in _metadata.items():
            if k.endswith("rank"):
                rankset.append(int(v))

        assert len(set(rankset)) == 1, "Rank should be the same per model"
        total_rank += rankset[0]
        total_metadata.update(_metadata)
        ranklist.append(rankset[0])

    tensorkeys = set()
    for safelora in lora_safetenors:
        tensorkeys.update(safelora.keys())

    for keys in tensorkeys:
        if keys.startswith("text_encoder") or keys.startswith("unet"):
            tensorset = [safelora.get_tensor(keys) for safelora in lora_safetenors]

            is_down = keys.endswith("down")

            if is_down:
                _tensor = torch.cat(tensorset, dim=0)
                assert _tensor.shape[0] == total_rank
            else:
                _tensor = torch.cat(tensorset, dim=1)
                assert _tensor.shape[1] == total_rank

            total_tensor[keys] = _tensor
            keys_rank = ":".join(keys.split(":")[:-1]) + ":rank"
            total_metadata[keys_rank] = str(total_rank)
    token_size_list = []
    for idx, safelora in enumerate(lora_safetenors):
        tokens = [k for k, v in safelora.metadata().items() if v == "<embed>"]
        for jdx, token in enumerate(sorted(tokens)):
            if total_metadata.get(token, None) is not None:
                del total_metadata[token]
            total_tensor[f"<s{idx}-{jdx}>"] = safelora.get_tensor(token)
            total_metadata[f"<s{idx}-{jdx}>"] = "<embed>"
            print(f"Embedding {token} replaced to <s{idx}-{jdx}>")

        token_size_list.append(len(tokens))

    return total_tensor, total_metadata, ranklist, token_size_list


def url_local_fn(url):
    return sha512(url.encode()).hexdigest() + ".safetensors"


def download_lora(url):
    # TODO: allow-list of domains

    fn = url_local_fn(url)

    if not os.path.exists(fn):
        print("Downloading LoRA model... from", url)
        # stream chunks of the file to disk
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    else:
        print("Using disk cache...")

    return fn


def lora_add(merged_fn, path_1, alpha_1, path_2, alpha_2):
    """Scales each lora by appropriate weights & returns"""
    start_time = time.time()
    safeloras_1 = safe_open(path_1, framework="pt", device="cpu")
    safeloras_2 = safe_open(path_2, framework="pt", device="cpu")

    metadata = dict(safeloras_1.metadata())
    metadata.update(dict(safeloras_2.metadata()))

    ret_tensor = {}

    for keys in set(list(safeloras_1.keys()) + list(safeloras_2.keys())):
        if keys.startswith("text_encoder") or keys.startswith("unet"):

            tens1 = safeloras_1.get_tensor(keys)
            tens2 = safeloras_2.get_tensor(keys)

            tens = alpha_1 * tens1 + alpha_2 * tens2
            ret_tensor[keys] = tens
        else:
            if keys in safeloras_1.keys():

                tens1 = safeloras_1.get_tensor(keys)
            else:
                tens1 = safeloras_2.get_tensor(keys)

            ret_tensor[keys] = tens1

    print(f"merge time: {time.time() - start_time}")

    # we don't need to go to-> from safetensors here, adding in now for compat's sake
    start_time = time.time()
    save_file(ret_tensor, merged_fn, metadata)
    print(f"saving time: {time.time() - start_time}")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        feature_extractor = CLIPFeatureExtractor.from_json_file(
            f"{MODEL_CACHE}/feature_extractor/preprocessor_config.json"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        self.loaded = None

    def merge_loras(self, url_1, scale_1, url_2, scale_2):
        merged_fn = url_local_fn(f"{url_1}-{url_2}-{scale_1}-{scale_2}")

        if self.loaded == merged_fn:
            print("The requested two LoRAs are already scaled and loaded.")
            return

        lora_1 = download_lora(url_1)
        lora_2 = download_lora(url_2)

        st = time.time()
        lora_add(merged_fn, lora_1, scale_1, lora_2, scale_2)
        print(f"merging time: {time.time() - st}")

        patch_pipe(self.pipe, merged_fn)
        # merging tunes lora scale so we don't need to do that here.
        self.loaded = merged_fn

    def join_many_lora(self, urllists: List[str], scales: List[float]):
        assert len(urllists) == len(scales), "Number of LoRAs and scales must match."

        merged_fn = url_local_fn(f"{'-'.join(urllists)}")

        if self.loaded == merged_fn:
            print("The requested LoRAs are loaded.")

        else:
            lora_safetenors = [
                safe_open(download_lora(url), framework="pt", device="cpu")
                for url in urllists
            ]
            st = time.time()

            tensors, metadata, ranklist, token_size_list = lora_join(lora_safetenors)
            save_file(tensors, merged_fn, metadata)

            print(f"merging time: {time.time() - st}")

            patch_pipe(self.pipe, merged_fn)
            self.loaded = merged_fn
            self.token_size_list = token_size_list
            self.ranklist = ranklist

        diags = []
        for scale, rank in zip(scales, self.ranklist):
            diags = diags + [scale] * rank

        set_lora_diag(self.pipe.unet, torch.tensor(diags))

    def load_lora(self, url, scale):
        if url == self.loaded:
            print("The requested LoRA model is already loaded...")
            return

        start_time = time.time()
        local_lora_safetensors = download_lora(url)
        print("download_lora time:", time.time() - start_time)

        start_time = time.time()
        patch_pipe(self.pipe, local_lora_safetensors)
        print("patch_pipe time:", time.time() - start_time)

        start_time = time.time()
        tune_lora_scale(self.pipe.unet, scale)
        tune_lora_scale(self.pipe.text_encoder, scale)
        print("tune_lora_scale time:", time.time() - start_time)

        self.loaded = url

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt. Use <1>, <2>, <3>, etc., to specify LoRA concepts",
            default="a photo of <1> riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        lora_urls: str = Input(
            description="List of urls for safetensors of lora models, seperated with | . If provided, it will override all above options.",
            default="",
        ),
        lora_scales: str = Input(
            description="List of scales for safetensors of lora models, seperated with | ",
            default="0.3",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        lora_urls = [u.strip() for u in lora_urls.split("|")]
        lora_scales = [float(s.strip()) for s in lora_scales.split("|")]
        self.join_many_lora(lora_urls, lora_scales)
        if prompt is not None:
            for idx, tok_size in enumerate(self.token_size_list):
                prompt = prompt.replace(
                    f"<{idx + 1}>",
                    "".join([f"<s{idx}-{jdx}>" for jdx in range(tok_size)]),
                )

        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
