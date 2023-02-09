# LoRA Replicate inference model

Run inference on Replicate:

[![Replicate](https://replicate.com/replicate/lora/badge)](https://replicate.com/replicate/lora)

Training models:

- Easy-to-use model pre-configured for faces, objects, and styles: [![Replicate](https://replicate.com/replicate/lora-training/badge)](https://replicate.com/replicate/lora-training)
- Advanced model with all the parameters: [![Replicate](https://replicate.com/replicate/lora-advanced-training/badge)](https://replicate.com/replicate/lora-advanced-training)

If you have questions or ideas, please join the `#lora` channel in the [Replicate Discord](https://discord.gg/replicate).

## Deployments

You can deploy any models at huggingface or ones you trained yourself. You can add LoRA with these models

### 1. Manual deployment

We have a default SD1.5 deployed at [replicate](https://github.com/replicate/lora-inference), so you can run your own in a scalable manner. If you would like to launch your own model, run

```
cog run script/download-weights.py
```

to download the weights and place them in the cache directory. This will save base model that will get mounted to the cog container.

Either push the model to [replicate](https://replicate.com/) (follow these instructions for pushing model to replicate) or run

```
cog predict -i prompt="monkey scuba diving"

```

to run locally.

### 2. Deploy & Push to replicate with bash script

First, make a model at replicate.com. Create one [here](https://replicate.com/create)

Specify the following parameter file at `deploy_others.sh` file.

```bash
export MODEL_ID="lambdalabs/dreambooth-avatar" # change this to model at huggingface or your local repository.
export SAFETY_MODEL_ID="CompVis/stable-diffusion-safety-checker"
export IS_FP16=1
export USERNAME="cloneofsimo" # change this to your replicate ID.
export REPLICATE_MODEL_ID="avatar" #replciate model ID,
```

Run it with

```bash
bash deploy_others.sh
```
