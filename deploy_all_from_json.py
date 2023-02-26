import json
import os

if __name__ == "__main__":
    with open("./deployments/deploy_infos.json") as f:
        deploy_infos = json.load(f)

    for model_info in deploy_infos:

        MODEL_ID = model_info["model_id"]
        SAFETY_MODEL_ID = model_info.get(
            "safety_model_id", "CompVis/stable-diffusion-safety-checker"
        )
        IS_FP16 = model_info.get("is_fp16", 1)
        USERNAME = "cloneofsimo"  # TODO : this is obviously not the way to do it, but cannot think of a better way right now
        REPLICATE_MODEL_ID = model_info["name"]

        print(f"Deploying {REPLICATE_MODEL_ID}...")
        print(f"\tMODEL_ID: {MODEL_ID}")
        print(f"\tSAFETY_MODEL_ID: {SAFETY_MODEL_ID}")
        print(f"\tIS_FP16: {IS_FP16}")
        print(f"\tUSERNAME: {USERNAME}")

        # put this to .env
        with open(".env", "w") as f:
            f.write(f"MODEL_ID={MODEL_ID}\n")
            f.write(f"SAFETY_MODEL_ID={SAFETY_MODEL_ID}\n")
            f.write(f"IS_FP16={IS_FP16}\n")

        os.system(f"python script/download-weights.py")
        os.system(f"cog run python test.py --test_img2img --test_text2img")
        os.system(f"cog push r8.im/{USERNAME}/{REPLICATE_MODEL_ID}")
