export MODEL_ID="SG161222/Realistic_Vision_V1.3" # change this
export SAFETY_MODEL_ID="CompVis/stable-diffusion-safety-checker"
export IS_FP16=1
export USERNAME="cloneofsimo" # change this
export REPLICATE_MODEL_ID="realistic_vision_v1.3" # change this

echo "MODEL_ID=$MODEL_ID" > .env
echo "SAFETY_MODEL_ID=$SAFETY_MODEL_ID" >> .env
echo "IS_FP16=$IS_FP16" >> .env

python script/download-weights.py
cog run python test.py --test_img2img --test_text2img --test_adapter
cog push r8.im/$USERNAME/$REPLICATE_MODEL_ID