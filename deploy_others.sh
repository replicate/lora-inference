export MODEL_ID="lambdalabs/dreambooth-avatar" # change this
export SAFETY_MODEL_ID="CompVis/stable-diffusion-safety-checker"
export IS_FP16=1
export USERNAME="cloneofsimo" # change this
export REPLICATE_MODEL_ID="avatar"

echo "MODEL_ID=$MODEL_ID" > .env
echo "SAFETY_MODEL_ID=$SAFETY_MODEL_ID" >> .env
echo "IS_FP16=$IS_FP16" >> .env

#cog run script/download-weights.py
cog run test.py
cog push r8.im/$USERNAME/$REPLICATE_MODEL_ID