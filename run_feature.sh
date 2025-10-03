export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For final infer
"

# Noise


python\
    -m pdb \
    main.py \
    --base "./config/vae_beta.yaml" "./config/clip_infer.yaml" \
    -f "_clip_infer" \
    -t False \
    -i True \
    --devices 1