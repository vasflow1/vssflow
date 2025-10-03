export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For vae training
"



python \
    main.py \
    --base "./config/dp_tts.yaml" \
    -f "_Lattn" \
    -t True \
    -i False \
    --devices '4,5,6,7'