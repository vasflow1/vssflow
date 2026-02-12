export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For vae training
"



# # # # # concat lip synch
# # python \
# #     main.py \
# #     --base "./config/vaflow_noise_lip_synch.yaml" \
# #     -f "_concat_lip_synch" \
# #     -t True \
# #     -i False \
# #     --devices '4,5,6,7'

# # # # # concat lip synch video text
# # python \
# #     main.py \
# #     --base "./config/vaflow_noise_lip_synch_video_text.yaml" \
# #     -f "_concat_lip_synch_video_text" \
# #     -t True \
# #     -i False \
# #     --devices '4,5,6,7'


# # # # # concat video text
# # python \
# #     main.py \
# #     --base "./config/vaflow_noise_video_text.yaml" \
# #     -f "_concat_video_text" \
# #     -t True \
# #     -i False \
# #     --devices '4,5,6,7'


# # # # # concat none
# # python \
# #     main.py \
# #     --base "./config/vaflow_noise_none.yaml" \
# #     -f "_concat_none" \
# #     -t True \
# #     -i False \
# #     --devices '4,5,6,7'


# # # # # concat none
# # python \
# #     main.py \
# #     --base "./config/vaflow_noise_lip_synch_text.yaml" \
# #     -f "_concat_lip_synch_text" \
# #     -t True \
# #     -i False \
# #     --devices '7,'


# concat none
python \
    main.py \
    --base "./config/vaflow_noise_lip_synch_text_mixtrain.yaml" \
    -f "_debug_mix" \
    -t True \
    -i False \
    --devices '7,'