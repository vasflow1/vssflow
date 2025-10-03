
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For final infer
"



python \
    main.py \
    --base "./config/vaflow_sda_dit_noise_text_mel.yaml" "./config/vaflow_sda_dit_noise_text_mel_infer.yaml" \
    -f "_infer_chem" \
    -t False \
    -i True \
    --devices 4,5,6,7 \


#     --base "./config/vaflow_sda_dit_noise_text_mel.yaml" "./config/vaflow_sda_dit_noise_text_mel_infer.yaml" \
#     --base "./config/vaflow_sda_dit_noise_text_clip_mel.yaml" "./config/vaflow_sda_dit_noise_text_clip_mel_infer.yaml" \


# Noise

# python main.py \
#     --base "./config/vaflow_sda_dit_noise.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_noise_e74_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 0,1,2,3 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_18-23_57_33-vaflow_sda_dit_noise/ckpt/epoch=0074-step=1.71e+05.ckpt" 



# python main.py \
#     --base "./config/vaflow_sda_dit_noise.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_noise_e94_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 0,1,2,3 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_18-23_57_33-vaflow_sda_dit_noise/ckpt/epoch=0094-step=2.17e+05.ckpt" 


# Full raw


# python main.py \
#     --base "./config/vaflow_sda_dit_full.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_fullraw_jt_e49_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 2,3 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_08-11_17_40-vaflow_sda_dit_joint_tune_vae/ckpt/epoch=0049-step=1.21e+05.ckpt" 


# python main.py \
#     --base "./config/vaflow_sda_dit_full.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_fullraw_e74_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_05-20_05_42-vaflow_sda_dit/ckpt/epoch=0074-step=1.82e+05.ckpt" 


# python main.py \
#     --base "./config/vaflow_sda_dit_full.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_fullaligned_jt_e49_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 6,7 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_03_03-00_56_14-vaflow_sda_dit_full_aligned_joint_tune_vae/ckpt/epoch=0049-step=1.43e+05.ckpt" 
