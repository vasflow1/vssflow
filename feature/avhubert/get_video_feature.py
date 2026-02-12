# import cv2
# import tempfile
import torch
from argparse import Namespace
# import .fairseq
# from .fairseq import checkpoint_utils, options, tasks, utils
import sys
# sys.path.append('/nfs-02/yuyue/visualtts/DSU-AVO-main/fairseq')
# sys.path.append('/nfs-02/yuyue/visualtts/DSU-AVO-main')
from fairseq import checkpoint_utils, options, tasks, utils
import utils as avhubert_utils_
# import fairseq
# import checkpoint_utils, options, tasks
# from ..fairseq import util
# from IPython.display import HTML
import os 
import numpy as np
import glob
import tqdm
import pdb
import cv2
os.chdir('/home/chengxin/chengxin/vssflow/feature/avhubert')

def load_video(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")

def extract_visual_feature(video_path, ckpt_path, user_dir, is_finetune_ckpt=False):
  utils.import_user_module(Namespace(user_dir=user_dir))
  models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  # models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
  transform = avhubert_utils_.Compose([
      avhubert_utils_.Normalize(0.0, 255.0),
      avhubert_utils_.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
      avhubert_utils_.Normalize(task.cfg.image_mean, task.cfg.image_std)])
  # pdb.set_trace()
  # frames = avhubert_utils_.load_video(video_path)
  frames = load_video(video_path)
  
  # print(f"Load video {video_path}: shape {frames.shape}")
  frames = transform(frames)
  # print(f"Center crop video to: {frames.shape}")
  frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
  model = models[0]
  if hasattr(models[0], 'decoder'):
    # print(f"Checkpoint: fine-tuned")
    model = models[0].encoder.w2v_model
  else:
    print(f"Checkpoint: pre-trained w/o fine-tuning")
  model.cuda()
  model.eval()
  with torch.no_grad():
    # Specify output_layer if you want to extract feature of an intermediate layer
    feature, _ = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
    feature = feature.squeeze(dim=0)
  # print(f"Video feature shape: {feature.shape}")
  return feature



if __name__ == "__main__":
  # mouth_roi_source_path = '/nfs-02/yuyue/visualtts/DSU-AVO-main/avhubert/preparation/data/lrs2/mouth'
  mouth_roi_source_path = '/home/chengxin/chengxin/vssflow/feature/avhubert/infer_data2/mouth'
  mouth_roi_paths = sorted(glob.glob(os.path.join(mouth_roi_source_path, '*.mp4')))
  ckpt_path = "/home/chengxin/chengxin/vssflow/feature/avhubert/data/self_large_vox_433h.pt"
  user_dir = "/home/chengxin/chengxin/vssflow/feature/avhubert"
  lip_embedding_path = '/home/chengxin/chengxin/vssflow/feature/avhubert/infer_data2/avhubert_features'
  os.makedirs(lip_embedding_path, exist_ok=True)

  # group = len(mouth_roi_paths) / 20
  # group_number = 18  # 0, 1, 2, 3, 4
  # start = int(group * group_number)
  # end = int(group * (group_number + 1))
  # subset = mouth_roi_paths[start:min(len(mouth_roi_paths), end)]
  subset = mouth_roi_paths
  # subset.reverse()

  for mouth_roi_path in tqdm.tqdm(subset):
    
    save_path = os.path.join(lip_embedding_path, os.path.basename(mouth_roi_path).replace('.mp4', '.npy'))
    if os.path.exists(save_path):
      continue
    import shutil
    # basename = os.path.basename(mouth_roi_path)
    # old_path = mouth_roi_path
    # mouth_roi_path = os.path.join('/nfs-02/yuyue/visualtts/DSU-AVO-main/avhubert/temp_data', basename)
    # shutil.copy(old_path, mouth_roi_path)
    # try:
    feature = extract_visual_feature(mouth_roi_path, ckpt_path, user_dir).cpu().numpy()
    print(save_path)
    np.save(save_path, feature)
    # except:
    #   print(mouth_roi_path)

  # 直接写一个提feature的脚本

  # 现在，首先需要重新提取一遍audio，从video（25fps）中直接提取