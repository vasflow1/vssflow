import csv
from tqdm import tqdm
import torch, torchaudio, torchvision
import numpy as np
import os
import random


def collate_fn_for_text(batch):
    video_ids = [i['video_id'] for i in batch]
    avhubert_length = [i['avhubert_length'] for i in batch]
    phone_length = [i['phone_length'] for i in batch]

    length = max([len(i['phone_id']) for i in batch])
    duration_matrix = torch.stack([torch.nn.functional.pad(i['duration_matrix'], (0, length - i['duration_matrix'].shape[-1]), mode='constant', value=0) for i in batch])
    # duration_span = torch.stack([torch.nn.functional.pad(i['duration_span'], (0, 0, 0, length - i['duration_span'].shape[0]), mode='constant', value=0) for i in batch])
    duration_span = torch.stack([i['duration_span'] for i in batch])

    phone_id = torch.stack([torch.nn.functional.pad(i['phone_id'], (0, length - i['phone_id'].shape[-1]), mode='constant', value=0) for i in batch])
    avhubert = torch.stack([torch.nn.functional.pad(i['avhubert'], (0, 0, 0, 250 - i['avhubert'].shape[0]), mode='constant', value=0) for i in batch])
    
    return {
            "video_id": video_ids,
            "duration_matrix": duration_matrix,
            "duration_span":duration_span,
            "avhubert": avhubert,
            "avhubert_length":avhubert_length,
            "phone_id": phone_id,
            "phone_length": phone_length
    }





class LipPhoneDataset(torch.utils.data.Dataset):
    def __init__(self, meta_dir: str, split: str = 'train', mask_ratio = 0):
        self.metas      = self._load_meta(meta_dir, split)
        self.mask_ratio = mask_ratio


    def _load_meta(self, meta_dir: str, split: str):
        metas = []
        with open(f'{meta_dir}/{split}.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            for row in reader:
                # [duration_matrix, avhubert_feature, phoneme_seq]
                metas.append(row)
        return metas

    def find_continuous_ones_indices(self, tensor):
        if tensor.sum() == 0:
            return torch.zeros(tensor.size(1), 2, dtype=torch.long)
        start_indices = torch.argmax(tensor, dim=0)
        flipped_tensor = torch.flip(tensor, dims=[0])
        end_indices = tensor.size(0) - torch.argmax(flipped_tensor, dim=0)
        result = torch.stack([start_indices, end_indices], dim=1)

        mask = tensor.sum(dim=0) == 0
        result[mask] = 0
        return result
    
    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        duration_matrix_path, avhubert_path, phoneme_seq = self.metas[idx]
        video_id, _ = os.path.splitext(os.path.basename(duration_matrix_path))

        duration_matrix = np.load(duration_matrix_path)
        duration_matrix = torch.from_numpy(duration_matrix).float()
        duration_span = self.find_continuous_ones_indices(duration_matrix.T)


        avhubert = np.load(avhubert_path)
        avhubert = torch.from_numpy(avhubert).float()
        if random.uniform(0, 1) < self.mask_ratio:
            avhubert = torch.ones_like(avhubert).to(avhubert)
            video_id = f"{video_id}_mask"
        avhubert_length = avhubert.shape[0]

        phone_id = phoneme_seq.split(' ')
        phone_id = torch.tensor([int(x) for x in phone_id], dtype=torch.int64)
        phone_length = phone_id.shape[0]

        return {
            "video_id": video_id,
            "duration_matrix": duration_matrix,
            "duration_span": duration_span,
            "avhubert": avhubert,
            "avhubert_length": avhubert_length,
            "phone_id": phone_id,
            "phone_length": phone_length
        }



if __name__ == "__main__":
    dataset = LipPhoneDataset('/home/chengxin/chengxin/Dataset_Sound/MetaData/vaflow2_meta/dp', split='train_25_Chem_GRID')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_for_text)

    for batch in dataloader:
        print(batch['duration_matrix'].shape)  # [bs, avhubert_len, phone_seq_len]
        print(batch['avhubert'].shape)         # [bs, avhubert_len, 1024]
        print(batch['phone_id'].shape)         # [bs, phone_seq_len]
        print(batch['avhubert_length'])         # [bs]
        print(batch['phone_length'])            # [bs]
        break