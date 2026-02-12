import os
import re
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import csv
import pandas as pd
from glob import glob
import shutil

# from text import _clean_text
# import tacotron_cleaner.cleaners
from typeguard import typechecked
from typing import Collection, Optional

try:
    from vietnamese_cleaner import vietnamese_cleaners
except ImportError:
    vietnamese_cleaners = None

try:
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
except (ImportError, SyntaxError):
    BasicTextNormalizer = None

# 对原始数据集中的wav和text处理

_square_brackets_re = re.compile(r"\[[\w\d\s]+\]")
_inv_square_brackets_re = re.compile(r"(.*?)\](.+?)\[(.*)")

class TextCleaner:
    """Text cleaner.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    """

    @typechecked
    def __init__(self, cleaner_types: Optional[Collection[str]] = None):

        if cleaner_types is None:
            self.cleaner_types = []
        elif isinstance(cleaner_types, str):
            self.cleaner_types = [cleaner_types]
        else:
            self.cleaner_types = list(cleaner_types)

        self.whisper_cleaner = None
        if BasicTextNormalizer is not None:
            for t in self.cleaner_types:
                if t == "whisper_en":
                    self.whisper_cleaner = EnglishTextNormalizer()
                elif t == "whisper_basic":
                    self.whisper_cleaner = BasicTextNormalizer()

    def __call__(self, text: str) -> str:
        for t in self.cleaner_types:
            if t == "tacotron":
                text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            elif t == "jaconv":
                text = jaconv.normalize(text)
            elif t == "vietnamese":
                if vietnamese_cleaners is None:
                    raise RuntimeError("Please install underthesea")
                text = vietnamese_cleaners.vietnamese_cleaner(text)
            elif t == "korean_cleaner":
                text = KoreanCleaner.normalize_text(text)
            elif "whisper" in t and self.whisper_cleaner is not None:
                text = self.whisper_cleaner(text)
            else:
                raise RuntimeError(f"Not supported: type={t}")

        return text


def int_zfill(n):
    '''
    n : int wav #.

    Returns
    filled str with 3 digits.
    e.g., 12 -> '012'

    '''
    return str(n).zfill(3)


def get_sorted_items(items):
    # sort by key
    return sorted(items, key=lambda x:x[0])

def add_spaces_around_digits(input_string):
    # 使用正则表达式查找数字和字母之间没有空格的情况
    result = re.sub(r'(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)', r' \g<0> ', input_string)
    return result


def prepare_align():
    audio_dir = "/nfs-04/yuyue/visualtts_datasets/grid/raw_data/speech"
    text_dir = "/nfs-04/yuyue/visualtts_datasets/grid/raw_data/text"
    video_dir = "/nfs-04/yuyue/visualtts_datasets/grid/raw_data/videos"
    out = "/nfs-04/yuyue/visualtts_datasets/grid/preprocessed_data"
    out_dir = "/nfs-04/yuyue/visualtts_datasets/grid/preprocessed_data/speakers"
    video_out_dir = "/nfs-04/yuyue/visualtts_datasets/grid/preprocessed_data/videos_25fps"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(video_out_dir, exist_ok=True)
    
    sampling_rate = 16000
    max_wav_value = 32768.0
    cleaner = TextCleaner("tacotron")

    filelist_fixed = open(f'{out}/filelist.txt', 'w', encoding='utf-8')
    # speaker_list = [f"s{str(i).zfill(2)}" for i in range(1, 35)]
    speaker_list = [f"s{i}" for i in range(1, 35)]

    # for spk in speaker_list:
    for s in range(len(speaker_list)):
        spk = speaker_list[s]
        spk_wav_list = glob(os.path.join(audio_dir, spk, spk, "*.wav"))
        for wav_path in spk_wav_list:
            basename = os.path.basename(wav_path).replace(".wav", "")
            new_basename = f"s{str(s+1).zfill(2)}_{basename}"
            new_spk = f"s{str(s+1).zfill(2)}"

            os.makedirs(os.path.join(out_dir, new_spk), exist_ok=True)
            os.makedirs(os.path.join(video_out_dir, new_spk), exist_ok=True)

            wav, _ = librosa.load(path=wav_path, sr=sampling_rate)
            # wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                filename=os.path.join(out_dir, new_spk, new_basename+".wav"),
                rate=sampling_rate,
                data=wav.astype(np.float32),)

            text_path = os.path.join(text_dir, spk, "align", basename+".align")
            with open(text_path, 'r') as file:
                text = []
                for line in file.readlines():
                    word = line.strip().split(" ")[-1]
                    if word != "sil":
                        text.append(word)
                text = " ".join(text)
                text = cleaner(text)
            lab_path = os.path.join(out_dir, new_spk, new_basename + '.lab')
            with open(lab_path, 'w') as file:
                file.write(text + "\n")

            
            # video_path = os.path.join(video_dir, spk, spk, basename + ".mpg")
            # shutil.copy(video_path, os.path.join(video_out_dir, new_spk, new_basename + '.mpg'))

            # filelist_fixed.write(f"{new_basename}\n")
            
    
    # filelist_fixed.close()

    # Save Speaker Info
    with open(f'{out}/speaker_info.txt', 'w', encoding='utf-8') as f:
        for i in range(len(speaker_list)):
        # for spk in speaker_list:
            f.write(f"s{str(i+1).zfill(2)}\n")


if __name__ == "__main__":
    import yaml
    # # 用espnet的cleaner处理文本，对wav做norm，然后output到speakers_processed
    # config = "/nfs-04/yuyue/visualtts_datasets/lrs2/codes/preprocess.yaml"
    # config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    prepare_align()