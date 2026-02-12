import os, sys
import random
import json

import tgt
import librosa
import numpy as np
from tqdm import tqdm
import glob
import yaml
from scipy.io import wavfile
import ffmpeg
import subprocess

os.chdir(sys.path[0])
random.seed(1234)

class Preprocessor:
    def __init__(self, config):
        # input paths
        self.config = config
        self.dataset = config["dataset"]
        self.root_path = config["path"]["root_path"]
        self.preprocess_path = os.path.join(self.root_path, config["path"]["preprocessed_sub"])
        self.output_path = os.path.join(config["path"]["output_path"])
        os.makedirs(self.output_path, exist_ok=True)

        # input paths
        self.wav_path = os.path.join(self.root_path, config["path"]["preprocessed_sub"], config["path"]["wav_sub"])
        # self.av_path = os.path.join(self.root_path, config["path"]["av_sub"])
        self.av_path = "/nfs-04/yuyue/visualtts_datasets/grid/avhubert_features/av_feature"
        self.textgrid_path = os.path.join(self.preprocess_path, config["path"]["textgrid_sub"])
        self.video_stamp_path = os.path.join(self.preprocess_path, "video_stamps")
        
        # data_splits
        self.use_data_splits = config["preprocessing"]["use_data_splits"]
        if self.use_data_splits:
            self.data_splits_path = os.path.join(self.root_path, config["path"]["splits_sub"])

        # output paths
        self.output_data_split_path = os.path.join(self.output_path, "data_split")
        self.duration_path = os.path.join(self.output_path, config["path"]["preprocessed_sub"], config["path"]["duration_sub"])
        self.lip_path = os.path.join(self.output_path, config["path"]["preprocessed_sub"], config["path"]["lip_sub"])
        os.makedirs(self.output_data_split_path, exist_ok=True)
        os.makedirs(self.lip_path, exist_ok=True)
        os.makedirs(self.duration_path, exist_ok=True)
        
        # self.gt_code_path = os.path.join(self.output_path, config["path"]["gt_sub"])

        # speakers
        self.speakers = dict()
        self.speakers = self.load_speaker_dict()

        # espnet phonemes check
        self.check_list = self.load_phone_set("./espnet_g2p_phonemes")
        self.phone_set = set()
        
        # parameters
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.frame_rate = config["preprocessing"]["video"]["frame_rate"]
        self.trim = config["preprocessing"]["trim"]
        if self.trim:
            self.trim_wav_path = os.path.join(self.output_path, config["path"]["preprocessed_sub"], config["path"]["trim_wav_sub"])
            self.trim_video_path = os.path.join(self.output_path, config["path"]["preprocessed_sub"], config["path"]["trim_video_sub"])
            os.makedirs(self.trim_wav_path, exist_ok=True)
            os.makedirs(self.trim_video_path, exist_ok=True)

    def load_speaker_dict(self):
        # spk_dir = os.path.join(self.config["path"]["raw_path"], 'speaker_info.txt')
        spk_path = os.path.join(self.root_path, self.config["path"]["preprocessed_sub"], 'speaker_info.txt')
        spk_dict = dict()
        with open(spk_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                spk_id = line.strip().split("|")[0]
                spk_dict[spk_id] = i
        return spk_dict
    
    def load_phone_set(self, phone_set_path):
        phone_set = []
        with open(phone_set_path, 'r') as file:
            for line in file.readlines():
                phone_set.append(line.strip().replace("g2p_", ""))
        phone_set.append("")
        return phone_set

    def build_from_path(self):
        print("Processing Data ...")

        speakers = self.speakers.copy()

        # data_splits = os.listdir(self.data_splits_path)
        data_splits = ['train.txt', 'val.txt', 'test.txt']  #  
        for data_split in data_splits:
            phonemes, durations = list(), list()
            with open(os.path.join(self.data_splits_path, data_split), 'r') as file:
                data_info = file.readlines()
            for line in tqdm(data_info):
                basename = line.strip().split("|")[0]
                spk = basename.rsplit('_', 1)[0]
                tg_path = os.path.join(self.textgrid_path, spk, f"{basename}.TextGrid")
                if os.path.exists(tg_path):
                    try:
                        text, duration = self.process_utterance(basename, spk)  # 这里返回的text其实是一个phoneme序列
                    except:
                        print(basename)
                    if text and duration:  # 如果不缺失信息，并且每条数据都正常处理，那么就返回处理后的数据，否则返回None
                        phonemes.append(text)
                        durations.append(duration)
                    else:
                        continue
            
            # random.shuffle(out)
            os.makedirs(os.path.join(self.output_path, data_split.split('.')[0]), exist_ok=True)
            with open(os.path.join(self.output_path, data_split.split('.')[0], "phoneme"), 'w') as file:
                for m in phonemes:
                    file.write(m + '\n')
            with open(os.path.join(self.output_path, data_split.split('.')[0], "duration"), 'w') as file:
                for m in durations:
                    file.write(m + '\n')

        # Save files
        with open(os.path.join(self.output_path, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))
        with open(os.path.join(self.output_path, "token_list"), 'w') as file:
            for item in self.phone_set:
                file.write(f"g2p_{item}\n")

        return

    def process_utterance(self, basename, speaker):
        wav_path = os.path.join(self.wav_path, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.wav_path, speaker, "{}.lab".format(basename))
        lip_path = os.path.join(self.av_path, "{}.npy".format(basename.replace("chem_00_", "")))
        tg_path = os.path.join(self.textgrid_path, speaker, "{}.TextGrid".format(basename))
        video_path = os.path.join(self.root_path, "video_25fps", f"{basename}.mp4")

        try:
            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True) 

            if self.trim:
                phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))
                assert len(phone) == len(duration)
                text = "{" + " ".join(phone) + "}"
                if start >= end:
                    return None

                # Read and trim wav files
                wav, sr = librosa.load(wav_path, sr=self.sampling_rate)
                if self.sampling_rate != sr:
                    wav = librosa.resample(wav, sr, self.sampling_rate)
                wav = wav[int(self.sampling_rate * start) : int(self.sampling_rate * end)].astype(np.float32)   # 取从第star秒到end秒的音频
                
                start_idx = int(self.frame_rate * start)
                end_idx = int(np.ceil(self.frame_rate * end))
                lip = np.load(lip_path)
                lip = lip[start_idx : min(end_idx, lip.shape[0]), :].astype(np.float32)

                trim_video_filename = f"{basename}.mp4"
                self.cut_video(video_path, start, end, os.path.join(self.trim_video_path, trim_video_filename))
                
            else:
                phone, duration, start, end = self.get_alignment_no_trim(textgrid.get_tier_by_name("phones"), basename)
                assert len(phone) == len(duration)
                if start >= end:
                    return None

                # Read wav files
                wav, sr = librosa.load(wav_path, sr=self.sampling_rate)
                # wav, sr - librosa.load(wa)
                if self.sampling_rate != sr:
                    wav = librosa.resample(wav, sr, self.sampling_rate)

                # trim tailing visual embedding
                duration_sum = int(np.ceil(sum(duration) / 2))

                video_stamp_path = os.path.join(self.video_stamp_path, speaker, basename+'.npy')
                video_start, video_end = np.load(video_stamp_path)
                frame_num = video_end - video_start
                # end_idx = int(np.ceil(self.frame_rate * end))  # 秒数✖️
                lip = np.load(lip_path)
                # if basename == "s01_bbwt1a":
                #     print(duration_sum)
                if duration_sum - frame_num >5:
                    exit(f"av_feature from {basename} may be invalid")
                lip = lip[video_start: video_start + duration_sum, :].astype(np.float32)
                # lip = lip[video_start: video_start + min(max(end_idx, duration_sum), lip.shape[0]), :].astype(np.float32)

            # Read raw text
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")

            # Save files
            # dur_filename = "{}-duration-{}.npy".format(speaker, basename)
            dur_filename = f"{basename}.npy"
            np.save(os.path.join(self.duration_path, dur_filename), duration)
            # lip_filename = "{}-lip-{}.npy".format(speaker, basename)
            lip_filename = f"{basename}.npy"
            np.save(os.path.join(self.lip_path, lip_filename), lip) # TODO: change lip to sync
            if self.trim:
                trim_wav_filename = f"{basename}.wav"
                wavfile.write(os.path.join(self.trim_wav_path, trim_wav_filename),
                              self.sampling_rate,
                              wav.astype(np.int16))

            text = basename + " " + " ".join(phone)  # 写入
            duration = basename + " " + " ".join(str(i) for i in duration)

            return text, duration
            
            return ("|".join([basename, speaker, text, raw_text]))  # text就是phoneme，raw_text是原始的文本。raw_text大小写都无所谓，text对了就行
        except Exception as e:
            print(e)
            return None

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]
        # sil: silence; sp:  spn: unkown words

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if p == "":
                p = "sil"

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def get_alignment_no_trim(self, tier, basename):

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        for t in tier._objects:  # 每个音素开始结束时间和content
            s, e, p = t.start_time, t.end_time, t.text
            if p not in self.check_list:
                exit(f"phone {p} from {basename} not in check list")
            if p == "":
                p = "<blank>"
            self.phone_set.add(p)
            phones.append(p)
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )
            end_time = e

        return phones, durations, start_time, end_time
    
    def get_alignment_no_trim_for_DSU(self, tier, basename):

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        for t in tier._objects:  # 每个音素开始结束时间和content
            s, e, p = t.start_time, t.end_time, t.text
            if p not in self.check_list:
                exit(f"phone {p} from {basename} not in check list")
            if p == "":
                p = "sil"
            self.phone_set.add(p)
            phones.append(p)
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )
            end_time = e

        return phones, durations, start_time, end_time
    
    
    def cut_video(self, input_file, start_time, end_time, output_file):
        # 构建 FFmpeg 命令
        command = [
            'ffmpeg',
            '-i', input_file,      # 输入文件
            '-ss', f"{start_time}",     # 开始时间
            '-to', f"{end_time}",       # 结束时间
            '-loglevel', 'quiet',
            output_file            # 输出文件
        ]

        try:
            # 执行命令
            subprocess.run(command, check=True)
            print(f"Video successfully cut and saved to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
    
    def generate_data_splits(self):
        if self.use_data_splits:
            # parse pre-assigned data splits
            print("Use pre-assgined data splits from {}: ".format(os.path.join(self.corpus_path, self.data_splits_path)))
            data_splits = {}
            for split_file in os.listdir(os.path.join(self.corpus_path, self.data_splits_path)):
                print(split_file)
                split_name = split_file.split(".")[0]   
                assert split_name in ["train", "val", "test"]
                data_splits[split_name] = []
                with open(os.path.join(self.corpus_path, self.data_splits_path, split_file), "r") as f:
                    for line in f.readlines():
                        data_splits[split_name].append(line.split("|")[0])
            
            # parse auto-generated train.txt and val.txt in build_from_path
            data_splits_info = {}
            for split_name in data_splits.keys():
                data_splits_info[split_name] = []
            for split_file in ["train_ori.txt", "val_ori.txt"]:
                with open(os.path.join(self.out_dir, split_file), "r") as f:
                    for line in f.readlines():
                        basename = line.split("|")[0]
                        for split_name in data_splits.keys():
                            if basename in data_splits[split_name]:
                                data_splits_info[split_name].append(line)
                                data_splits[split_name].remove(basename)
            for split_name in data_splits.keys():
                with open(os.path.join(self.out_dir, "{}.txt".format(split_name)), "w", encoding="utf-8") as f:
                    f.writelines(data_splits_info[split_name])
                print(split_name)
                print(len(data_splits[split_name]))

    def build_from_path_for_DSU(self):  # 需要对每一个data_split都生成一个文件，DSU才能训练
        print("Processing Data ...")

        speakers = self.speakers.copy()

        # data_splits = os.listdir(self.data_splits_path)
        data_splits = ['train.txt', 'val.txt', 'test.txt']  #  
        for data_split in data_splits:
            out = list()
            with open(os.path.join(self.data_splits_path, data_split), 'r') as file:
                data_info = file.readlines()
            for line in tqdm(data_info):
                basename = line.strip()
                spk = basename.rsplit('_', 1)[0]
                code_path = os.path.join("/nfs-04/yuyue/visualtts_datasets/grid/DSU_data/preprocessed_data/gt_code", spk, basename + '.npy')
                if not os.path.exists(code_path):
                    print(basename)
                    continue
                tg_path = os.path.join(self.textgrid_path, spk, f"{basename}.TextGrid")
                if os.path.exists(tg_path):
                    # text, duration = self.process_utterance(basename, spk)  # 这里返回的text其实是一个phoneme序列
                    info = self.process_utterance_for_DSU(basename, spk)
                    if info:  # 如果不缺失信息，并且每条数据都正常处理，那么就返回处理后的数据，否则返回None
                        out.append(info)
                    else:
                        continue
            
            out = [r for r in out if r is not None]
            with open(os.path.join(self.output_data_split_path, data_split), 'w') as file:
                for m in out:
                    file.write(m + '\n')
        with open(os.path.join(self.output_path, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))
    
    def process_utterance_for_DSU(self, basename, speaker):
        wav_path = os.path.join(self.wav_path, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.wav_path, speaker, "{}.lab".format(basename))
        lip_path = os.path.join(self.av_path, "{}.npy".format(basename))
        tg_path = os.path.join(self.textgrid_path, speaker, "{}.TextGrid".format(basename))
        video_path = os.path.join(self.root_path, "video_25fps", f"{basename}.mp4")

        try:
            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True) 

            if self.trim:
                phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))
                assert len(phone) == len(duration)
                text = "{" + " ".join(phone) + "}"
                if start >= end:
                    return None
                
            else:
                phone, duration, start, end = self.get_alignment_no_trim_for_DSU(textgrid.get_tier_by_name("phones"), basename)
                assert len(phone) == len(duration)
                text = "{" + " ".join(phone) + "}"
                if start >= end:
                    return None
                

                # trim tailing visual embedding
                duration_sum = int(np.ceil(sum(duration) / 2))

                video_stamp_path = os.path.join(self.video_stamp_path, speaker, basename+'.npy')
                video_start, video_end = np.load(video_stamp_path)
                frame_num = video_end - video_start
                # end_idx = int(np.ceil(self.frame_rate * end))  # 秒数✖️
                lip = np.load(lip_path)
                # if basename == "s01_bbwt1a":
                #     print(duration_sum)
                if duration_sum - frame_num >5:
                    exit(f"av_feature from {basename} may be invalid")
                lip = lip[video_start: video_start + duration_sum, :].astype(np.float32)

            # Read raw text
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")

            dur_filename = f"{basename}.npy"
            np.save(os.path.join(self.duration_path, dur_filename), duration)
            # lip_filename = "{}-lip-{}.npy".format(speaker, basename)
            lip_filename = f"{basename}.npy"
            np.save(os.path.join(self.lip_path, lip_filename), lip) # TODO: change lip to sync
            if self.trim:
                # Read wav files
                wav, sr = librosa.load(wav_path, sr=self.sampling_rate)
                if self.sampling_rate != sr:
                    wav = librosa.resample(wav, sr, self.sampling_rate)
                trim_wav_filename = f"{basename}.wav"
                wavfile.write(os.path.join(self.trim_wav_path, trim_wav_filename),
                              self.sampling_rate,
                              wav.astype(np.int16))
            
            return ("|".join([basename, speaker, text, raw_text]))  # text就是phoneme，raw_text是原始的文本。raw_text大小写都无所谓，text对了就行
        except Exception as e:
            print(e)
            return None



if __name__ == "__main__":
    # config = '/nfs-04/yuyue/visualtts_datasets/grid/code/preprocess.yaml'
    # config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    # preprocessor = Preprocessor(config)
    # preprocessor.build_from_path()   # 应该走的是这个

    config_for_DSU = "/nfs-04/yuyue/visualtts_datasets/grid/code/preprocess.yaml"
    config_for_DSU = yaml.load(open(config_for_DSU, "r"), Loader=yaml.FullLoader)
    preprocessor_for_DSU = Preprocessor(config_for_DSU)
    preprocessor_for_DSU.build_from_path_for_DSU()
