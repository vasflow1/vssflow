import re
import os
import sys
import warnings
from glob import glob
from tqdm import tqdm
# from text import _clean_text, text_to_sequence
import g2p_en
import tacotron_cleaner.cleaners
from pathlib import Path
from typeguard import typechecked
from abc import ABC, abstractmethod
from typing import Collection, Optional, List, Iterable, Union
import torch
import yaml

os.chdir(os.path.dirname(sys.path[0]))

try:
    from vietnamese_cleaner import vietnamese_cleaners
except ImportError:
    vietnamese_cleaners = None

try:
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
except (ImportError, SyntaxError):
    BasicTextNormalizer = None

# 加载现有的词典，并添加现有词典中不存在的词 在lab之后

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

class G2p_en:
    """On behalf of g2p_en.G2p.

    g2p_en.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2p_en.G2p is instantiated upon calling this class.

    """

    def __init__(self, no_space: bool = False):
        self.no_space = no_space
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            self.g2p = g2p_en.G2p()

        phones = self.g2p(text)
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))
        return phones
    
    def for_words(self, text) -> List[str]:
        if self.g2p is None:
            self.g2p = g2p_en.G2p()

        # phones = self.g2p(text)
        words, phones = self.g2p.for_word(text)
        
        return words, phones

class AbsTokenizer(ABC, torch.nn.Module):
    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError

class PhonemeTokenizer(AbsTokenizer):
    @typechecked
    def __init__(
        self,
        g2p_type: Union[None, str],  # g2p_en_no_space
        non_linguistic_symbols: Union[None, Path, str, Iterable[str]] = None,  # None
        space_symbol: str = "<space>",  # 默认（不传） "<space>"
        remove_non_linguistic_symbols: bool = False,  # false 不传
    ):
        if g2p_type == "g2p_en":
            self.g2p = G2p_en(no_space=False)
        elif g2p_type == "g2p_en_no_space":
            self.g2p = G2p_en(no_space=True)
        else:
            raise NotImplementedError(f"Not supported: g2p_type={g2p_type}")

        self.g2p_type = g2p_type
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'g2p_type="{self.g2p_type}", '
            f'space_symbol="{self.space_symbol}", '
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            ")"
        )

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                tokens.append(t)
                line = line[1:]

        line = "".join(tokens)
        tokens = self.g2p(line)
        return tokens

    def text2tokens_for_words(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                tokens.append(t)
                line = line[1:]

        line = "".join(tokens)
        words, phones = self.g2p.for_words(line)

        return words, phones

    def tokens2text(self, tokens: Iterable[str]) -> str:
        # phoneme type is not invertible
        return "".join(tokens)

    def text2tokens_svs(self, syllable: str) -> List[str]:
        # Note(Yuning): fix syllabel2phoneme mismatch
        # If needed, customed_dic can be changed into extra input
        customed_dic = {
            "へ": ["h", "e"],
            "は": ["h", "a"],
            "シ": ["sh", "I"],
            "ヴぁ": ["v", "a"],
            "ヴぃ": ["v", "i"],
            "ヴぇ": ["v", "e"],
            "ヴぉ": ["v", "o"],
            "でぇ": ["dy", "e"],
            "くぁ": ["k", "w", "a"],
            "くぃ": ["k", "w", "i"],
            "くぅ": ["k", "w", "u"],
            "くぇ": ["k", "w", "e"],
            "くぉ": ["k", "w", "o"],
            "ぐぁ": ["g", "w", "a"],
            "ぐぃ": ["g", "w", "i"],
            "ぐぅ": ["g", "w", "u"],
            "ぐぇ": ["g", "w", "e"],
            "ぐぉ": ["g", "w", "o"],
            "くぉっ": ["k", "w", "o", "cl"],
        }
        tokens = self.g2p(syllable)
        if syllable in customed_dic:
            tokens = customed_dic[syllable]
        return tokens


def normalize_nonchar(text, inference=False):
    return re.sub(r"\{[^\w\s]?\}", "{sp}", text) if inference else\
            re.sub(r"[^\w\s|\']?", "", text)

def extract_nonen(preprocess_config):
    in_dir = preprocess_config["path"]["raw_path"]
    filelist = open(f'{in_dir}/nonen.txt', 'w', encoding='utf-8')

    count = 0
    nonen = set()
    print("Extract non english charactors...")
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_count = len(lines)
        for line in tqdm(lines):
            wav = line.split('|')[0]
            text = line.split('|')[1]

            reg = re.compile("""[^ a-zA-Z~!.,?:`"'＂“‘’”’]+""")
            impurities = reg.findall(text)
            if len(impurities) == 0:
                count+=1
                continue
            norm = _clean_text(text, preprocess_config["preprocessing"]["text"]["text_cleaners"])
            impurities_str = ','.join(impurities)
            filelist.write(f'{norm}|{text}|{impurities_str}|{wav}\n')
            for imp in impurities:
                nonen.add(imp)
    filelist.close()
    print('Total {} non english charactors from {} lines'.format(len(nonen), total_count-count))
    print(sorted(list(nonen)))


def extract_lexicon():
    """
    Extract lexicon and build grapheme-phoneme dictionary for MFA training
    """
    # in_dir = preprocess_config["path"]["corpus_path"]
    in_dir = "/nfs-04/yuyue/visualtts_datasets/grid/preprocessed_data/speakers"
    lexicon_path = "/nfs-04/yuyue/visualtts_datasets/grid/preprocessed_data/grid_dict.txt"  # 字典的地址
    filelist = open(lexicon_path, 'a+', encoding='utf-8')

    # Load Lexicon Dictionary 
    done = set()
    if os.path.isfile(lexicon_path):
        filelist.seek(0)
        for line in filelist.readlines():
            grapheme = line.split("\t")[0]
            done.add(grapheme)

    print("Extract lexicon...")

    # cleaner = TextCleaner("tacotron")
    tokenizer = PhonemeTokenizer(g2p_type="g2p_en_no_space",
                                 non_linguistic_symbols=None,)
    
    lab_list = glob(f'{in_dir}/**/*.lab', recursive=True)
    for lab in tqdm(lab_list):
        with open(lab, 'r', encoding='utf-8') as f:
            text = f.readline().strip("\n")
        # text = cleaner(text)  在get_wav_lab的时候已经clean过了，所以不需要再clean一次
        phonemes = tokenizer.text2tokens(text)
        words, word_tokens = tokenizer.text2tokens_for_words(text)
        
        phoneme = []
        for item in word_tokens:
            phoneme += item
        if phoneme != phonemes:
            print(text)
            # print(phoneme)
            # print(phonemes)
        for i in range(len(words)):
            word, phoneme = words[i], word_tokens[i]
            phoneme = " ".join(phoneme)
            filelist.write("{}\t{}\n".format(word, phoneme))
            done.add(word)
        
    filelist.close()

    return lexicon_path

if __name__ == "__main__":
    # preprocess_config = "/nfs-04/yuyue/visualtts_datasets/lrs2/codes/preprocess_for_chem.yaml"
    # preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    lexicon_path = extract_lexicon()

    print(f"get lexicon in {lexicon_path}")
    
    # def add_spaces_around_digits(input_string):
    #     # 使用正则表达式查找数字和字母之间没有空格的情况
    #     result = re.sub(r'(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)', r' \g<0> ', input_string)
    #     return result
    
    # cleaner = TextCleaner("tacotron")
    # tokenizer = PhonemeTokenizer(g2p_type="g2p_en_no_space",
    #                              non_linguistic_symbols=None,)
    
    # # # process text
    # text = "That's the final state minus the initial state-- R minus 1/4R is 3/4R."
    # # # text = "JUST BEFORE WE TAKE OUR FIRST QUESTION IF YOU WANT TO BE AS THOUGH YOU WERE HERE IN THE AUDIENCE ARGUING WHAT'S SAID BY THE PANEL YOU'VE GOT FACEBOOK AND TWITTER YOU CAN TEXT US ON 83981 AND YOU CAN COMMENT ON ANYTHING THAT'S SAID SO DO DO THAT OUR FIRST QUESTION FROM LEWIS KELLER PLEASE LEWIS KELLER"
    # # text = "BECAUSE LIKE HIM TIME AND TIDE WAITS FOR NO MAN OR BIRD AS THE SPRING TIDE ADVANCES UP THE MUDFLATS SUDDENLY THE WATER COMES RIGHT UP TO THE TOP OF THE KNOTS' LEGS AND THEY TAKE FLIGHT THAT IS WHEN THE SPECTACLE BEGINS "
    # text = add_spaces_around_digits(text)
    # print(text)
    # text = cleaner(text)
    # print(text)
    # phonemes = tokenizer.text2tokens(text)
    # words, word_tokens = tokenizer.text2tokens_for_words(text)
    # phoneme = []
    # for item in word_tokens:
    #     phoneme += item
    # # value = [f"{tok}" for tok in text]  # 如何写？
    # print(phonemes)
    # print(phoneme)
    # print(phoneme == phonemes)