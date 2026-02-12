import warnings
from PIL import Image
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .clip import build_clip
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


_tokenizer = _Tokenizer()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def _load_clip(ckpt_path: str, 
               device   : Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load a CLIP model

    Parameters
    ----------
    ckpt_path : str
        path to a model checkpoint (jit) containing the state_dict 

    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    model_jit = torch.jit.load(ckpt_path, map_location="cpu")
    model = build_clip(model_jit.state_dict()).to(device)
    return model, _transform(model.visual.input_resolution)


def tokenize(texts          : Union[str, List[str]], 
             context_length : int = 77, 
             truncate       : bool = False,
             ) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


if __name__=="__main__":
    clip, _ = _load_clip(ckpt_path="/home/t-huluo/v-xihua/data/ckpt/clip/ViT-B-16.pt")
    print("loaded.")