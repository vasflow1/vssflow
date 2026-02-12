import numpy as np
from collections import OrderedDict
# from pkg_resources import packaging
import packaging
from typing import Tuple, Union, List, Any

import torch
from torch import nn

from .simple_tokenizer import SimpleTokenizer as _Tokenizer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width) 
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # NOTE Commented out here. If necessary, it needs to be called explicitly from outside.
        # x = self.ln_post(x[:, 0, :])
        # x = self.ln_post(x[:, 1:, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class CLIPViT(nn.Module):
    def __init__(self,
                 ckpt_path: str = "/home/t-huluo/v-xihua/data/ckpt/clip/ViT-B-16.pt",
                 ):
        super().__init__()
        # init and load CLIP module
        model_jit = torch.jit.load(ckpt_path, map_location="cpu")
        state_dict = model_jit.state_dict()

        embed_dim = state_dict["text_projection"].shape[1]
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,      # 224
            patch_size=vision_patch_size,           # 16
            width=vision_width,                     # 768
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim                    # 512
        )

        visual_state_dict = {k:v for k,v in state_dict.items() if k.startswith("visual")}
        self.load_state_dict(visual_state_dict, strict=True)
        
        # NOTE Added extensions. Post transformer layer.
        # self.transformer_post = Transformer(vision_width, 3, vision_heads)
        # self.ln_post = LayerNorm(vision_width) 
        # scale = vision_width ** -0.5
        # self.proj = nn.Parameter(scale * torch.randn(vision_width, 32))
        
        self.vision_width = vision_width

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image):
        x = self.visual(image.type(self.dtype))
        
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer_post(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        
        # x = self.ln_post(x[:, 0, :])
        # x = self.ln_post(x[:, 1:, :])
        # if self.proj is not None:
        #     x = x @ self.proj
        
        return x
    

class CLIPBert(nn.Module):
    def __init__(self,
                 ckpt_path: str = "/home/t-huluo/v-xihua/data/ckpt/clip/ViT-B-16.pt",
                 ):
        super().__init__()
        model_jit = torch.jit.load(ckpt_path, map_location="cpu")
        state_dict = model_jit.state_dict()

        self.embed_dim = state_dict["text_projection"].shape[1]
        self.vocab_size = state_dict["vocab_size"]
        self.context_length = state_dict["context_length"]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        # self.text_projection = nn.Parameter(torch.empty(transformer_width, self.embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        bert_state_dict = {k:v for k,v in state_dict.items() if (k.startswith("transformer") or
                                                                 k.startswith("token_embedding") or
                                                                 k.startswith("positional_embedding") or
                                                                 k.startswith("ln_final"))}
        self.load_state_dict(bert_state_dict, strict=True)
        self.transformer_width = transformer_width
        self.tokenizer = _Tokenizer()

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype
    
    def tokenize(self,
                 texts          : Union[str, List[str]], 
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
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
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

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, text):
        # text = self.tokenize(text, 77, True)
        text_features = self.encode_text(text)

        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features


if __name__=="__main__":
    clip_vit = CLIPViT(ckpt_path="/data_mount/ckpt/clip/ViT-B-32.pt")
    x = torch.rand((32, 3, 224, 224))
    y = clip_vit(x)
    print(x.size())
    print(y.size())
    
    # clip_bert = CLIPBert(ckpt_path="/home/t-huluo/v-xihua/data/ckpt/clip/ViT-B-16.pt")
    # print("loaded.")
    # x = ["I love you.", "", "I'm missing you so much.", 'a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror.a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror. a mirror. Inside the mirror.']
    # y = clip_bert(x)
    # print(y.size())