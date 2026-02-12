import torch
from torch import nn
import math
import numpy as np

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        """
        初始化正弦位置编码
        Args:
            d_model (int): 模型的隐藏维度（必须为偶数）
            max_seq_len (int): 最大序列长度
            dropout (float): Dropout 概率
        """
        super(SinusoidalPositionalEmbedding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal positional encoding"

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)


    def forward(self, x, position_ids=None):
        seq_len = x.size(1)
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        if position_ids is None:
            pe = self.pe[:, :seq_len, :]  # (1, seq_len, d_model)
        else:
            pe = self.pe[:, position_ids, :]  # 支持自定义位置 ID

        
        x = x + pe
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # 自注意力机制
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout
        
    
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接和层归一化
        return x
    

class Encoder(nn.Module):
    def __init__(self, inputq_dim: int, output_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 1024, dropout: float = 0.0) -> None:
        super(Encoder, self).__init__()
        self.inputq_proj = nn.Linear(inputq_dim, d_model)
        self.positial_embedding_q = SinusoidalPositionalEmbedding(d_model, max_seq_len=1024)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, q: torch.Tensor):
        q = self.inputq_proj(q) # [bs, seq_len, d_model]
        q = self.positial_embedding_q(q)  # [bs, seq_len, d_model]
        for layer in self.layers:
            q = layer(q)
        q = self.output_proj(q)
        return q
    


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # 自注意力机制
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # 交叉注意力机制
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Dropout
        
    def forward(self, x, enc_output):
        attn_output, _  = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        attn_output, attn_weight = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn_output))  # 残差连接和层归一化
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # 残差连接和层归一化
        return x, attn_weight
    



class Decoder(nn.Module):
    def __init__(self, inputq_dim: int, inputkv_dim: int, output_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 1024, dropout: float = 0.0) -> None:
        super(Decoder, self).__init__()
        self.inputq_proj = nn.Linear(inputq_dim, d_model)
        self.inputkv_proj = nn.Linear(inputkv_dim, d_model)
        self.positial_embedding_q = SinusoidalPositionalEmbedding(d_model, max_seq_len=1024)
        self.positial_embedding_kv = SinusoidalPositionalEmbedding(d_model, max_seq_len=1024)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout=dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, q: torch.Tensor, kv = torch.Tensor):
        q = self.inputq_proj(q)  # [bs, seq_len, d_model]
        q = self.positial_embedding_q(q)  # [bs, seq_len, d_model]
        kv = self.inputkv_proj(kv) 
        kv = self.positial_embedding_kv(kv)  # [bs, seq_len, d_model]

        attn_weight_list = []
        for layer in self.layers:

            q, aw = layer(q, kv)
            attn_weight_list.append(aw)
        q = self.output_proj(q)
        return q, attn_weight_list



class AttnDP_visualtts(nn.Module):
    def __init__(self, 
                 phone_inputq_dim: int, 
                 phone_inputkv_dim: int, 
                 phone_d_model: int, 
                 phone_nheads: int,
                 hubert_inputq_dim: int, 
                 hubert_inputkv_dim: int,
                 hubert_d_model: int,
                 hubert_nheads: int,
                 encoder_num_layers: int, 

                 attn_inputq_dim: int, 
                 attn_inputkv_dim: int,
                 attn_output_dim: int,
                 attn_d_model: int,
                 attn_nhead: int,
                 attn_num_layers: int,

                 output_dim: int = 2
                 ) -> None:
        super(AttnDP_visualtts, self).__init__()
        self.phone_embedding = nn.Embedding(1001, phone_inputq_dim)  # Assuming phoneme IDs are in range [0, 63]
        self.phone_encoder = Encoder(phone_inputq_dim, attn_inputq_dim, phone_d_model, phone_nheads, encoder_num_layers)
        self.hubert_encoder = Encoder(hubert_inputq_dim, attn_inputkv_dim, hubert_d_model, hubert_nheads, encoder_num_layers)

        self.attn_decoder = Decoder(attn_inputq_dim, attn_inputkv_dim, attn_output_dim, attn_d_model, attn_nhead, attn_num_layers)
        self.output_proj = nn.Linear(attn_output_dim, output_dim)

    def forward(self, phoneme: torch.Tensor, avhubert: torch.Tensor):
        phoneme = self.phone_embedding(phoneme)  # [bs, phone_seq_len, phone_inputq_dim]
        hubert_feature = self.hubert_encoder(q=avhubert)
        phone_feature = self.phone_encoder(q=phoneme)
        attn_output, attn_weight_list = self.attn_decoder(q = hubert_feature, kv = phone_feature)
        output = self.output_proj(attn_output)
        return output, attn_weight_list
    



if __name__ == "__main__":
    bs = 16
    hubert_feature = np.load('/nfs-04/yuyue/visualtts_datasets/chem/chem_for_espnet/preprocessed_data/lip_feature/chem_00_ZwsWjelzqDA-062.npy')  # [hubert_seq_len, 1024]
    hubert_feature = torch.from_numpy(hubert_feature).float()  
    hubert_feature = torch.stack([hubert_feature for _ in range(bs)], dim=0)  # [bs, hubert_seq_len, 1024]
    phoneme = torch.tensor([43,58,17,28,4,59,34,51,61,56,7,23,59,50,20,61,59,34,51,59])  # [phone_seq_len]
    phoneme = phoneme.unsqueeze(0).repeat(bs, 1).to(torch.int64)            # [bs, phone_seq_len, 1]
 

    dp = AttnDP_visualtts(phone_inputq_dim = 64, 
                 phone_inputkv_dim = 64,
                 phone_d_model = 1024, 
                 phone_nheads = 8,

                 hubert_inputq_dim = 1024, 
                 hubert_inputkv_dim = 1024,
                 hubert_d_model = 1024,
                 hubert_nheads = 8,
                 encoder_num_layers = 4, 

                 attn_inputq_dim = 1024, 
                 attn_inputkv_dim = 1024,
                 attn_output_dim = 256,
                 attn_d_model = 512,
                 attn_nhead = 1,
                 attn_num_layers = 4)
    duration, attn_weight = dp(phoneme, hubert_feature)
    print(duration.shape, attn_weight[-1].shape)  # [bs, phone_seq_len, 2]
