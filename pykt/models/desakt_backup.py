import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class desakt(Module):
    def __init__(self, num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", pretrain_dim=768,lamb=1e-4):
        super().__init__()
        self.model_name = "desakt"
        self.emb_type = emb_type

        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en
        self.lamb = lamb

        if emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = Embedding(num_c, emb_size)
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)
        self.init_alphas = nn.Parameter(1e-3 * torch.randn(self.seq_len, 2))
    
    def generate_differentiable_attention_mask(self):
        logits = self.init_alphas
        tau = 1
        dim = -1
        hard = True
        eps = 1e-5

        gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)

        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            mask = y_hard - y_soft.detach() + y_soft
        else:
            mask = y_soft
        mask = mask[:, 0].unsqueeze(-1)   # [num_heads, seq_len, 1]
        mask = mask.expand(self.num_attn_heads, self.seq_len, self.seq_len)
        mask = torch.triu(mask)
        mask = torch.rot90(mask, 1, (1, 2))
        mask_tri = torch.zeros(self.num_attn_heads, self.seq_len, self.seq_len).to(mask.device)
        mask_tri[:, 0] = mask[:, 0]
        for i in range(1, self.seq_len):
            mask_tri[:, i, i:] = mask[:, i, :-i]
        masks = mask_tri + torch.transpose(torch.triu(mask_tri, 1), 1, 2)

        mask = masks.to(dtype=next(self.parameters()).dtype)
        return mask

    def base_emb(self, q, r, qry):
        x = q + self.num_c * r
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
    
        posemb = self.position_emb(pos_encode(xemb.shape[1]).to(device))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, q, r, qry, qtest=False):
        emb_type = self.emb_type
        qemb, qshftemb, xemb = None, None, None
        if emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
        dam = self.generate_differentiable_attention_mask()
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb, dam=dam)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        mask_L1 = torch.norm(self.init_alphas, p=1)
        if not qtest:
            return p, mask_L1
        else:
            return p, xemb, mask_L1

class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None, dam = None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(seq_len=k.shape[0]).to(k.device)
        if dam is not None:
            attn_mask = causal_mask.masked_fill(dam == 0, float('-inf'))
        else:
            attn_mask = causal_mask
        
        attn_emb, _ = self.attn(q, k, v, attn_mask=attn_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb