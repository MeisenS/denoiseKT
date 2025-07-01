import torch 
import torch.nn as nn
from torch.nn import Dropout
import pandas as pd
from .utils import transformer_FFN, get_clones, ut_mask, pos_encode
from torch.nn import Embedding, Linear
import math
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w, beta_sche):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w
        self.beta_sche = beta_sche

        if beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start) 
            # noise = torch.randn_like(x_start) / 100
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_x = denoise_model(x_noisy, h, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h):
        x = torch.randn_like(h)

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h, torch.full((h.shape[0], ), n, device=device, dtype=torch.long), n)

        return x

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


class desaint(nn.Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, n_blocks=1, emb_type="qid", emb_path="", pretrain_dim=768,
                 lamb=1e-4,timesteps=200, beta_start=0.0001, beta_end=0.02, diffusion_w=2.0, beta_sche='exp'):
        super().__init__()
        print(f"num_q: {num_q}, num_c: {num_c}")
        if num_q == num_c and num_q == 0:
            assert num_q != 0
        self.num_q = num_q
        self.num_c = num_c
        self.model_name = "desaint"
        self.num_en = n_blocks
        self.num_de = n_blocks
        self.emb_type = emb_type
        self.seq_len = seq_len
        self.lamb = lamb
        self.num_attn_heads = num_attn_heads
        self.embd_pos = nn.Embedding(seq_len, embedding_dim = emb_size) 
        self.init_alphas = nn.Parameter(1e-3 * torch.randn(self.num_attn_heads, self.seq_len, 2))
        
        # Diffusion parameters
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.diffusion_w = diffusion_w
        self.beta_sche = beta_sche

        # Initialize diffusion model
        self.diffusion = diffusion(timesteps=self.timesteps, 
                                  beta_start=self.beta_start, 
                                  beta_end=self.beta_end, 
                                  w=self.diffusion_w, 
                                  beta_sche=self.beta_sche)
        
        # Sinusoidal time step embeddings for diffusion
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.emb_size),
            nn.Linear(self.emb_size, self.emb_size*2),
            nn.GELU(),
            nn.Linear(self.emb_size*2, self.emb_size),
        )
        
        # Denoise model for diffusion
        self.denoise_model = nn.Sequential(
            nn.Linear(self.emb_size * 3, self.emb_size*2),
            nn.GELU(),
            nn.Linear(self.emb_size*2, self.emb_size),
        )
        
        # Unconditional denoise model
        self.unconditional_embedding = nn.Embedding(1, self.emb_size)

        if emb_type.startswith("qid"):
            self.encoder = get_clones(Encoder_block(emb_size, num_attn_heads, num_q, num_c, seq_len, dropout), self.num_en)
        
        self.decoder = get_clones(Decoder_block(emb_size, 2, num_attn_heads, seq_len, dropout,num_q,num_c), self.num_de)

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=emb_size, out_features=1)
    
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

        mask = mask[:, :, 0].unsqueeze(2)   # [num_heads, seq_len, 1]
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

    def denoise(self, x_noisy, h, t):
        """Denoise model for diffusion process"""
        t_emb = self.time_mlp(t)
        return self.denoise_model(torch.cat([x_noisy, h, t_emb], dim=-1))
    
    def unconditional_denoise(self, x_noisy, t):
        """Unconditional denoise model for diffusion process"""
        batch_size = x_noisy.shape[0]
        h = self.unconditional_embedding(torch.zeros(batch_size, dtype=torch.long).to(device))
        t_emb = self.time_mlp(t)
        return self.denoise_model(torch.cat([x_noisy, h, t_emb], dim=-1))
    
    def forward(self, in_ex, in_cat, in_res, qtest=False):
        raw_in_ex = in_ex
        raw_in_cat = in_cat
        emb_type = self.emb_type       

        dam = self.generate_differentiable_attention_mask()

        if self.num_q > 0:
            in_pos = pos_encode(in_ex.shape[1]).to(device)
        else:
            in_pos = pos_encode(in_cat.shape[1]).to(device)
        in_pos = self.embd_pos(in_pos)
        # in_pos = self.embd_pos.unsqueeze(0)
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            if emb_type == "qid": # same to qid in saint

                in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block, dam = dam)
            in_cat = in_ex
        ## pass through each decoder blocks in sequence
        start_token = torch.tensor([[0]]).repeat(in_res.shape[0], 1).to(device)#give start token defalut response to 2
        in_res = torch.cat((start_token, in_res), dim=-1)
       
    
        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](raw_in_ex, raw_in_cat,in_res, in_pos, en_out=in_ex, first_block=first_block, dam = dam)
        
        ## Output layer

        res = self.out(self.dropout(in_res))
        res = torch.sigmoid(res).squeeze(-1)
        mask_L1 = torch.norm(self.init_alphas, p=1)
        if not qtest:
            return res, mask_L1
        else:
            return res, in_res, mask_L1


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout, emb_path="", pretrain_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex
        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex, embedding_dim = dim_model)                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = Embedding.from_pretrained(embs)
                self.linear = Linear(pretrain_dim, dim_model)
                
        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim = dim_model)
       

        self.multi_en = nn.MultiheadAttention(embed_dim = dim_model, num_heads = heads_en, dropout = dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True, dam=None):

        ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            embs = []
            if self.total_ex > 0:#question embedding
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:#concept embedding
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
            # in_pos = self.embd_pos(in_pos)
        else:
            out = in_ex
        
        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        if dam is not None:
            dam_head = dam
        else:
            dam_head = None

        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape)
        
        # norm -> attn -> drop -> skip corresponging to transformers' norm_first
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1(out)                           # Layer norm
        skip_out = out 

        if dam_head is not None:
            attn_mask = ut_mask(seq_len=n).to(out.device)
            attn_mask = attn_mask.masked_fill(dam_head == 0, float('-inf'))
        else:
            attn_mask = ut_mask(seq_len=n).to(out.device)
        
        out, attn_wt = self.multi_en(out, out, out, attn_mask=attn_mask.repeat(64, 1, 1))
        out = self.dropout1(out)
        out = out + skip_out                                  # skip connection

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2(out)                           # Layer norm 
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out                                    # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout,num_q,num_c):
        super().__init__()
        self.seq_len    = seq_len
        self.num_q = num_q
        self.num_c = num_c
        self.embd_res = nn.Embedding(total_res+1, embedding_dim = dim_model)                  #response embedding, include a start token
        self.embd_ex = nn.Embedding(num_q*2+1, embedding_dim = dim_model)
        self.emb_cat = nn.Embedding(num_c*2+1, embedding_dim = dim_model)
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding
        self.multi_de1  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = transformer_FFN(dim_model, dropout)                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        


    def forward(self,in_ex, in_cat,in_res, in_pos, en_out,first_block=True, dam=None):

         ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            in_in = self.embd_res(in_res)
            # print(f"in_ex is {in_ex}")
            # print(f"self.num_q is {self.num_q}")
            # print(f"in_res is {in_res}")
            # print(in_ex + self.num_q * in_res)
            que_emb = self.embd_ex(in_ex + self.num_q * in_res)
            cat_emb = self.emb_cat(in_cat + self.num_c * in_res)
            
            #combining the embedings
            out = in_in + que_emb + cat_emb + in_pos                 # (b,n,d)
        else:
            out = in_res

        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape)
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        if dam is not None:
            dam_head = dam
            attn_mask = ut_mask(seq_len=n).to(out.device)
            attn_mask = attn_mask.masked_fill(dam_head == 0, float('-inf'))
        else:
            attn_mask = ut_mask(seq_len=n).to(out.device)

        out, attn_wt = self.multi_de1(out, out, out, attn_mask=attn_mask.repeat(64, 1, 1))
        out = self.dropout1(out)
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        # out, attn_wt = self.multi_de2(out, en_out, en_out,
        #                             attn_mask=ut_mask(seq_len=n).to(out.device))  # attention mask upper triangular
        if dam is not None:
            dam_head = dam
            attn_mask = ut_mask(seq_len=n).to(out.device)
            attn_mask = attn_mask.masked_fill(dam_head == 0, float('-inf'))
        else:
            attn_mask = ut_mask(seq_len=n).to(out.device)
        out, attnZ_wt = self.multi_de2(out, en_out, en_out,
                                    attn_mask=attn_mask.repeat(64, 1, 1))  # attention mask upper triangular
        out = self.dropout2(out)
        out = out + skip_out

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3(out)                               # Layer norm 
        skip_out = out
        out = self.ffn_en(out)                                    
        out = self.dropout3(out)
        out = out + skip_out                                        # skip connection

        return out
