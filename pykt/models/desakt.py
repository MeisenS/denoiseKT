import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones
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


class desakt(Module):
    def __init__(self, num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", pretrain_dim=768, lamb=1e-4,
                 timesteps=200, beta_start=0.0001, beta_end=0.02, diffusion_w=2.0, beta_sche='exp'):
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

    def denoise(self, x_noisy, h, t):
        """Denoise model for diffusion process"""
        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, x_noisy.shape[1], -1)  
        cat_in = torch.cat([x_noisy, h, t_emb], dim=-1)
        return self.denoise_model(cat_in)
    
    def unconditional_denoise(self, x_noisy, t):
        """Unconditional denoise model for diffusion process"""
        batch_size = x_noisy.shape[0]
        h0 = self.unconditional_embedding(torch.zeros(batch_size, dtype=torch.long).to(device))
        h = h0.unsqueeze(0).expand(x_noisy.shape[0], -1, -1)
        t_emb = self.time_mlp(t)
        return self.denoise_model(torch.cat([x_noisy, h, t_emb], dim=-1))

    def forward(self, q, r, qry, qtest=False):
        emb_type = self.emb_type
        qemb, qshftemb, xemb = None, None, None
        # Get base embeddings
        if emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
        # Get differentiable attention mask    
        dam = self.generate_differentiable_attention_mask()
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")

        # Standard DESAKT transformer forward pass
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb, dam=dam)
         
        # Using diffusion to enhance representations
        if not qtest:
            # Training phase: calculate diffusion loss
            batch_size = xemb.shape[0]
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
            diff_loss, _ = self.diffusion.p_losses(
                denoise_model=lambda x, h, t: self.denoise(x, h, t),
                x_start=xemb,
                h=qshftemb,
                t=t
            )
            
            # Regular prediction
            p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
            mask_L1 = torch.norm(self.init_alphas, p=1)
            
            # Return prediction with added diffusion loss
            return p, mask_L1 + self.lamb * diff_loss
        else:
            # Testing phase: use diffusion sampling to enhance representation
            enhanced_xemb = self.diffusion.sample(
                model_forward=lambda x, h, t: self.denoise(x, h, t),
                model_forward_uncon=lambda x, t: self.unconditional_denoise(x, t),
                h=qshftemb
            )
            
            # Combine original and enhanced embeddings
            xemb = 0.7 * xemb + 0.3 * enhanced_xemb  # Weighted combination
            
            # Make prediction with enhanced representation
            p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
            mask_L1 = torch.norm(self.init_alphas, p=1)
            
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