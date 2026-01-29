import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from . import initialization as init
from einops import rearrange, reduce, pack, unpack
from typing import Optional, Tuple, Union



########################
# define all activation functions that are not in pytorch library. 
########################
class Swish(nn.Module):
	def __init__(self):
		super().__init__()
		self.Sigmoid = nn.Sigmoid()
	def forward(self,x):
		return x*self.Sigmoid(x)

class Sine(nn.Module):
    def __init__(self, w0 = init.DEFAULT_W0):
        self.w0 = float(w0)                               # w0 is not learnable
        super().__init__()

    def forward(self, input):
        return torch.sin(self.w0 * input)

class Sine_tw(nn.Module):
    def __init__(self, w0 = init.DEFAULT_W0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor([w0], dtype = torch.float32))          # w0 is learnable

    def forward(self, input):
        return torch.sin(self.w0 * input)


# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
# special first-layer initialization scheme
# different layers has different initialization schemes:
NLS_AND_INITS = {
    # act name: (init func, first layer init func )
    'sine':(Sine(), init.sine_init_wrapper, init.first_layer_sine_init), 
    'relu':(nn.ReLU(inplace=True), init.init_weights_normal, None),
    'sigmoid':(nn.Sigmoid(), init.init_weights_xavier, None),
    'tanh':(nn.Tanh(), init.init_weights_xavier, None),
    'selu':(nn.SELU(inplace=True), init.init_weights_selu, None),
    'softplus':(nn.Softplus(), init.init_weights_normal, None),
    'elu':(nn.ELU(inplace=True), init.init_weights_elu, None),
    'swish':(Swish(), init.init_weights_xavier, None),
}


########################
# denfine all the basic layers 
########################

class BatchLinear(nn.Linear):
    '''
    This is a linear transformation implemented manually. It also allows maually input parameters. 
    for initialization, (in_features, out_features) needs to be provided. 
    weight is of shape (out_features*in_features)
    bias is of shape (out_features)
    '''
    __doc__ = nn.Linear.__doc__
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        
        weight = params['weight']
        bias = params.get('bias', None)
        
        return F.linear(input, weight, bias)


class FeatureMapping:
    '''
    This is feature mapping class for  fourier feature networks 
    '''
    def __init__(self, in_features, mode = 'basic', 
                 gaussian_mapping_size = 256, gaussian_rand_key = 0, gaussian_tau = 1.,
                 pe_num_freqs = 4, pe_scale = 2, pe_init_scale = 1,pe_use_nyquist=True, pe_lowest_dim = None, 
                 rbf_out_features = None, rbf_range = 1., rbf_std=0.5):
        '''
        inputs:
            in_freatures: number of input features
            mapping_size: output features for Gaussian mapping
            rand_key: random key for Gaussian mapping
            tau: standard deviation for Gaussian mapping
            num_freqs: number of frequencies for P.E.
            scale = 2: base scale of frequencies for P.E.
            init_scale: initial scale for P.E.
            use_nyquist: use nyquist to calculate num_freqs or not. 
        
        '''
        self.mode = mode
        if mode == 'basic':
            self.B = np.eye(in_features)
        elif mode == 'gaussian':
            rng = np.random.default_rng(gaussian_rand_key)
            self.B = rng.normal(loc = 0., scale = gaussian_tau, size = (gaussian_mapping_size, in_features))
        elif mode == 'positional':
            if pe_use_nyquist == 'True' and pe_lowest_dim:  
                pe_num_freqs = self.get_num_frequencies_nyquist(pe_lowest_dim)
            self.B = pe_init_scale * np.vstack([(pe_scale**i)* np.eye(in_features) for i in range(pe_num_freqs)])
            self.dim = self.B.shape[0]*2
        elif mode == 'rbf':
            self.centers = nn.Parameter(torch.empty((rbf_out_features, in_features), dtype = torch.float32))
            self.sigmas = nn.Parameter(torch.empty(rbf_out_features, dtype = torch.float32))
            nn.init.uniform_(self.centers, -1*rbf_range, rbf_range)
            nn.init.constant_(self.sigmas, rbf_std)

    def __call__(self, input):
        if self.mode in ['basic', 'gaussian', 'positional']: 
            return self.fourier_mapping(input,self.B)
        elif self.mode =='rbf':
            return self.rbf_mapping(input)
            
    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    # Fourier feature mapping
    @staticmethod
    def fourier_mapping(x, B):
        '''
        x is the input, B is the reference information 
        '''
        if B is None:
            return x
        else:
            B = torch.tensor(B, dtype = torch.float32, device = x.device)
            x_proj = (2.*np.pi*x) @ B.T
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
    # rbf mapping
    def rbf_mapping(self, x):

        size = (x.shape[:-1])+ self.centers.shape
        x = x.unsqueeze(-2).expand(size)
        # c = self.centres.unsqueeze(0).expand(size)
        # distances = (x - self.centers).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        distances = (x - self.centers).pow(2).sum(-1) * self.sigmas
        return self.gaussian(distances)
    
    @staticmethod
    def gaussian(alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi
    

class LipschitzLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        with torch.no_grad():
            c0 = self.weight.abs().sum(dim=1).max()
        self.c = nn.Parameter(c0.clone())

    def normalized_weight(self):
        """
        W_hat = W * min(1, softplus(c) / row_sum)
        """
        row_sum = self.weight.abs().sum(dim=1)          # [out]
        scale = torch.minimum(
            torch.ones_like(row_sum),
            F.softplus(self.c) / row_sum
        )
        return self.weight * scale[:, None]

    def forward(self, x):
        W_hat = self.normalized_weight()
        return F.linear(x, W_hat, self.bias)

    def get_c(self):
        return F.softplus(self.c)




# =======================================================
#                Transolver Autoencoder
# =======================================================


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


def matmul_single(fx_mid, slice_weights):
    return fx_mid.T @ slice_weights


def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class Physics_Attention(nn.Module):
    """Transolver original implementation"""

    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0., 
        slice_num: int = 64,
        slice_only: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        if not slice_only:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
        
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_latent=False, return_slice_weights=False):
        x_flat, ps = pack_one(x, 'b * c')
        
        fx_mid = rearrange(self.in_project_fx(x_flat), 'b n (h d) -> b h n d', h=self.heads)
        x_mid = rearrange(self.in_project_x(x_flat),'b n (h d) -> b h n d', h=self.heads)
        
        slice_logits = self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        slice_weights = torch.softmax(slice_logits, dim=-1)  # [B, H, N, G]
        
        slice_tokens = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        norm = reduce(slice_weights, 'b h n g -> b h g', 'sum')
        slice_tokens = slice_tokens / (norm[..., None] + 1e-5)
        
        q_slice_token = self.to_q(slice_tokens)
        k_slice_token = self.to_k(slice_tokens)
        v_slice_token = self.to_v(slice_tokens)
        
        attn = torch.softmax(torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale, dim=-1)  # [B H G G]
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # [B H G D]
        
        if return_latent:
            latent = rearrange(out_slice_token, 'b h g d -> b g (h d)')
            if return_slice_weights:
                return latent, slice_weights
            return latent
        
        out_flat = torch.einsum('bhgd,bhng->bhnd', out_slice_token, slice_weights)
        out_flat = rearrange(out_flat, 'b h n d -> b n (h d)')
        out_flat = self.to_out(out_flat)  # [B, N, C]
        
        out = unpack_one(out_flat, ps, 'b * c')

        if return_slice_weights and not return_latent:
            return out, slice_weights
        return out


class Physics_Attention_plus(nn.Module):
    """Transolver ++ implementation"""

    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0., 
        slice_num: int = 64,
        slice_only: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        
        # regression temperature
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )
        
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        if not slice_only:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_latent=False, return_slice_weights=False):
        x_flat, ps = pack_one(x, 'b * c')
        
        x_mid = rearrange(self.in_project_x(x_flat),'b n (h d) -> b h n d', h=self.heads)
        
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        norm = reduce(slice_weights, 'b h n g -> b h g', 'sum')
        
        slice_tokens = torch.einsum('bhnc,bhng->bhgc', x_mid, slice_weights)
        slice_tokens = slice_tokens / (norm[..., None] + 1e-5)
        
        q_slice_token = self.to_q(slice_tokens)
        k_slice_token = self.to_k(slice_tokens)
        v_slice_token = self.to_v(slice_tokens)
        
        attn = torch.softmax(torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale, dim=-1)  # [B H G G]
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # [B H G D]
        
        if return_latent:
            latent = rearrange(out_slice_token, 'b h g d -> b g (h d)')
            if return_slice_weights:
                return latent, slice_weights
            return latent
        
        out_flat = torch.einsum('bhgd,bhng->bhnd', out_slice_token, slice_weights)
        out_flat = rearrange(out_flat, 'b h n d -> b n (h d)')
        out_flat = self.to_out(out_flat)  # [B, N, C]
        
        out = unpack_one(out_flat, ps, 'b * c')

        if return_slice_weights and not return_latent:
            return out, slice_weights
        return out


class Physics_Cross_Attention_plus(nn.Module):
    """Transolver ++ implementation"""

    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0., 
        slice_num: int = 64,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * heads
        
        # regression temperature
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )
        
        self.in_project_q = nn.Linear(dim, inner_dim)
        self.in_project_kv = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, return_slice_weights=False):
        # q shape: [B, N, C], kv shape: [B, G, C]
        assert len(q.shape) == 3 and len(kv.shape) == 3, "Input tensors must be 3-dimensional"
        
        q_mid = rearrange(self.in_project_q(q),'b n (h d) -> b h n d', h=self.heads)
        
        temperature = self.proj_temperature(q_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(q_mid), temperature)
        norm = reduce(slice_weights, 'b h n g -> b h g', 'sum')
        
        q_slice_tokens = torch.einsum('bhnd,bhng->bhgd', q_mid, slice_weights)
        q_slice_tokens = q_slice_tokens / (norm[..., None] + 1e-5)
        
        kv_slice_tokens = rearrange(self.in_project_kv(kv), 'b g (h d) -> b h g d', h=self.heads)
        
        q_slice_token = self.to_q(q_slice_tokens)
        k_slice_token = self.to_k(kv_slice_tokens)
        v_slice_token = self.to_v(kv_slice_tokens)
        
        attn = torch.softmax(torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale, dim=-1)  # [B H G G]
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # [B H G D]
        
        out_flat = torch.einsum('bhgd,bhng->bhnd', out_slice_token, slice_weights)
        out_flat = rearrange(out_flat, 'b h n d -> b n (h d)')
        out = self.to_out(out_flat)  # [B, N, C]
        
        if return_slice_weights:
            return out, slice_weights
        return out
    

class Cross_Attention(nn.Module):
    def __init__(
        self, 
        query_dim: int,
        context_dim: int,
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0.
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        
        inner_dim = dim_head * heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout_p = dropout

    def forward(self, q, kv, attn_mask=None):
        """
        Args:
            q: [B, N, query_dim]
            kv: [B, M, context_dim]
            attn_mask: [B, heads, N, M]
        """
        q = rearrange(self.to_q(q), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(kv), 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(self.to_v(kv), 'b m (h d) -> b h m d', h=self.heads)

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=64,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_plus(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver_cross_attn_block(nn.Module):
    """Transformer decoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=64,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Cross_Attention_plus(hidden_dim, heads=num_heads, dim_head=hidden_dim//num_heads, dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fq, fkv):
        fx = self.Attn(self.ln_q(fq), self.ln_kv(fkv)) + fq
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class FourierPositionalEmbedding(nn.Module):
    """
    Fourier Positional Embedding as described in
    Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains (Tancik et al. 2020)
    """
    
    def __init__(
        self,
        spatial_dim: int,
        embed_dim: int,
        sigma: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.sigma = sigma
        
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        
        num_frequencies = embed_dim // 2
        
        B = torch.randn(spatial_dim, num_frequencies) * sigma
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, coords):
        coords_proj = torch.matmul(coords, self.B)
        
        fourier_features = torch.cat([
            torch.sin(2 * math.pi * coords_proj),
            torch.cos(2 * math.pi * coords_proj)
        ], dim=-1)
        
        return fourier_features



class CrossAttentionTransformerLayer(nn.Module):
    def __init__(
        self, 
        query_dim: int, 
        context_dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0,
        mlp_ratio: int = 4,
        act='gelu',
    ):
        super().__init__()
        
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        
        self.attn = Cross_Attention(
            query_dim=query_dim, 
            context_dim=context_dim, 
            heads=heads, 
            dim_head=dim_head, 
            dropout=dropout
        )
        
        self.norm_mlp = nn.LayerNorm(query_dim)
        inner_dim = int(query_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, inner_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, kv, attn_mask=None):
        res = q
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out = self.attn(q_norm, kv_norm, attn_mask=attn_mask)
        x = res + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


# =======================================================
#             Transolver Neural Operator
# =======================================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class Physics_pooling(nn.Module):

    def __init__(
        self, 
        dim: int, 
        slice_num: int = 64,
    ):
        super().__init__()
        self.slice_num = slice_num
        
        # regression temperature
        self.bias = nn.Parameter(torch.ones([1, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        self.in_project_slice = nn.Linear(dim, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    def forward(self, x):
        assert x.ndim == 3, "Input must be of shape [B, N, C]"
        
        temperature = self.proj_temperature(x) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x), temperature)
        norm = reduce(slice_weights, 'b n g -> b g', 'sum')
        
        slice_tokens = torch.einsum('bnc,bng->bgc', x, slice_weights)
        slice_tokens = slice_tokens / (norm[..., None] + 1e-5)
        
        # to deslice, use deslice_token = torch.einsum('bgc,bng->bnc', slice_tokens, slice_weights)
        
        return slice_tokens, slice_weights


class Physics_unpooling(nn.Module):
    def forward(self, slice_tokens, slice_weights):
        # deslice
        out = torch.einsum('bgc,bng->bnc', slice_tokens, slice_weights)
        return out


class TokenTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, G, C]
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)  # [B, G, C]
        x = x + self.drop(a)
        x = x + self.mlp(self.ln2(x))
        return x


class TimeConditionedTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        last = self.adaLN_modulation[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, t_emb):
        mod = self.adaLN_modulation(t_emb) # [B, 1, 6*C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        h = self.ln1(x)
        h = h * (1 + scale_msa) + shift_msa
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa * a

        h = self.ln2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        m = self.mlp(h)
        x = x + gate_mlp * m
        return x


class TokenMixer(nn.Module):
    def __init__(self, dim, depth=2, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TokenTransformerBlock(dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, tokens):
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens


class TimeConditionedTokenMixer(nn.Module):
    def __init__(self, dim, depth=2, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TimeConditionedTransformerBlock(dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, tokens, t_emb):
        for blk in self.blocks:
            tokens = blk(tokens, t_emb)
        return tokens


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

class ChannelSkipGate(nn.Module):
    def __init__(self, dim: int, init_p: float = 0.1):
        super().__init__()
        gamma0 = torch.full((1, 1, dim), logit(init_p), dtype=torch.float32)
        self.gamma = nn.Parameter(gamma0)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # h, s: [B, g, C]
        gate = torch.sigmoid(self.gamma)  # [1, 1, C]
        return h + s * gate


class PhysicsOperator(nn.Module):
    """
    Multi-level tokens: N -> g0 -> g1 -> ... -> g_last (bottleneck)
    Decoder reverses:   g_last -> ... -> g0 -> N
    """

    def __init__(
        self,
        spatio_dim: int,
        c_in: int,
        c_out: int,
        width: int = 256,
        g_list=(512, 256, 128, 64),
        enc_depth: int = 2,
        dec_depth: int = 2,
        bottleneck_depth: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        gate_init_p: float = 0.1,
        final_mlp_layers: int = 0,           # 0 means just linear head
    ):
        super().__init__()
        self.spatio_dim = spatio_dim
        assert len(g_list) >= 1, "g_list must have at least one level"
        self.g_list = list(g_list)
        self.width = width

        self.in_proj = nn.Linear(c_in, width)
        self.pos_emb = FourierPositionalEmbedding(spatio_dim, width)

        # encoder
        self.pool0 = Physics_pooling(dim=width, slice_num=self.g_list[0])
        self.pools = nn.ModuleList([
            Physics_pooling(dim=width, slice_num=self.g_list[i+1])
            for i in range(len(self.g_list) - 1)])
        self.enc_mixers = nn.ModuleList([
            TokenMixer(dim=width, depth=enc_depth, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(len(self.g_list))])

        # bottleneck
        self.bottleneck = TokenMixer(dim=width, depth=bottleneck_depth, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio)

        # decoder
        self.unpool0 = Physics_unpooling()
        self.unpools = nn.ModuleList([
            Physics_unpooling()
            for _ in range(len(self.g_list) - 1)])
        self.dec_mixers = nn.ModuleList([
            TokenMixer(dim=width, depth=dec_depth, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(len(self.g_list))])

        # Gated skip for each g_i
        self.skip_gates = nn.ModuleList([
            ChannelSkipGate(dim=width, init_p=gate_init_p)
            for _ in range(len(self.g_list))
        ])

        # Optional small head MLP at N-resolution
        if final_mlp_layers > 0:
            layers = []
            for _ in range(final_mlp_layers):
                layers += [nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)]
            self.final_mlp = nn.Sequential(*layers)
        else:
            self.final_mlp = nn.Identity()

        self.out_proj = nn.Linear(width, c_out)

    def forward(self, field, coords):
        if field.ndim == 4:
            B, H, W, C = field.shape
            field = rearrange(field, 'b h w c -> b (h w) c').contiguous()
            coords = rearrange(coords, 'b h w d -> b (h w) d').contiguous()
            in_ndim = 4
        elif field.ndim == 5:
            B, Z, H, W, C = field.shape
            field = rearrange(field, 'b z h w c -> b (z h w) c').contiguous()
            coords = rearrange(coords, 'b z h w d -> b (z h w) d').contiguous()
            in_ndim = 5
        elif field.ndim == 3:
            in_ndim = 3
        else:
            raise ValueError("field must be [B,N,C] or [B,H,W,C] or [B,Z,H,W,C]")

        x = self.in_proj(field)  # [B, N, width]
        x = x + self.pos_emb(coords)

        tok0, w0 = self.pool0(x)  # tok0 [B,g0,C], w0 [B,N,g0]

        skip_tokens = []
        w_down = []

        # level 0..L-1
        h = tok0
        for i in range(len(self.g_list)):
            h = self.enc_mixers[i](h)            # [B, g_i, C]
            skip_tokens.append(h)

            if i < len(self.g_list) - 1:
                h_next, w = self.pools[i](h)     # w: [B, g_i, g_{i+1}]
                w_down.append(w)
                h = h_next

        h = self.bottleneck(h)  # [B, g_last, C]

        # Decoder up path (reverse levels)
        for i in reversed(range(len(self.g_list))):
            if i < len(self.g_list) - 1:
                w = w_down[i]
                h = self.unpools[i](h, w)  # h: [B, g_i, C]

            h = self.skip_gates[i](h, skip_tokens[i])  # [B, g_i, C]
            h = self.dec_mixers[i](h)  # [B, g_i, C]

        # g0 -> N
        out = self.unpool0(h, w0)  # [B, N, C]
        out = self.final_mlp(out)
        out = self.out_proj(out)   # [B, N, C_out]
        
        if in_ndim == 4:
            out = rearrange(out, 'b (h w) c -> b h w c', h=H, w=W).contiguous()
        elif in_ndim == 5:
            out = rearrange(out, 'b (z h w) c -> b z h w c', z=Z, h=H, w=W).contiguous()
        return out


class TimeConditionedPhysicsOperator(nn.Module):
    """
    Multi-level tokens: N -> g0 -> g1 -> ... -> g_last (bottleneck)
    Decoder reverses:   g_last -> ... -> g0 -> N
    """

    def __init__(
        self,
        spatio_dim: int,
        c_in: int,
        c_out: int,
        width: int = 256,
        g_list=(512, 256, 128, 64),
        enc_depth: int = 2,
        dec_depth: int = 2,
        bottleneck_depth: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        gate_init_p: float = 0.1,
        final_mlp_layers: int = 0,           # 0 means just linear head
    ):
        super().__init__()
        self.spatio_dim = spatio_dim
        assert len(g_list) >= 1, "g_list must have at least one level"
        self.g_list = list(g_list)
        self.width = width

        self.in_proj = nn.Linear(c_in, width)
        self.pos_emb = FourierPositionalEmbedding(spatio_dim, width)
        self.t_emb = TimestepEmbedder(hidden_size=width)

        # encoder
        self.pool0 = Physics_pooling(dim=width, slice_num=self.g_list[0])
        self.pools = nn.ModuleList([
            Physics_pooling(dim=width, slice_num=self.g_list[i+1])
            for i in range(len(self.g_list) - 1)])
        self.enc_mixers = nn.ModuleList([
            TimeConditionedTokenMixer(dim=width, depth=enc_depth, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(len(self.g_list))])

        # bottleneck
        self.bottleneck = TimeConditionedTokenMixer(dim=width, depth=bottleneck_depth, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio)

        # decoder
        self.unpool0 = Physics_unpooling()
        self.unpools = nn.ModuleList([
            Physics_unpooling()
            for _ in range(len(self.g_list) - 1)])
        self.dec_mixers = nn.ModuleList([
            TimeConditionedTokenMixer(dim=width, depth=dec_depth, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio)
            for _ in range(len(self.g_list))])

        # Gated skip for each g_i
        self.skip_gates = nn.ModuleList([
            ChannelSkipGate(dim=width, init_p=gate_init_p)
            for _ in range(len(self.g_list))
        ])

        # Optional small head MLP at N-resolution
        if final_mlp_layers > 0:
            layers = []
            for _ in range(final_mlp_layers):
                layers += [nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)]
            self.final_mlp = nn.Sequential(*layers)
        else:
            self.final_mlp = nn.Identity()

        self.out_proj = nn.Linear(width, c_out)

    def forward(self, field, coords, t):
        if field.ndim == 4:
            B, H, W, C = field.shape
            field = rearrange(field, 'b h w c -> b (h w) c').contiguous()
            coords = rearrange(coords, 'b h w d -> b (h w) d').contiguous()
            in_ndim = 4
        elif field.ndim == 5:
            B, Z, H, W, C = field.shape
            field = rearrange(field, 'b z h w c -> b (z h w) c').contiguous()
            coords = rearrange(coords, 'b z h w d -> b (z h w) d').contiguous()
            in_ndim = 5
        elif field.ndim == 3:
            in_ndim = 3
        else:
            raise ValueError("field must be [B,N,C] or [B,H,W,C] or [B,Z,H,W,C]")
        
        t_emb = self.t_emb(t).unsqueeze(1)  # [B, 1, width]
        x = self.in_proj(field)  # [B, N, width]
        x = x + self.pos_emb(coords)

        tok0, w0 = self.pool0(x)  # tok0 [B,g0,C], w0 [B,N,g0]

        skip_tokens = []
        w_down = []

        # level 0..L-1
        h = tok0
        for i in range(len(self.g_list)):
            h = self.enc_mixers[i](h, t_emb)            # [B, g_i, C]
            skip_tokens.append(h)

            if i < len(self.g_list) - 1:
                h_next, w = self.pools[i](h)     # w: [B, g_i, g_{i+1}]
                w_down.append(w)
                h = h_next

        h = self.bottleneck(h, t_emb)  # [B, g_last, C]

        # Decoder up path (reverse levels)
        for i in reversed(range(len(self.g_list))):
            if i < len(self.g_list) - 1:
                w = w_down[i]
                h = self.unpools[i](h, w)  # h: [B, g_i, C]

            h = self.skip_gates[i](h, skip_tokens[i])  # [B, g_i, C]
            h = self.dec_mixers[i](h, t_emb)  # [B, g_i, C]

        # g0 -> N
        out = self.unpool0(h, w0)  # [B, N, C]
        out = self.final_mlp(out)
        out = self.out_proj(out)   # [B, N, C_out]
        
        if in_ndim == 4:
            out = rearrange(out, 'b (h w) c -> b h w c', h=H, w=W).contiguous()
        elif in_ndim == 5:
            out = rearrange(out, 'b (z h w) c -> b z h w c', z=Z, h=H, w=W).contiguous()
        return out


