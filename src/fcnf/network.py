import torch
from torch import nn
import numpy as np
import random
from einops import rearrange, repeat
import math
import copy

from .components import *
from .initialization import init_weights_uniform_siren_scale, init_weights_trunc_normal
from .fsq import FSQ
from .vector_quantize import VectorQuantize


class SIRENencoder(nn.Module):
    '''
    siren encoder
    input: coords, vector values at the coords
    output: latent code
    '''
    def __init__(self, in_coord_features, in_field_features, out_latent_features, num_hidden_layers, hidden_features,
                 nonlinearity='sine', weight_init=None, premap_mode=None, omega_0=30, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if self.premap_mode is not None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode=premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features  
        
        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
        self.nl = copy.deepcopy(self.nl)  # ! deepcopy to avoid weight sharing
        
        if nonlinearity == 'sine':
            if isinstance(self.nl.w0, torch.nn.Parameter):
                with torch.no_grad():
                    self.nl.w0.fill_(float(omega_0))
            else:
                self.nl.w0 = float(omega_0)

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init  # those are default init funcs 

        # create the net for the nf 
        self.nf_net = nn.ModuleList([BatchLinear(in_coord_features+in_field_features, hidden_features)] + 
                                  [BatchLinear(hidden_features, hidden_features) for _ in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features, out_latent_features)])
        
        # self.physics_pooling = Physics_Attention(dim=hidden_features, heads=8, dim_head=hidden_features//8, dropout=0.0, slice_num=64, slice_only=True)
        # self.physics_pooling.apply(init_weights_trunc_normal)

        if self.weight_init is not None:
            self.nf_net.apply(self.weight_init(omega_0))
        self.last_norm = nn.LayerNorm(out_latent_features, elementwise_affine=False)
        self.scale = nn.Parameter(torch.randn(out_latent_features)*0.01, requires_grad=True)  # ! this is a very important scale parameter that we must add

        # if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
        #     self.nf_net[0].apply(first_layer_init) # DO NOT INITIALIZE FIRST LAYER WITH SIREN INIT
    
    def forward(self, coords: torch.Tensor, field_values: torch.Tensor) -> torch.Tensor:
        # coords: <t,h,w,c_coord> or <t,n,c_coords>
        # field_values: <t,h,w,c_field> or <t,n,c_field>
        
        # premap
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords
        
        # concat coord and field values 
        x = torch.cat([x, field_values], dim=-1)

        # pass it through  the nf network 
        for i in range(len(self.nf_net) -1):
            x = self.nl(self.nf_net[i](x))

        if x.ndim == 4:
            # x: <t,h,w,c_latent>
            b, h, w, c = x.shape
            x = x.view(b, h*w, c)
        elif x.ndim == 3:
            # x: <t,n,c_latent>
            pass
        else:
            raise ValueError(f"Expected output latent dimension 3 or 4, but got {x.dim()}")
        
        x = torch.mean(x, dim=1)  # <t,c_latent>
        # x = self.physics_pooling(x, return_latent=True)  # <t,64,c_latent>
        # print(f">>> after pooling, x shape: {x.shape}")
        
        x = self.nf_net[-1](x)
        x = self.last_norm(x)
        x = x * self.scale.unsqueeze(0)

        return x


class SIRENdecoder(nn.Module):
    '''
    siren decoder with full projection conditioning
    input: coords, latents
    output: signal value at the coords
    '''
    def __init__(self, in_coord_features, in_latent_features, out_field_features, num_hidden_layers, hidden_features,
                 nonlinearity='sine', weight_init=None, premap_mode=None, omega_0=30, rank=32, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        self.rank = rank
        self.hidden_features = hidden_features
        self.in_coord_features = in_coord_features
        self.in_latent_features = in_latent_features
        if self.premap_mode is not None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode=premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features  
        
        # self.fourier_transform = GaussianFourierFeatureTransform(in_coord_features, int(hidden_features//2), 10)   

        self.first_layer_init = None
        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
        self.nl = copy.deepcopy(self.nl)
        
        if nonlinearity == 'sine':
            if isinstance(self.nl.w0, torch.nn.Parameter):
                with torch.no_grad():
                    self.nl.w0.fill_(float(omega_0))
            else:
                self.nl.w0 = float(omega_0)
        
        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        # create the net for the nf 
        self.nf_net = nn.ModuleList([LipschitzLinear(in_coord_features,hidden_features)] + 
                                  [LipschitzLinear(hidden_features,hidden_features) for _ in range(num_hidden_layers)] + 
                                  [LipschitzLinear(hidden_features,out_field_features)])

        # (1) Full projection
        # self.hw_net = nn.ModuleList([BatchLinear(in_latent_features, in_coord_features*hidden_features, bias=False)]+
        #                             [BatchLinear(in_latent_features, hidden_features*hidden_features, bias=False) for _ in range(num_hidden_layers)])
        # self.hb_net = nn.ModuleList([BatchLinear(in_latent_features, hidden_features, bias=False) for _ in range(num_hidden_layers+1)])
        # (2) FiLM
        # self.film_net = nn.ModuleList([BatchLinear(in_latent_features, 2 * hidden_features, bias=False)
        #                             for _ in range(num_hidden_layers + 1)])
        # (3) Low-rank full projection
        self.hw_net_A = nn.ModuleList([LipschitzLinear(in_latent_features, in_coord_features * rank, bias=False)] +
                                      [LipschitzLinear(in_latent_features, hidden_features * rank, bias=False) for _ in range(num_hidden_layers)])
        self.hw_net_B = nn.ModuleList([LipschitzLinear(in_latent_features, hidden_features * rank, bias=False)] +
                                      [LipschitzLinear(in_latent_features, hidden_features * rank, bias=False) for _ in range(num_hidden_layers)])
        self.hb_net = nn.ModuleList([LipschitzLinear(in_latent_features, hidden_features, bias=False) for _ in range(num_hidden_layers+1)])

        if self.weight_init is not None:
            self.nf_net.apply(self.weight_init(omega_0))

        # if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
        #     self.nf_net[0].apply(first_layer_init)
        self.init_hyper_layer()     

    def init_hyper_layer(self):
        # init weights
        self.hw_net_A.apply(init_weights_trunc_normal)
        # self.hw_net_B.apply(init_weights_trunc_normal)
        for m in self.hw_net_B:
            if isinstance(m, BatchLinear) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
        self.hb_net.apply(init_weights_trunc_normal)
        # for m in self.film_net:
        #     if isinstance(m, BatchLinear) or isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 0)

    def forward(self, coords: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        t_size = latents.shape[0]
        if coords.ndim == 4:
            # coords: <t,h,w,c_coord> or <1,h,w,c_coords>
            # latents: <t,h,w,c_latent> or <t,1,1,c_latent>
            dim = 2
        elif coords.ndim == 3:
            # coords: <t,n,c_coord> or <1,n,c_coords>
            # latents: <t,n,c_latent> or <t,1,c_latent>
            dim = 1
        else:
            raise ValueError(f"Expected input coord dimension 3 or 4, but got {coords.dim()}")
        
        while latents.ndim < coords.ndim:
            latents = latents.unsqueeze(1)
        assert len(coords.shape) == len(latents.shape),\
            f"Coord dim {coords.dim()} and latent dim {latents.dim()} not compatible"
        
        # premap
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords
        
        # fourier transform
        # x = self.fourier_transform(coords)

        # pass it through  the nf network 
        for i in range(len(self.nf_net) -1):
            
            # (1) Full projection
            # reshape_dims = (t_size,) + (1,) * dim + self.nf_net[i].weight.shape
            # x = (
            #         self.nf_net[i](x) + 
            #         # torch.einsum(
            #         #     '...i,...ji->...j', 
            #         #     x, 
            #         #     # self.hw_net[i](latents).reshape((t_size, 1, 1)+self.nf_net[i].weight.shape)
            #         #     self.hw_net[i](latents).reshape(reshape_dims)
            #         # )
            #         + self.hb_net[i](latents)
            #     )
            
            # (2) FiLM
            # h = self.nf_net[i](x)
            # film = self.film_net[i](latents)
            # gamma, beta = film.chunk(2, dim=-1)
            # x = gamma * h + beta
            
            # (3) Low-rank full projection
            d_out, d_in = self.nf_net[i].weight.shape
            A = self.hw_net_A[i](latents).reshape((t_size,) + (1,) * dim + (d_in, self.rank))
            B = self.hw_net_B[i](latents).reshape((t_size,) + (1,) * dim + (self.rank, d_out))
            b = self.hb_net[i](latents).reshape((t_size,) + (1,) * dim + (d_out,))

            x = self.nf_net[i](x) + torch.einsum('...i,...ir,...ro->...o', x, A, B) + b
            
            x = self.nl(x)

        x = self.nf_net[-1](x)

        return x 
    
    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def lipschitz_loss(self):
        # loss = 1.0
        # for m in self.modules():
        #     if isinstance(m, LipschitzLinear):
        #         loss = loss * m.get_c()
        # return loss
        cs = [
            m.get_c()
            for m in self.modules()
            if isinstance(m, LipschitzLinear)
            ]
        c = torch.stack(cs)
        return torch.exp(torch.mean(torch.log(c + 1e-12)))


class SIRENautoencoder(nn.Module):
    '''
    siren autoencoder with full projection conditioning
    input: coords, vector values at the coords
    output: signal value at the coords
    '''
    def __init__(self, 
                 coord_features, field_features,
                 latent_features,
                 encoder_num_hidden_layers, encoder_hidden_features,
                 decoder_num_hidden_layers, decoder_hidden_features,
                 encoder_nonlinearity='sine', decoder_nonlinearity='sine',
                 encoder_weight_init=None, decoder_weight_init=None,
                 encoder_premap_mode=None, decoder_premap_mode=None,
                 encoder_omega_0=30, decoder_omega_0=30,
                 quantizer_cfg=None,
                 **kwargs):
        super().__init__()
        self.encoder = SIRENencoder(
            coord_features,
            field_features,
            latent_features,
            encoder_num_hidden_layers,
            encoder_hidden_features,
            nonlinearity=encoder_nonlinearity,
            weight_init=encoder_weight_init,
            premap_mode=encoder_premap_mode,
            omega_0=encoder_omega_0,
            **kwargs
        )
        self.decoder = SIRENdecoder(
            coord_features,
            latent_features,
            field_features,
            decoder_num_hidden_layers,
            decoder_hidden_features,
            nonlinearity=decoder_nonlinearity,
            weight_init=decoder_weight_init,
            premap_mode=decoder_premap_mode,
            omega_0=decoder_omega_0,
            **kwargs
        )
        if quantizer_cfg is not None:
            if "codebook_size" in quantizer_cfg:
                self.quantizer = VectorQuantize(**quantizer_cfg)
            elif "levels" in quantizer_cfg:
                self.quantizer = FSQ(**quantizer_cfg)
            else:
                raise ValueError("Unknown quantizer configuration")
        self.quantizer_cfg = quantizer_cfg
    
    def encode(self, coords, field_values):
        latents = self.encoder(coords, field_values)
        indices = None
        loss = None
        
        if self.quantizer_cfg is not None:
            if isinstance(self.quantizer, VectorQuantize):
                latents, indices, loss = self.quantizer(latents)
                loss = loss.mean()
            elif  isinstance(self.quantizer, FSQ):
                latents, indices = self.quantizer(latents)
            
        return latents, indices, loss
    
    def decode(self, coords, latents):
        outputs = self.decoder(coords, latents)
        return outputs
    
    def forward(self, input_coords, input_field_values, query_coords=None, 
                return_latents=False, return_indices=False):

        latents, indices, loss = self.encode(input_coords, input_field_values)
        # print(f">>> latents shape: {latents.shape}, indices shape: {None if indices is None else indices.shape}")
        # latents = torch.mean(latents, dim=1, keepdim=True)
        # print(f"latents after max shape: {latents.shape}")
        # print(f"latent min: {latents.min().item()}, max: {latents.max().item()}, mean: {latents.mean().item()}")
        
        if query_coords is None:
            query_coords = input_coords
        
        outputs = self.decode(query_coords, latents)
        
        if return_latents and return_indices:
            return outputs, latents, indices
        if return_latents:
            return outputs, latents, loss
        if return_indices:
            return outputs, indices
        return outputs



class SIRENdecoder_decomp(nn.Module):
    '''
    siren decoder with full projection conditioning
    input: coords, latents
    output: signal value at the coords
    '''
    def __init__(self, in_coord_features, in_latent_features, out_field_features, num_hidden_layers, hidden_features,
                 nonlinearity='sine', weight_init=None, premap_mode=None, omega_0=30, rank=-1, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        self.rank = rank
        self.hidden_features = hidden_features
        self.in_coord_features = in_coord_features
        self.in_latent_features = in_latent_features
        if self.premap_mode is not None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode=premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features  
        
        # self.fourier_transform = GaussianFourierFeatureTransform(in_coord_features, int(hidden_features//2), 10)   

        self.first_layer_init = None
        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
        self.nl = copy.deepcopy(self.nl)
        
        if nonlinearity == 'sine':
            if isinstance(self.nl.w0, torch.nn.Parameter):
                with torch.no_grad():
                    self.nl.w0.fill_(float(omega_0))
            else:
                self.nl.w0 = float(omega_0)
        
        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        # create the net for the nf 
        self.nf_net = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for _ in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features,out_field_features)])

        self.hw_net_U = nn.ParameterList()
        self.hw_net_V = nn.ParameterList()
        for i in range(num_hidden_layers+1):
            in_dim = in_coord_features if i ==0 else hidden_features
            out_dim = hidden_features
            
            self.hw_net_U.append(nn.Parameter(torch.empty(out_dim, rank)))
            self.hw_net_V.append(nn.Parameter(torch.empty(in_dim, rank)))
            
        self.hb_net = nn.ModuleList([BatchLinear(in_latent_features, hidden_features, bias=False) for _ in range(num_hidden_layers+1)])
            
        if self.weight_init is not None:
            self.nf_net.apply(self.weight_init(omega_0))
        
        self.init_ortho()
        # if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
        #     self.nf_net[0].apply(first_layer_init)
    
    def init_ortho(self):
        with torch.no_grad():
            for u_param in self.hw_net_U:
                nn.init.orthogonal_(u_param, gain=0.1)  
            for v_param in self.hw_net_V:
                nn.init.orthogonal_(v_param, gain=0.0001)

    def forward(self, coords: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        t_size = latents.shape[0]
        if coords.ndim == 4:
            # coords: <t,h,w,c_coord> or <1,h,w,c_coords>
            # latents: <t,h,w,c_latent> or <t,1,1,c_latent>
            dim = 2
        elif coords.ndim == 3:
            # coords: <t,n,c_coord> or <1,n,c_coords>
            # latents: <t,n,c_latent> or <t,1,c_latent>
            dim = 1
        else:
            raise ValueError(f"Expected input coord dimension 3 or 4, but got {coords.dim()}")
        
        # while latents.ndim < coords.ndim:
        #     latents = latents.unsqueeze(1)
        # assert len(coords.shape) == len(latents.shape),\
        #     f"Coord dim {coords.dim()} and latent dim {latents.dim()} not compatible"
        assert len(latents.shape) == 2, \
            f"Latents should have shape <t_size, latent_features>"
        
        # premap
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords
        
        # fourier transform
        # x = self.fourier_transform(coords)

        # pass it through  the nf network 
        for i in range(len(self.nf_net) -1):
            U = self.hw_net_U[i]  # [out_dim, rank]
            V = self.hw_net_V[i]  # [in_dim, rank]

            delta_W = torch.einsum('or,br,ir->boi', U, latents, V)
            b = self.hb_net[i](latents)
            while b.ndim < x.ndim: b = b.unsqueeze(1)
            x = self.nf_net[i](x) + torch.einsum('b...i,boi->b...o', x, delta_W) + b

            x = self.nl(x)

        x = self.nf_net[-1](x)

        return x 
# =======================================================
#                MLP Autoencoder
# =======================================================
class MLPencoder(nn.Module):
    def __init__(self, coord_features, field_features, latent_features, num_hidden_layers, hidden_features, act='gelu'):
        super().__init__()
        
        self.in_proj = nn.Linear(field_features, hidden_features)
        self.pos_emb = FourierPositionalEmbedding(coord_features, hidden_features)
        
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                act()
            )
            for _ in range(num_hidden_layers)
        ])
        
        self.out_proj = nn.Linear(hidden_features, latent_features)
        
        # self.scale = nn.Parameter(torch.randn(latent_features)*0.01, requires_grad=True)  # ! this is a very important scale parameter that we must add
    
    def forward(self, coords: torch.Tensor, field_values: torch.Tensor) -> torch.Tensor:
        
        x = self.in_proj(field_values) + self.pos_emb(coords)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = rearrange(x, 'b ... d -> b ( ... ) d')
        
        x = torch.mean(x, dim=1)
        # print(f"*** before proj, encoder output min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
        
        x = self.out_proj(x)
        # x = x * self.scale.unsqueeze(0)
        # print(f">>> encoder output min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
        x = F.tanh(x)
        # print(f"%%% after act, encoder output min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")

        return x


class MLPdecoder(nn.Module):
    def __init__(self, coord_features, latent_features, field_features, num_hidden_layers, hidden_features, act='gelu', use_lipschitz=False):
        super().__init__()
        
        self.in_proj = nn.Linear(latent_features, hidden_features) if not use_lipschitz else LipschitzLinear(latent_features, hidden_features)
        self.pos_emb = FourierPositionalEmbedding(coord_features, hidden_features)
        
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features) if not use_lipschitz else LipschitzLinear(hidden_features, hidden_features),
                act()
            )
            for _ in range(num_hidden_layers)
        ])
        
        self.out_proj = nn.Linear(hidden_features, field_features) if not use_lipschitz else LipschitzLinear(hidden_features, field_features)
    
    def forward(self, coords: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        while latents.ndim < coords.ndim:
            latents = latents.unsqueeze(1)
        
        x = self.in_proj(latents) + self.pos_emb(coords)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.out_proj(x)

        return x
    
    def lipschitz_loss(self):
        # loss = 1.0
        # for m in self.modules():
        #     if isinstance(m, LipschitzLinear):
        #         loss = loss * m.get_c()
        # return loss
        cs = [
            m.get_c()
            for m in self.modules()
            if isinstance(m, LipschitzLinear)
            ]
        c = torch.stack(cs)
        return torch.exp(torch.mean(torch.log(c + 1e-12)))


class MLPautoencoder(nn.Module):
    def __init__(self, 
                 coord_features, field_features,
                 latent_features,
                 encoder_num_hidden_layers, encoder_hidden_features,
                 decoder_num_hidden_layers, decoder_hidden_features,
                 act='gelu',
                 quantizer_cfg=None,
                 use_lipschitz=False,
                 ):
        super().__init__()
        self.encoder = MLPencoder(
            coord_features,
            field_features,
            latent_features,
            encoder_num_hidden_layers,
            encoder_hidden_features,
            act=act
        )
        self.decoder = MLPdecoder(
            coord_features,
            latent_features,
            field_features,
            decoder_num_hidden_layers,
            decoder_hidden_features,
            act=act,
            use_lipschitz=use_lipschitz
        )
        if quantizer_cfg is not None:
            if "codebook_size" in quantizer_cfg:
                self.quantizer = VectorQuantize(**quantizer_cfg)
            elif "levels" in quantizer_cfg:
                self.quantizer = FSQ(**quantizer_cfg)
            else:
                raise ValueError("Unknown quantizer configuration")
        self.quantizer_cfg = quantizer_cfg
        
        self.apply(self.init_layers)
    
    def init_layers(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def encode(self, coords, field_values):
        latents = self.encoder(coords, field_values)
        indices = None
        loss = None
        
        if self.quantizer_cfg is not None:
            if isinstance(self.quantizer, VectorQuantize):
                latents, indices, loss = self.quantizer(latents)
                loss = loss.mean()
            elif  isinstance(self.quantizer, FSQ):
                latents, indices = self.quantizer(latents)
            
        return latents, indices, loss
    
    def decode(self, coords, latents):
        outputs = self.decoder(coords, latents)
        return outputs
    
    def forward(self, input_coords, input_field_values, query_coords=None, 
                return_latents=False, return_indices=False):

        latents, indices, loss = self.encode(input_coords, input_field_values)
        
        if query_coords is None:
            query_coords = input_coords
        
        outputs = self.decode(query_coords, latents)
        
        if return_latents and return_indices:
            return outputs, latents, indices
        if return_latents:
            return outputs, latents, loss
        if return_indices:
            return outputs, indices
        return outputs
            



# =======================================================
#                Transolver Autoencoder
# =======================================================


class TransolverEncoder(nn.Module):
    def __init__(self,
                 space_dim=1,
                 fun_dim=1,
                 out_dim=128,
                 slice_num=64,
                 n_layers=5,
                 n_hidden=128,
                 dropout=0,
                 n_head=8,
                 mlp_ratio=4,
                 act='gelu',
                 ):
        super().__init__()

        self.func_preprocess = MLP(fun_dim, n_hidden*2, n_hidden, n_layers=0, res=False, act=act)
        self.coord_preprocess = FourierPositionalEmbedding(space_dim, n_hidden)

        self.n_hidden = n_hidden
        self.space_dim = space_dim

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio, out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_layers -1))
                                     for _ in range(n_layers)])
        
        self.slice_layer = Physics_Attention_plus(dim=out_dim,
                                            heads=n_head,
                                            dim_head=out_dim//n_head,
                                            dropout=dropout,
                                            slice_num=slice_num,
                                            slice_only=True)
        
        self.apply(init_weights_trunc_normal)
    
    def forward(self, coords, func):
        
        func = self.func_preprocess(func)
        coords = self.coord_preprocess(coords)
        
        func_flat, ps = pack_one(func, 'b * c')
        coords_flat, _ = pack_one(coords, 'b * d')
        
        fx = func_flat + coords_flat
        
        for block in self.blocks:
            fx = block(fx)

        latent = self.slice_layer(fx, return_latent=True)  # [B, G, D]
        # latent = torch.mean(fx, dim=1)  # [B, D]
        
        return latent


class TransolverDecoder(nn.Module):
    def __init__(self,
                 space_dim=1,
                 latent_dim=128,
                 out_dim=1,
                 slice_num=64,
                 n_cross_layers=2,
                 n_mlp_layers=3,
                 n_hidden=128,
                 dropout=0,
                 n_head=8,
                 mlp_ratio=4,
                 act='gelu',
                 ):
        super().__init__()

        self.coord_preprocess = FourierPositionalEmbedding(space_dim, n_hidden)
        self.latent_proj = nn.Linear(latent_dim, n_hidden) if latent_dim != n_hidden else nn.Identity()
        
        self.cross_attn_blocks = nn.ModuleList([Transolver_cross_attn_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio, out_dim=out_dim if n_mlp_layers==0 else None,
                                                      slice_num=slice_num,
                                                      last_layer=(_ == n_cross_layers -1 and n_mlp_layers==0))
                                     for _ in range(n_cross_layers)])
        
        self.mlp_layers = MLP(n_hidden, n_hidden*2, out_dim, n_layers=n_mlp_layers, res=True, act=act) if n_mlp_layers >0 else nn.Identity()
        
        self.apply(init_weights_trunc_normal)
    
    def forward(self, coords, latent):
        
        # projection
        latent = self.latent_proj(latent)  # [B, G, D]
        coords = self.coord_preprocess(coords)  # [B, N, D]
        
        # sanity check
        if coords.ndim == 4:
            b, h, w, c = coords.shape
            query = coords.view(b, h*w, c).contiguous()
        elif coords.ndim ==5:
            b, d1, d2, d3, c = coords.shape
            query = coords.view(b, d1*d2*d3, c).contiguous()
        else:
            query = coords  # [B, N, D]
        
        # cross attention blocks
        for block in self.cross_attn_blocks:
            query = block(query, latent)  # [B, N, D]
        query = self.mlp_layers(query)
        
        # sanity reshape
        if coords.ndim ==4:
            query = query.view(b, h, w, -1).contiguous()
        elif coords.ndim ==5:
            query = query.view(b, d1, d2, d3, -1).contiguous()
            
        return query
    

class TransolverAutoencoder(nn.Module):
    def __init__(self,
                 coord_features,
                 field_features,
                 latent_features,
                 encoder_slice_num=64,
                 encoder_n_layers=4,
                 encoder_n_hidden=128,
                 decoder_slice_num=64,
                 decoder_n_cross_layers=4,
                 decoder_n_mlp_layers=3,
                 decoder_n_hidden=128,
                 encoder_dropout=0,
                 decoder_dropout=0,
                 encoder_n_head=8,
                 decoder_n_head=8,
                 encoder_mlp_ratio=4,
                 decoder_mlp_ratio=4,
                 encoder_act='gelu',
                 decoder_act='gelu',
                 quantizer_cfg=None,
                 ):
        super().__init__()
        self.encoder = TransolverEncoder(
            space_dim=coord_features,
            fun_dim=field_features,
            out_dim=latent_features,
            slice_num=encoder_slice_num,
            n_layers=encoder_n_layers,
            n_hidden=encoder_n_hidden,
            dropout=encoder_dropout,
            n_head=encoder_n_head,
            mlp_ratio=encoder_mlp_ratio,
            act=encoder_act
        )
        self.decoder = TransolverDecoder(
            space_dim=coord_features,
            latent_dim=latent_features,
            out_dim=field_features,
            slice_num=decoder_slice_num,
            n_cross_layers=decoder_n_cross_layers,
            n_mlp_layers=decoder_n_mlp_layers,
            n_hidden=decoder_n_hidden,
            dropout=decoder_dropout,
            n_head=decoder_n_head,
            mlp_ratio=decoder_mlp_ratio,
            act=decoder_act
        )
        
        if quantizer_cfg is not None:
            if "codebook_size" in quantizer_cfg:
                self.quantizer = VectorQuantize(**quantizer_cfg)
            elif "levels" in quantizer_cfg:
                self.quantizer = FSQ(**quantizer_cfg)
            else:
                raise ValueError("Unknown quantizer configuration")
        self.quantizer_cfg = quantizer_cfg
    
    def encode(self, coords, field_values):
        latents = self.encoder(coords, field_values)
        indices = None
        loss = None
        
        if self.quantizer_cfg is not None:
            if isinstance(self.quantizer, VectorQuantize):
                latents, indices, loss = self.quantizer(latents)
                loss = loss.mean()
            elif  isinstance(self.quantizer, FSQ):
                latents, indices = self.quantizer(latents)
                
        return latents, indices, loss
    
    def decode(self, coords, latents):
        outputs = self.decoder(coords, latents)
        return outputs
    
    def forward(self, input_coords, input_field_values, query_coords=None, 
                return_latents=False, return_indices=False):

        latents, indices, loss = self.encode(input_coords, input_field_values)
        
        if query_coords is None:
            query_coords = input_coords
        
        outputs = self.decode(query_coords, latents)
        
        if return_latents and return_indices:
            return outputs, latents, indices
        if return_latents:
            return outputs, latents, loss
        if return_indices:
            return outputs, indices
        return outputs



# =======================================================
#               Perceiver style Autoencoder
# =======================================================


class PerceiverEncoder(nn.Module):
    def __init__(self, coord_features, field_features,
                 num_latents=256, latent_dim=16,
                 hidden_features=128, mlp_ratio=4, n_head=8,
                 n_layers=4, dropout=0, act='gelu',
                 need_slice=False, slice_num=None):
        super().__init__()
        
        self.need_slice = need_slice
        self.slice_num = slice_num
        
        self.func_preprocess = MLP(field_features, hidden_features*2, hidden_features, n_layers=0, res=False, act=act)
        self.coord_preprocess = FourierPositionalEmbedding(coord_features, hidden_features)
        self.fuse = nn.Linear(hidden_features*2, hidden_features)
        
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_features))
        
        self.blocks = nn.ModuleList([CrossAttentionTransformerLayer(query_dim=hidden_features,
                                                                    context_dim=hidden_features,
                                                                    heads=n_head,
                                                                    dim_head=hidden_features//n_head,
                                                                    mlp_ratio=mlp_ratio,
                                                                    dropout=dropout,
                                                                    act=act) for _ in range(n_layers)])
        
        self.latents_proj = nn.Linear(hidden_features, latent_dim)
        if need_slice:
            assert slice_num is not None, "slice_num must be provided when need_slice is True"
            self.slice_layer = Physics_Attention_plus(dim=hidden_features,
                                                heads=n_head,
                                                dim_head=hidden_features//n_head,
                                                dropout=dropout,
                                                slice_num=slice_num,
                                                slice_only=True)
    
    def forward(self, coords, func):
        
        func = self.func_preprocess(func)
        coords = self.coord_preprocess(coords)
        
        func_flat = rearrange(func, 'b ... c -> b ( ... ) c')
        coords_flat = rearrange(coords, 'b ... d -> b ( ... ) d')
        
        fx = self.fuse(torch.cat([func_flat, coords_flat], dim=-1))
        
        b = fx.shape[0]
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        
        for block in self.blocks:
            latents = block(latents, fx)
        
        if self.need_slice:
            latents = self.slice_layer(latents, return_latent=True)  # [B, slice_num, hidden_features]
        
        latents = self.latents_proj(latents)  # [B, num_latents or slice_num, latent_dim]
        
        return latents


class PerceiverDecoder(nn.Module):
    def __init__(self, coord_features, latent_dim, field_features,
                 hidden_features=128, mlp_ratio=4, n_head=8,
                 n_layers=4, dropout=0, act='gelu',
                 need_slice=False, slice_num=None):
        super().__init__()
        
        self.coord_preprocess = FourierPositionalEmbedding(coord_features, hidden_features)
        self.latent_proj = nn.Linear(latent_dim, hidden_features) if latent_dim != hidden_features else nn.Identity()
        
        self.blocks = nn.ModuleList([CrossAttentionTransformerLayer(query_dim=hidden_features,
                                                                    context_dim=hidden_features,
                                                                    heads=n_head,
                                                                    dim_head=hidden_features//n_head,
                                                                    mlp_ratio=mlp_ratio,
                                                                    dropout=dropout,
                                                                    act=act) for _ in range(n_layers)])
        
        self.need_slice = need_slice
        self.slice_num = slice_num
        if need_slice:
            assert slice_num is not None, "slice_num must be provided when need_slice is True"
            self.slice_layer = Physics_Attention_plus(dim=hidden_features,
                                                heads=n_head,
                                                dim_head=hidden_features//n_head,
                                                dropout=dropout,
                                                slice_num=slice_num,
                                                slice_only=True)
            self.slice_head = n_head
        
        self.out_proj = nn.Linear(hidden_features, field_features)
        
    def forward(self, coords, latent):
        
        # projection
        latent = self.latent_proj(latent)  # [B, N_latent, D]
        coords = self.coord_preprocess(coords)  # [B, N_coords, D]
        
        # sanity check
        if coords.ndim == 4:
            b, h, w, c = coords.shape
            query = coords.view(b, h*w, c).contiguous()
        elif coords.ndim ==5:
            b, d1, d2, d3, c = coords.shape
            query = coords.view(b, d1*d2*d3, c).contiguous()
        else:
            query = coords  # [B, N, D]
        
        slice_weights = None
        if self.need_slice:
            # slice weights [B, H, N_coords, slice_num]
            query, slice_weights = self.slice_layer(query, return_latent=True, return_slice_weights=True)  # [B, slice_num, hidden_features]
        
        # cross attention blocks
        for block in self.blocks:
            query = block(query, latent)  # [B, G or N, D]
        
        if slice_weights is not None:
            query = rearrange(query, 'b g (h d) -> b h g d', h=self.slice_head)
            query = torch.einsum('b h g d, b h n g -> b h n d', query, slice_weights)
            query = rearrange(query, 'b h n d -> b n (h d)')
            
        query = self.out_proj(query)
        
        # sanity reshape
        if coords.ndim ==4:
            query = query.view(b, h, w, -1).contiguous()
        elif coords.ndim ==5:
            query = query.view(b, d1, d2, d3, -1).contiguous()
            
        return query


class PerceiverAutoencoder(nn.Module):
    def __init__(self,
                 coord_features,
                 field_features,
                 latent_features,
                 num_latents,
                 encoder_hidden_features=128,
                 encoder_n_layers=4,
                 decoder_hidden_features=128,
                 decoder_n_layers=4,
                 encoder_dropout=0,
                 decoder_dropout=0,
                 encoder_n_head=8,
                 decoder_n_head=8,
                 encoder_mlp_ratio=4,
                 decoder_mlp_ratio=4,
                 encoder_act='gelu',
                 decoder_act='gelu',
                 encoder_need_slice=False,
                 encoder_slice_num=None,
                 decoder_need_slice=False,
                 decoder_slice_num=None,
                 quantizer_cfg=None,
                 ):
        super().__init__()
        self.encoder = PerceiverEncoder(
            coord_features,
            field_features,
            num_latents=num_latents,
            latent_dim=latent_features,
            hidden_features=encoder_hidden_features,
            n_layers=encoder_n_layers,
            need_slice=encoder_need_slice,
            slice_num=encoder_slice_num,
            dropout=encoder_dropout,
            n_head=encoder_n_head,
            mlp_ratio=encoder_mlp_ratio,
            act=encoder_act
        )
        self.decoder = PerceiverDecoder(
            coord_features,
            latent_features,
            field_features,
            hidden_features=decoder_hidden_features,
            n_layers=decoder_n_layers,
            need_slice=decoder_need_slice,
            slice_num=decoder_slice_num,
            dropout=decoder_dropout,
            n_head=decoder_n_head,
            mlp_ratio=decoder_mlp_ratio,
            act=decoder_act
        )
        
        if quantizer_cfg is not None:
            if "codebook_size" in quantizer_cfg:
                self.quantizer = VectorQuantize(**quantizer_cfg)
            elif "levels" in quantizer_cfg:
                self.quantizer = FSQ(**quantizer_cfg)
            else:
                raise ValueError("Unknown quantizer configuration")
        self.quantizer_cfg = quantizer_cfg
    
    def encode(self, coords, field_values):
        latents = self.encoder(coords, field_values)
        indices = None
        loss = None
        
        if self.quantizer_cfg is not None:
            if isinstance(self.quantizer, VectorQuantize):
                latents, indices, loss = self.quantizer(latents)
                loss = loss.mean()
            elif  isinstance(self.quantizer, FSQ):
                latents, indices = self.quantizer(latents)
                
        return latents, indices, loss
    
    def decode(self, coords, latents):
        outputs = self.decoder(coords, latents)
        return outputs
    
    def forward(self, input_coords, input_field_values, query_coords=None, 
                return_latents=False, return_indices=False):

        latents, indices, loss = self.encode(input_coords, input_field_values)
        
        if query_coords is None:
            query_coords = input_coords
        
        outputs = self.decode(query_coords, latents)
        
        if return_latents and return_indices:
            return outputs, latents, indices
        if return_latents:
            return outputs, latents, loss
        if return_indices:
            return outputs, indices
        return outputs