import torch
from dataclasses import dataclass


@dataclass
class ComplementMasking:
    encoder_point_ratio: float
    
    def __call__(self, u, x, return_indices=False):
        """
        Splits the points into two disjoint sets: one for the encoder
        and the rest for the decoder.
        
        Args:
            u (torch.Tensor): Field values of shape [N, C].
            x (torch.Tensor): Coordinates of shape [N, C_coord].
        Returns:
            u_enc, x_enc, u_dec, x_dec (torch.Tensors).
        """
        if self.encoder_point_ratio == -1:
            return u, x, u, x
            
        if not (0.0 < self.encoder_point_ratio < 1.0):
            raise ValueError("`encoder_point_ratio` must be in range (0, 1).")
        
        n_total_pts = u.shape[0]
        n_enc_pts = int(self.encoder_point_ratio * n_total_pts)
            
        # Shuffle all indices once to ensure disjoint sets
        indices = torch.randperm(n_total_pts)
            
        # Partition the indices into two non-overlapping groups
        indices_enc = indices[:n_enc_pts]
        indices_dec = indices[n_enc_pts:]
        
        # Extract corresponding data
        u_enc, x_enc = u[indices_enc], x[indices_enc]
        u_dec, x_dec = u[indices_dec], x[indices_dec]

        if return_indices:
            return u_enc, x_enc, u_dec, x_dec, indices_enc, indices_dec
        
        return u_enc, x_enc, u_dec, x_dec


@dataclass
class RandomMasking:
    encoder_point_ratio: float
    decoder_point_ratio: float
    
    def __call__(self, u, x, return_indices=False):
        """
        Independently samples points for the encoder and decoder. 
        Sets may overlap.
        
        Args:
            u (torch.Tensor): Field values of shape [N, C].
            x (torch.Tensor): Coordinates of shape [N, C_coord].
            return_indices (bool): If True, also return the sampled indices.
        Returns:
            u_enc, x_enc, u_dec, x_dec (torch.Tensors).
            If return_indices=True, also returns indices_enc, indices_dec.
        """
        if self.encoder_point_ratio == -1 and self.decoder_point_ratio == -1:
            return u, x, u, x
            
        if not (0.0 < self.encoder_point_ratio < 1.0 and 0.0 < self.decoder_point_ratio < 1.0):
            raise ValueError("Point ratios must be in range (0, 1).")
        
        n_total_pts = u.shape[0]
        n_enc_pts = int(self.encoder_point_ratio * n_total_pts)
        n_dec_pts = int(self.decoder_point_ratio * n_total_pts)
        
        # Independent sampling: encoder and decoder indices are selected separately
        indices_enc = torch.randperm(n_total_pts)[:n_enc_pts]
        indices_dec = torch.randperm(n_total_pts)[:n_dec_pts]
        
        u_enc, x_enc = u[indices_enc], x[indices_enc]
        u_dec, x_dec = u[indices_dec], x[indices_dec]
        
        if return_indices:
            return u_enc, x_enc, u_dec, x_dec, indices_enc, indices_dec
        
        return u_enc, x_enc, u_dec, x_dec