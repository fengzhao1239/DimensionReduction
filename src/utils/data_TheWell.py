import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import warnings
import random

from the_well.data import WellDataset
from the_well.data import normalization
from .masking import ComplementMasking, RandomMasking



class MaskedReconstructionWellDataset(WellDataset):
    def __init__(
        self,
        *args,
        masking_strategy: str | None = None,          # None / 'complement' / 'random'
        encoder_point_ratio: float | None = None,
        decoder_point_ratio: float | None = None,
        normalize_coords: bool = True,
        downsample_factor: int = 0,
        **kwargs,
    ):
        if "n_steps_input" in kwargs and kwargs["n_steps_input"] != 1:
            warnings.warn(
                f"ReconstructionWellDataset ignores n_steps_input={kwargs['n_steps_input']} "
                f"and forces n_steps_input=1.",
                UserWarning,
            )
        if "n_steps_output" in kwargs and kwargs["n_steps_output"] != 0:
            warnings.warn(
                f"ReconstructionWellDataset ignores n_steps_output={kwargs['n_steps_output']} "
                f"and forces n_steps_output=0.",
                UserWarning,
            )
        if "return_grid" in kwargs and kwargs["return_grid"] is not True:
            warnings.warn(
                f"MaskedReconstructionWellDataset ignores return_grid={kwargs['return_grid']} and forces return_grid=True.",
                UserWarning,
            )
        kwargs["n_steps_input"] = 1
        kwargs["n_steps_output"] = 0
        kwargs["return_grid"] = True
        super().__init__(*args, **kwargs)
        
        self.normalize_coords = normalize_coords
        self.masking_strategy = masking_strategy
        self.encoder_point_ratio = encoder_point_ratio
        self.decoder_point_ratio = decoder_point_ratio
        self.downsample_factor = downsample_factor

        self.masking_fn = None
        if masking_strategy is None or masking_strategy == "none":
            self.masking_fn = None
            print("** No masking strategy applied")
        elif masking_strategy == "complement":
            if encoder_point_ratio is None:
                raise ValueError("encoder_point_ratio must be provided for complement masking")
            self.masking_fn = ComplementMasking(encoder_point_ratio=encoder_point_ratio)
            print(f"** Using ComplementMasking with encoder_point_ratio={encoder_point_ratio}")
        elif masking_strategy == "random":
            if encoder_point_ratio is None or decoder_point_ratio is None:
                raise ValueError("Both encoder_point_ratio and decoder_point_ratio must be provided for random masking")
            self.masking_fn = RandomMasking(
                encoder_point_ratio=encoder_point_ratio,
                decoder_point_ratio=decoder_point_ratio
            )
            print(f"** Using RandomMasking with encoder_point_ratio={encoder_point_ratio}, decoder_point_ratio={decoder_point_ratio}")
        else:
            raise ValueError(f"Unknown masking_strategy: {masking_strategy}")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        coords = sample["space_grid"]
        field = sample["input_fields"][...,0:1]
        # assert field.min() >= -1.0 and field.max() <= 1.0
        
        if field.ndim == 4 and field.shape[0] >= 1:
            ii = torch.randint(0, field.shape[0], (1,)).item()
            field = field[ii]
            # field = field.squeeze(0)
        else:
            raise ValueError(f"Expected input_fields to have shape [1, H, W, F], got {field.shape}")
        
        if self.normalize_coords:
            coords = self._apply_coord_normalization(coords)
        
        # F = field.shape[-1]
        # ch = random.randrange(F)
        coords = coords[::self.downsample_factor, ::self.downsample_factor, :]
        field = field[::self.downsample_factor, ::self.downsample_factor, :]
        
        if self.masking_fn is None:
            return coords, field

        coords_flat = rearrange(coords, "h w d -> (h w) d")
        field_flat = rearrange(field, "h w f -> (h w) f")

        foi_enc, coords_enc, foi_dec, coords_dec = self.masking_fn(field_flat, coords_flat)
        
        sparse = {
                'coords_enc': coords_enc,   # [N_enc, D]
                'foi_enc': foi_enc,         # [N_enc, F]
                'coords_dec': coords_dec,   # [N_dec, D]
                'foi_dec': foi_dec,         # [N_dec, F]
            }

        return sparse
    
    def _apply_coord_normalization(self, coords):
        # [H, W, D] or [N, D]
        spatial_dims = tuple(range(coords.ndim - 1))
        c_min = coords.amin(dim=spatial_dims, keepdim=True)
        c_max = coords.amax(dim=spatial_dims, keepdim=True)
        return 2.0 * (coords - c_min) / (c_max - c_min) - 1.0



def get_data(base_path, well_dataset_name, masking_strategy=None, encoder_point_ratio=None, downsample_factor=1, **kwargs):
    train_dataset = MaskedReconstructionWellDataset(
        well_base_path=base_path,
        well_dataset_name=well_dataset_name,
        well_split_name="train",
        n_steps_input=1,
        n_steps_output=0,
        use_normalization=True,
        normalization_type=normalization.ZScoreNormalization,
        masking_strategy=masking_strategy,
        encoder_point_ratio=encoder_point_ratio,
        normalize_coords=True,
        downsample_factor=downsample_factor,
        **kwargs,
    )
    dataset_length = len(train_dataset)
    random_idx = torch.randperm(dataset_length)
    train_idx = random_idx[: int(0.8 * dataset_length)]
    val_idx = random_idx[int(0.8 * dataset_length) :]
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    
    print(
        "\n==================== TRAIN DATASET ====================\n"
        f"  Dataset name : {well_dataset_name}\n"
        f"  Num samples : {len(train_dataset)}\n"
        "======================================================\n"
    )
    
    val_dataset = MaskedReconstructionWellDataset(
        well_base_path=base_path,
        well_dataset_name=well_dataset_name,
        well_split_name="train",
        n_steps_input=1,
        n_steps_output=0,
        use_normalization=True,
        normalization_type=normalization.ZScoreNormalization,
        masking_strategy=None,
        encoder_point_ratio=None,
        normalize_coords=True,
        downsample_factor=downsample_factor,
        **kwargs,
    )
    val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
    print(
        "\n==================== VALID DATASET ====================\n"
        f"  Dataset name : {well_dataset_name}\n"
        f"  Num samples : {len(val_dataset)}\n"
        "======================================================\n"
    )
    return train_dataset, val_dataset


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
