import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .normalizer import Normalizer
from .masking import ComplementMasking, RandomMasking
from einops import rearrange, repeat
import os
from torch.utils.data import random_split
from tqdm import tqdm
import h5py
import math


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper

def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name](**kwargs)

def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    return DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory)

def get_data(dataset_name,  **kwargs):

    dataset = get_dataset(dataset_name, **kwargs)
    print(f"\n\n---- Dataset {dataset_name} length: {len(dataset)}\n\n")
    
    return dataset



@register_dataset("TurbulentInlet")
class InletDataset(Dataset):
    def __init__(self, data_dir, norm_dir, norm_method='-11', train_val_flag='train',
                 masking_strategy=None, encoder_point_ratio=None, decoder_point_ratio=None):
        print("====================Preparing Inlet Dataset====================")
        
        data = np.load(data_dir)
        if train_val_flag=='train':
            self.data = data[:, :3480]
        else:
            self.data = data[:, 3480:]
        print(f"** Loaded {train_val_flag} data shape: {self.data.shape}")
        self.data = rearrange(self.data, 'e t h w -> (e t) h w')  # [B, H, W]
        print(f"** Reshaped data to: {self.data.shape}")
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(-1)  # [B, H, W, C=1]
        self.normalizer = Normalizer(method=norm_method, dim=(0, 1, 2))
        self.foi = self._load_or_fit_normalizer(self.data, norm_dir)
        print(f"** Normalized data shape (foi): {self.foi.shape}")
        
        # self.data = []
        # num_envs = 0
        # with np.load(data_dir) as f:
        #     for k in tqdm(f.files):
        #         data_env = f[k]
        #         num_traj = data_env.shape[0]
        #         flag_idx = int(0.8 * num_traj)
        #         # self.data.append(f[k][::4])    # every env has 4872 trajectories [4872, 10, 64, 64]
        #         if train_val_flag == 'train':
        #             self.data.append(data_env[:flag_idx][::2])  # take first 80% for training
        #         else:
        #             self.data.append(data_env[flag_idx:][::4])  # take last 20% for validation/testing
        #         num_envs += 1
        
        # self.data = np.concatenate(self.data, axis=0)  # [B, T, H, W]
        # self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)  # [B, C=1, T, H, W]
        # assert len(self.data.shape) == 5, "Input data must be 5-dimensional (B, C, T, H, W)"
        # print(f"** Loaded data shape: {self.data.shape} from {data_dir} with {num_envs} environments")
        # self.normalizer = Normalizer(method=norm_method, dim=(0, 2, 3, 4))
        # self.normed_data = self._load_or_fit_normalizer(self.data, norm_dir)
        # assert self.data.shape == self.normed_data.shape, "Normalized data must have the same shape as input data"
        # self.foi = rearrange(self.normed_data, 'b c t h w -> (b t) h w c')
        
        
        self.length = self.foi.shape[0]
        
        coords = self._make_Cartesian_coord()  # [H, W, 2]
        coords = torch.tensor(coords, dtype=torch.float32)
        self.coords = coords.unsqueeze(0).repeat(self.length, 1, 1, 1)  # [B*T, H, W, 2]
        print("** Coordinate shape: ", self.coords.shape)
        
        # Setup masking strategy
        self.masking_strategy = masking_strategy
        self.masking_fn = None
        if masking_strategy == 'complement':
            if encoder_point_ratio is None:
                raise ValueError("encoder_point_ratio must be provided for complement masking")
            self.masking_fn = ComplementMasking(encoder_point_ratio=encoder_point_ratio)
            print(f"** Using ComplementMasking with encoder_point_ratio={encoder_point_ratio}")
        elif masking_strategy == 'random':
            if encoder_point_ratio is None or decoder_point_ratio is None:
                raise ValueError("Both encoder_point_ratio and decoder_point_ratio must be provided for random masking")
            self.masking_fn = RandomMasking(
                encoder_point_ratio=encoder_point_ratio,
                decoder_point_ratio=decoder_point_ratio
            )
            print(f"** Using RandomMasking with encoder_point_ratio={encoder_point_ratio}, decoder_point_ratio={decoder_point_ratio}")
        else:
            print("** No masking strategy applied")
        
        print("====================Dataset Preparation Done====================")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            If no masking:
                coords: [H, W, 2]
                foi: [H, W, C]
            If masking is applied:
                dict with keys:
                    'coords_enc': [N_enc, 2]
                    'foi_enc': [N_enc, C]
                    'coords_dec': [N_dec, 2]
                    'foi_dec': [N_dec, C]
        """
        coords = self.coords[idx]  # [H, W, 2]
        foi = self.foi[idx]  # [H, W, C]
        
        if self.masking_fn is None:
            # Return in original format (H, W, C)
            return coords, foi
        else:
            # Reshape to (N, C) format for masking
            H, W, C = foi.shape
            coords_flat = rearrange(coords, 'h w c -> (h w) c')  # [N, 2]
            foi_flat = rearrange(foi, 'h w c -> (h w) c')  # [N, C]
            foi_enc, coords_enc, foi_dec, coords_dec = self.masking_fn(foi_flat, coords_flat)
            
            return {
                'coords_enc': coords_enc,  # [N_enc, 2]
                'foi_enc': foi_enc,  # [N_enc, C]
                'coords_dec': coords_dec,  # [N_dec, 2]
                'foi_dec': foi_dec,  # [N_dec, C]
            }
    
    def _load_or_fit_normalizer(self, data, norm_dir):
        if os.path.exists(f"{norm_dir}/normalizer_params.pt"):
            print(f"Loading normalizer parameters from {norm_dir}/normalizer_params.pt")
            norm_params = torch.load(f"{norm_dir}/normalizer_params.pt")
            self.normalizer.params = norm_params["normalizer_params"]
        else:
            print("No noramlization file found! Calculating normalizer parameters and save.")
            self.normalizer.fit_normalize(data)
            print(f"Saving normalizer parameters to {norm_dir}/normalizer_params.pt")
            print(f"Calculated normalizer params: {self.normalizer.get_params()}")
            toSave = {
            "normalizer_params": self.normalizer.get_params(),
            }
            if not os.path.exists(f"{norm_dir}"):
                os.makedirs(norm_dir, exist_ok=True)
            torch.save(toSave, norm_dir + f"/normalizer_params.pt")
        return self.normalizer.normalize(data)
    
    def _load_or_fit_denormalizer(self, data, norm_dir):
        if os.path.exists(f"{norm_dir}/normalizer_params.pt"):
            print(f"Loading normalizer parameters from {norm_dir}/normalizer_params.pt")
            norm_params = torch.load(f"{norm_dir}/normalizer_params.pt")
            self.normalizer.params = norm_params["normalizer_params"]
        else:
            print("No noramlization file found! Calculating normalizer parameters and save.")
            self.normalizer.fit_normalize(data)
            print(f"Saving normalizer parameters to {norm_dir}/normalizer_params.pt")
            print(f"Calculated normalizer params: {self.normalizer.get_params()}")
            toSave = {
            "normalizer_params": self.normalizer.get_params(),
            }
            if not os.path.exists(f"{norm_dir}"):
                os.makedirs(norm_dir, exist_ok=True)
            torch.save(toSave, norm_dir + f"/normalizer_params.pt")
        return self.normalizer.denormalize(data)
    
    @staticmethod
    def _make_Cartesian_coord():
        '''return <h,w,c_coord>'''
        h = w = 64
        x_coord = np.linspace(-1, 1, h)
        y_coord = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape ({h}, {w}, 2), but got {xy_coord.shape}"
        return xy_coord.astype(np.float32)


@register_dataset("KolmogorovFlow")
class KolmogorovDataset(Dataset):
    def __init__(self, data_dir, norm_dir, norm_method='-11', train_val_flag='train',
                 masking_strategy=None, encoder_point_ratio=None, decoder_point_ratio=None):
        print(f"====================Preparing Kolmogorov Dataset ({train_val_flag})====================")
        self.data_dir = data_dir
        self.train_val_flag = train_val_flag
        
        with h5py.File(self.data_dir, 'r') as f:
            # 原始维度: [env, total_traj, total_t, h, w]
            self.env_num, self.traj_num, self.total_t, self.h, self.w = f['data/vorticity'].shape
            self.global_min = f['meta'].attrs['global_min']
            self.global_max = f['meta'].attrs['global_max']

        # --- 核心修改：定义轨迹（traj）和时间（t）的切分逻辑 ---
        if train_val_flag == 'train':
            # 训练集：前 8 条轨迹，前 80 个时间步
            self.traj_start, self.traj_end = 0, 4
            self.t_start, self.t_end = 0, self.total_t
        else:
            # 其他（val/test）：第 8 到 10 条轨迹，剩余时间步
            self.traj_start, self.traj_end = 4, 6
            self.t_start, self.t_end = 0, self.total_t
        
        self.traj_len = self.traj_end - self.traj_start
        self.t_len = self.t_end - self.t_start
        
        # 总长度 = 环境数 * 选中的轨迹数 * 选中的时间步数
        self.length = self.env_num * self.traj_len * self.t_len
        
        # 预计算单份坐标
        self.base_coords = torch.tensor(self._make_Cartesian_coord(), dtype=torch.float32)
        
        self.file = None
        self.dset = None
        self.masking_strategy = masking_strategy
        self.masking_fn = self._setup_masking(masking_strategy, encoder_point_ratio, decoder_point_ratio)
        
        print(f"** Config: Traj[{self.traj_start}:{self.traj_end}], Time[{self.t_start}:{self.t_end}]")
        print(f"** Total samples: {self.length}")
        print("====================Dataset Preparation Done====================")

    def _setup_masking(self, strategy, enc_ratio, dec_ratio):
        if strategy == 'complement':
            if enc_ratio is None: raise ValueError("Need encoder_point_ratio")
            return ComplementMasking(encoder_point_ratio=enc_ratio)
        elif strategy == 'random':
            if enc_ratio is None or dec_ratio is None: raise ValueError("Need ratios")
            return RandomMasking(encoder_point_ratio=enc_ratio, decoder_point_ratio=dec_ratio)
        return None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.data_dir, 'r', swmr=True)
            self.dset = self.file['data/vorticity']

        # --- 核心修改：三级索引映射 ---
        # 假设展开顺序是 (env, traj, t)
        t_offset = idx % self.t_len
        remainder = idx // self.t_len
        
        r_offset = remainder % self.traj_len
        e_idx = remainder // self.traj_len
        
        # 映射回 H5 文件中的真实索引
        real_t = self.t_start + t_offset
        real_r = self.traj_start + r_offset
        
        # 从磁盘读取
        val = self.dset[e_idx, real_r, real_t, ...]
        
        # 归一化与后期处理
        val = self._min_max_norm(val, self.global_min, self.global_max)
        foi = torch.from_numpy(val).float().unsqueeze(-1)
        coords = self.base_coords.clone()

        if self.masking_fn is None:
            return coords, foi
        else:
            H, W, C = foi.shape
            coords_flat = rearrange(coords, 'h w c -> (h w) c')
            foi_flat = rearrange(foi, 'h w c -> (h w) c')
            foi_enc, coords_enc, foi_dec, coords_dec = self.masking_fn(foi_flat, coords_flat)
            
            return {
                'coords_enc': coords_enc,
                'foi_enc': foi_enc,
                'coords_dec': coords_dec,
                'foi_dec': foi_dec,
            }

    @staticmethod
    def _min_max_norm(x, min_val, max_val):
        return 2 * (x - min_val) / (max_val - min_val) - 1
        
    @staticmethod
    def _make_Cartesian_coord():
        h = w = 128
        x_coord = np.linspace(-1, 1, h)
        y_coord = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        return np.stack((xx, yy), axis=-1).astype(np.float32)
        
        


@register_dataset("KolmogorovFlow_ST")
class KolmogorovSTDataset(Dataset):
    """
    Kolmogorov vorticity dataset with temporal chunking.

    Original H5 tensor:
      data/vorticity: (env=1000, traj=10, time=1000, h=128, w=128)

    Splits (both applied):
      - time split:  first 80 steps for train, last 20 for val
      - traj split:  first 8 trajs for train, last 2 for val

    Chunk sampling:
      - Each sample corresponds to one (env, traj, chunk_start) triple
      - Returns a temporal window of length num_input_frames

    Returns:
      - if masking_fn is None:
          coords, foi
            coords: (N, coord_dim)
            foi:    (T_window, N, 1)
      - else:
          dict with
            'coords_enc': (T_window, N_enc, coord_dim)
            'foi_enc':    (T_window, N_enc, 1)
            'coords_dec': (T_window, N_dec, coord_dim)
            'foi_dec':    (T_window, N_dec, 1)
    """
    def __init__(
        self,
        data_dir: str,
        norm_dir=None,
        norm_method: str = "-11",
        train_val_flag: str = "train",
        # chunking
        num_input_frames: int = 4,
        chunk_stride: int = 1,
        drop_last_incomplete_chunk: bool = True,
        # masking
        masking_strategy=None,
        encoder_point_ratio=None,
        decoder_point_ratio=None,
    ):
        print(f"====================Preparing Kolmogorov Dataset ({train_val_flag})====================")
        self.data_dir = data_dir
        self.train_val_flag = train_val_flag

        self.num_input_frames = int(num_input_frames)
        if self.num_input_frames < 1:
            raise ValueError(f"num_input_frames must be >= 1, got {self.num_input_frames}")
        self.chunk_stride = int(chunk_stride)
        if self.chunk_stride < 1:
            raise ValueError(f"chunk_stride must be >= 1, got {self.chunk_stride}")
        self.drop_last_incomplete_chunk = bool(drop_last_incomplete_chunk)

        with h5py.File(self.data_dir, "r") as f:
            self.env_num, self.traj_num, self.total_t, self.h, self.w = f["data/vorticity"].shape
            self.global_min = f["meta"].attrs["global_min"]
            self.global_max = f["meta"].attrs["global_max"]

        # --- required split policy ---
        # trajectories: 8 train, 2 val
        # time steps:   800 train, 200 val
        if train_val_flag == "train":
            self.traj_start, self.traj_end = 0, 8
            self.t_start, self.t_end = 0, 80
        else:
            self.traj_start, self.traj_end = 8, 10
            self.t_start, self.t_end = 80, 100

        self.traj_len = self.traj_end - self.traj_start
        self.t_len = self.t_end - self.t_start

        # how many chunk start indices exist inside [t_start, t_end)
        max_start = self.t_len - self.num_input_frames
        if max_start < 0:
            raise ValueError(
                f"Time range too short for num_input_frames: t_len={self.t_len}, num_input_frames={self.num_input_frames}"
            )

        if self.drop_last_incomplete_chunk:
            self.num_chunk_starts = (max_start // self.chunk_stride) + 1
        else:
            # allow last chunk to be shorter and repeat-pad
            self.num_chunk_starts = (self.t_len - 1) // self.chunk_stride + 1

        self.length = self.env_num * self.traj_len * self.num_chunk_starts

        # coords: (H,W,2) -> (N,2)
        self.base_coords = torch.tensor(self._make_Cartesian_coord(self.h, self.w), dtype=torch.float32)

        self.file = None
        self.dset = None

        self.masking_strategy = masking_strategy
        self.encoder_point_ratio = encoder_point_ratio
        self.decoder_point_ratio = decoder_point_ratio
        
        self.masking_fn = self._setup_masking(masking_strategy, encoder_point_ratio, decoder_point_ratio)

        print(f"** Split: Traj[{self.traj_start}:{self.traj_end}], Time[{self.t_start}:{self.t_end}]")
        print(f"** Chunk: num_input_frames={self.num_input_frames}, chunk_stride={self.chunk_stride}, "
              f"num_chunk_starts={self.num_chunk_starts}, drop_last_incomplete_chunk={self.drop_last_incomplete_chunk}")
        print(f"** Total samples: {self.length}")
        print("====================Dataset Preparation Done====================")

    def _setup_masking(self, strategy, enc_ratio, dec_ratio):
        if strategy == "complement":
            if enc_ratio is None:
                raise ValueError("Need encoder_point_ratio")
            return ComplementMasking(encoder_point_ratio=enc_ratio)
        elif strategy == "random":
            if enc_ratio is None or dec_ratio is None:
                raise ValueError("Need encoder_point_ratio and decoder_point_ratio")
            return RandomMasking(encoder_point_ratio=enc_ratio, decoder_point_ratio=dec_ratio)
        return None
    


    def __len__(self):
        return self.length

    def _lazy_open(self):
        if self.file is None:
            self.file = h5py.File(self.data_dir, "r", swmr=True)
            self.dset = self.file["data/vorticity"]

    def _index_to_env_traj_chunkstart(self, idx: int):
        # flattened order: (env, traj, chunk_start)
        chunk_offset = idx % self.num_chunk_starts
        remainder = idx // self.num_chunk_starts

        traj_offset = remainder % self.traj_len
        env_idx = remainder // self.traj_len

        real_traj = self.traj_start + traj_offset
        return env_idx, real_traj, chunk_offset

    def __getitem__(self, idx):
        self._lazy_open()

        env_idx, real_traj, chunk_offset = self._index_to_env_traj_chunkstart(idx)

        start_t = self.t_start + chunk_offset * self.chunk_stride
        end_t = start_t + self.num_input_frames

        if end_t <= self.t_end:
            vals = self.dset[env_idx, real_traj, start_t:end_t, ...]  # (T,H,W)
        else:
            # only possible if drop_last_incomplete_chunk=False
            vals = self.dset[env_idx, real_traj, start_t:self.t_end, ...]  # (t_remain,H,W)
            t_remain = vals.shape[0]
            if t_remain <= 0:
                raise RuntimeError("Invalid chunk indexing produced empty slice.")
            last = vals[-1:, ...]
            pad = np.repeat(last, repeats=(self.num_input_frames - t_remain), axis=0)
            vals = np.concatenate([vals, pad], axis=0)  # (T,H,W)

        vals = self._min_max_norm(vals, self.global_min, self.global_max)  # (T,H,W)
        foi = torch.from_numpy(vals).float().unsqueeze(-1)                 # (T,H,W,1)
        coords = self.base_coords.clone()
        coords = repeat(coords, "h w c -> t h w c", t=foi.shape[0])

        # No masking applied
        if self.masking_fn is None:
            return coords, foi

        # masking per frame, stacked over time
        foi = rearrange(foi, "t h w c -> t (h w) c")  # (T,N,1)
        coords = rearrange(coords, "t h w c -> t (h w) c")  # (T,N,2)

        # Apply static masking
        _, _, _, _, indices_enc, indices_dec = self.masking_fn(foi[0], coords[0], return_indices=True)

        foi_enc = foi[:, indices_enc, :]
        coords_enc = coords[:, indices_enc, :]
        foi_dec = foi[:, indices_dec, :]
        coords_dec = coords[:, indices_dec, :]

        return {
                'coords_enc': coords_enc,
                'foi_enc': foi_enc,
                'coords_dec': coords_dec,
                'foi_dec': foi_dec,
            }

    @staticmethod
    def _min_max_norm(x, min_val, max_val):
        return 2 * (x - min_val) / (max_val - min_val) - 1

    @staticmethod
    def _make_Cartesian_coord(h: int, w: int):
        x_coord = np.linspace(-1, 1, h)
        y_coord = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing="ij")
        return np.stack((xx, yy), axis=-1).astype(np.float32)

