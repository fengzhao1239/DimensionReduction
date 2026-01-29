import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import xarray

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import jax.image

import numpy
from scipy.ndimage import zoom
from tqdm import tqdm

import h5py


def run_sim(viscosity, seed):
    # physical parameters
    max_velocity = 7
    grid = grids.Grid((512, 512), domain=((0, 1 * jnp.pi), (0, 1 * jnp.pi)))
    dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)
    smooth = True # use anti-aliasing 

    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)

    final_time = 24.
    outer_steps = 120
    inner_steps = (final_time // dt) // outer_steps

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    _, trajectory = trajectory_fn(vorticity_hat0)

    w_sol =jnp.fft.irfftn(trajectory, axes=(1,2))
    
    burnin_sol = w_sol[20:]
    downsampled_sol = jax.image.resize(
        burnin_sol,
        shape=(burnin_sol.shape[0], 128, 128),
        method='bicubic'
    )
    assert downsampled_sol.shape == (outer_steps - 20, 128, 128)
    return downsampled_sol


def sample_viscosity(min=1e-3, max=1e-2, num=500):
    return numpy.linspace(min, max, num).tolist()

def run(
    h5_path="/ehome/zhao/pretrain/mycode/dataset/datasets/HIT/kolmogorov_flow.h5",
    env_num=1000,
    traj_per_env=10,
    vmin=1e-3,
    vmax=1e-2,
    base_seed=None,
    dtype=numpy.float32,
    compression_level=4
):
    
    viscosity_list = numpy.linspace(vmin, vmax, env_num, dtype=numpy.float32)

    ss = numpy.random.SeedSequence(base_seed)
    n_runs = env_num * traj_per_env
    child_seqs = ss.spawn(n_runs)
    seeds = numpy.array(
        [cs.generate_state(1, dtype=numpy.uint32)[0] for cs in child_seqs],
        dtype=numpy.uint32
    ).reshape(env_num, traj_per_env)

    T, H, W = 100, 128, 128
    chunk = (1, traj_per_env, 10, H, W)
    
    global_min = numpy.inf
    global_max = -numpy.inf

    with h5py.File(h5_path, "w") as f:
        g_data = f.create_group("data")
        d_obs = g_data.create_dataset(
            "vorticity",
            shape=(env_num, traj_per_env, T, H, W),
            dtype=dtype,
            chunks=chunk,
            compression="gzip",
            compression_opts=int(compression_level),
            shuffle=True
        )

        g_meta = f.create_group("meta")
        g_meta.create_dataset("viscosity", data=viscosity_list, dtype=numpy.float32)
        g_meta.attrs["layout"] = "env, traj, time, x, y"

        pbar = tqdm(total=n_runs, desc="Simulating & writing (env,traj)")
        for e in range(env_num):
            nu = float(viscosity_list[e])
            for k in range(traj_per_env):
                sd = int(seeds[e, k])  # 保证是 Python int

                sol = run_sim(nu, sd)
                sol_np = numpy.asarray(jax.device_get(sol), dtype=dtype)

                d_obs[e, k] = sol_np
                
                local_min = float(sol_np.min())
                local_max = float(sol_np.max())
                
                if local_min < global_min:
                    global_min = local_min
                if local_max > global_max:
                    global_max = local_max
                    
                pbar.update(1)

        pbar.close()
        
        g_meta.attrs["global_min"] = global_min
        g_meta.attrs["global_max"] = global_max
        
        f.flush()

    print(f"Saved HDF5: {h5_path}")
    print(f"obs shape: ({env_num}, {traj_per_env}, {T}, {H}, {W}), dtype={numpy.dtype(dtype)}")
    print(f"global min / max: {global_min:.6g}, {global_max:.6g}")



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    
    run()