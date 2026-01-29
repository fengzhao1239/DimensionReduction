export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$(pwd)

echo "Using CUDA device: ${CUDA_VISIBLE_DEVICES}"
echo "Using PYTHONPATH: ${PYTHONPATH}"

python3 scripts/train_dim_red_theWell.py --config-name fcnf_vq_era5 training.devices=1 training.batch_size=128