export CUDA_VISIBLE_DEVICES=0,2
export PYTHONPATH=$(pwd)

echo "Using CUDA device: ${CUDA_VISIBLE_DEVICES}"
echo "Using PYTHONPATH: ${PYTHONPATH}"

python3 scripts/train_dim_red.py --config-name STAE_HIT training.devices=2 training.batch_size=10