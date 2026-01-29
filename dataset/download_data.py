from the_well.data import WellDataset
from the_well.utils.download import well_download
import time

start = time.time()
base_path = '/ehome/zhao/pretrain/mycode/dataset/'
print("Downloading dataset...")
well_download(base_path=base_path, dataset="gray_scott_reaction_diffusion", split="train")
print(f"Downloading validation set..., took {(time.time() - start)/60:.2f} minutes")
well_download(base_path=base_path, dataset="gray_scott_reaction_diffusion", split="valid")
print(f"Finished downloading dataset, took {(time.time() - start)/60:.2f} minutes in total.")