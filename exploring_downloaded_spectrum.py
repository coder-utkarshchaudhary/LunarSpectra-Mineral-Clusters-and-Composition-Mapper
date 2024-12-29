import numpy as np
import pandas as pd
import os

log = open("spectral_matching_algorithms/min_max_wavelength.txt", "w+")

for _idx,i in enumerate(os.listdir(r"converted_csv_files")):
    df = pd.read_csv(os.path.join(r"converted_csv_files", i))
    log.write(f"{_idx+1}\t{i.split('_')[0]}\t\t{np.min(df.iloc[1:,0])}\t{np.max(df.iloc[1:,0])}\t{np.average(np.array([float(df.iloc[idx+1, 0])-float(df.iloc[idx,0]) for idx in range(1,df.shape[0]-1)], dtype=np.float32))}\t{df.iloc[1:, :].shape[0]}\n")

log.close()