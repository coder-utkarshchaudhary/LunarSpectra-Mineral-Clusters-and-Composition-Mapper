import pandas as pd
import os

for _idx,i in enumerate(os.listdir(r"converted_csv_files")):
    df = pd.read_csv(os.path.join(r"converted_csv_files", i))
    last_idx = 0
    for j in range(df.shape[0]):
        if str(df.iloc[j,0]).lower()[0] not in "abcdefghijklmnopqrstuvxyz":
            pass
        else:
            last_idx=j
            break

    df=df.iloc[:last_idx, :]
    df.to_csv(os.path.join(r"converted_csv_files", i), index=False)
    print(f"Done for file {_idx+1}")