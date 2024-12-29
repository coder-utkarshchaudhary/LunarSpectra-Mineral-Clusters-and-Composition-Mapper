import pandas as pd
from matplotlib import pyplot as plt
import os

for _idx,i in enumerate(os.listdir(r"converted_csv_files")):
    df = pd.read_csv(os.path.join(r"converted_csv_files", i))
    plt.plot(df['Wavelength (nm)'], df['Reflectance'], color="red", linewidth=1.5)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f"Spectral plot of {i.split('_')[0]}")
    os.makedirs("spectral_curves_for_minerals_in_lookup_table", exist_ok=True)
    plt.savefig(f"spectral_curves_for_minerals_in_lookup_table/{i.split('_')[0]}.png")
    plt.close()

print("Done!!!")