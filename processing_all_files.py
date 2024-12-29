import os
import zipfile
from mineral_clustering_for_heatmap.storing_images_to_dataframe import main
import asyncio

def unzip_folders(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for _zipfile in os.listdir(input_path):
        new_output_path = os.path.join(output_path, os.path.splitext(_zipfile)[0])
        os.makedirs(new_output_path, exist_ok=True)
        try:
            with zipfile.ZipFile(os.path.join(input_path, _zipfile), 'r') as zip_ref:
                zip_ref.extractall(new_output_path)
        except:
            print("Not a .zip file")

        print(f"Files extracted for {_zipfile}")
        # os.remove(os.path.join(input_path, _zipfile))
        # print(f"Deleted {_zipfile} from {input_path}")

async def process_images(input_path, array_path):
    await main(data_path=input_path, array_path=array_path)

if __name__ == "__main__":
    INPUT_PATH = r"TRAIN_DATA/ch2_iir_nci_20240501T2302103346_d_img_d18_DATA_FROM_MOON_MAP"
    OUTPUT_PATH = r"TRAIN_DATA/ch2_iir_nci_20240501T2302103346_d_img_d18_DATA_FROM_MOON_MAP"
    ARRAY_PATH = r"mineral_clustering_for_heatmap/pixel_vector_data_new_async.npy"

    unzip_folders(INPUT_PATH, OUTPUT_PATH)
    # process_images(OUTPUT_PATH, ARRAY_PATH)