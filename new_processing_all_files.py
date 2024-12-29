import numpy as np
import zipfile
import xml.etree.ElementTree as ET
import asyncio
import aiofiles
import os

# GLOBAL VARIABLES
VECTOR_DIMENSION_START = 7
VECTOR_DIMENSION_END = 115
CONCURRENT_TASKS_LIMIT = 7  # Limit the number of concurrent tasks to manage memory usage
IDX = 0

async def read_image_shape(image_file_xml_path):
    async with aiofiles.open(image_file_xml_path, 'r') as f:
        content = await f.read()
    tree = ET.ElementTree(ET.fromstring(content))
    root = tree.getroot()

    axis_elements = {}
    namespaces = {
        'ns': 'http://pds.nasa.gov/pds4/pds/v1'
    }

    for axis_array in root.findall('.//ns:Axis_Array', namespaces):        
        axis_name_element = axis_array.find('ns:axis_name', namespaces)
        if axis_name_element is not None:
            axis_name = axis_name_element.text.strip()
        else:
            print("Warning: 'axis_name' tag is missing.")
            continue

        elements_element = axis_array.find('ns:elements', namespaces)
        if elements_element is not None:
            try:
                elements = int(elements_element.text.strip())
            except ValueError:
                print(f"Error: 'elements' value is not a valid integer.")
                continue
        else:
            print(f"Warning: 'elements' tag is missing.")
            continue
        axis_elements[axis_name] = elements

    return (256, axis_elements.get('LINE', 0), axis_elements.get('SAMPLE', 0))

async def read_image(image_path, shape):
    async with aiofiles.open(image_path, 'rb') as f:
        image_data = np.reshape(np.frombuffer(await f.read(), np.float32), shape)
    return image_data

def extract_pixel_array(image, pixel_coords_in_image):
    return np.asarray(image[VECTOR_DIMENSION_START:VECTOR_DIMENSION_END, pixel_coords_in_image[0], pixel_coords_in_image[1]])

async def process_single_image(array_path, image_path, geometry_path, image_file_xml_path, idx):
    shape = await read_image_shape(image_file_xml_path)
    image = await read_image(image_path, shape)

    try:
        dataset = np.load(array_path)
    except FileNotFoundError:
        np.save(array_path, np.zeros(shape=(VECTOR_DIMENSION_END - VECTOR_DIMENSION_START,), dtype=np.float32))
        dataset = np.load(array_path)

    batch_size = 20
    for i in range(shape[1]):
        batch = []
        for j in range(shape[2]):
            pixel_array = extract_pixel_array(image, [i, j])
            batch.append(pixel_array)
            if len(batch) >= batch_size:
                dataset = np.vstack((dataset, batch))
                batch = []
        
        if batch:
            dataset = np.vstack((dataset, batch))
        
        np.save(array_path, dataset)
        if i % 20 == 0:
            print(f"Dataset shape after row {i}: {dataset.shape}")

    print(f"Processed image {idx + 1}.")
    return idx + 1

async def get_geometry_file_path(sub_folders_path):
    geometry_path = os.path.join(sub_folders_path, "geometry/calibrated")
    for file_name in os.listdir(geometry_path):
        if file_name.endswith(".csv"):
            return os.path.join(geometry_path, file_name)

async def get_image_file_path(sub_folders_path):
    data_path = os.path.join(sub_folders_path, "data/calibrated")
    for file_name in os.listdir(data_path):
        if file_name.endswith(".qub"):
            return os.path.join(data_path, file_name)

async def get_image_xml_file_path(sub_folders_path):
    data_path = os.path.join(sub_folders_path, "data/calibrated")
    for file_name in os.listdir(data_path):
        if file_name.endswith(".xml"):
            return os.path.join(data_path, file_name)

async def process_folder(folder, data_path, array_path, semaphore):
    async with semaphore:
        geometry_file_path = await get_geometry_file_path(os.path.join(data_path, folder))
        image_file_path = await get_image_file_path(os.path.join(data_path, folder))
        image_xml_file_path = await get_image_xml_file_path(os.path.join(data_path, folder))

        global IDX
        IDX = await process_single_image(array_path=array_path, image_path=image_file_path, geometry_path=geometry_file_path, image_file_xml_path=image_xml_file_path, idx=IDX)
        print(f"Processed folder {folder}.")

async def process_images(data_path, array_path):
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS_LIMIT)
    tasks = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            tasks.append(process_folder(folder, data_path, array_path, semaphore))
    
    await asyncio.gather(*tasks)

def unzip_folders(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for _zipfile in os.listdir(input_path):
        new_output_path = os.path.join(output_path, os.path.splitext(_zipfile)[0])
        os.makedirs(new_output_path, exist_ok=True)
        with zipfile.ZipFile(os.path.join(input_path, _zipfile), 'r') as zip_ref:
            zip_ref.extractall(new_output_path)

        print(f"Files extracted for {_zipfile}")
        os.remove(os.path.join(input_path, _zipfile))
        print(f"Deleted {_zipfile} from {input_path}")

    os.remove(input_path)

if __name__ == "__main__":
    # INPUT_PATH = r"NEW_DATA"
    OUTPUT_PATH = r"DATASET"
    ARRAY_PATH = r"mineral_clustering_for_heatmap/pixel_vector_data_train_files_new.npy"

    # Uncomment the line below if you need to unzip the folders
    # unzip_folders(INPUT_PATH, OUTPUT_PATH)
    
    # asyncio.run(process_images(OUTPUT_PATH, ARRAY_PATH))
    for moon_area in os.listdir(OUTPUT_PATH):
        moon_area_path = os.path.join(OUTPUT_PATH, moon_area)
        for folder in os.listdir(moon_area_path):
            if folder == "train":
                asyncio.run(process_images(os.path.join(moon_area_path, folder), ARRAY_PATH))
