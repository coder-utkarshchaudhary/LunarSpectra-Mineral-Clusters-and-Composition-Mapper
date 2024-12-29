import os
import numpy as np
import threading
import xml.etree.ElementTree as ET
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

VECTOR_START = 7
VECTOR_END = 115
RELATIVE_BAND_WITH_1500_WAVELENGTH = 48-VECTOR_START
WINDOW_SIZE = 7
POLY_ORDER = 2
D_AU = 1

F0 = np.array([
  136.1259307,
  129.8781929,
  125.1457188,
  120.4566749,
  115.2187742,
  110.7989129,
  105.971862,
  102.2853476,
  98.83159112,
  95.00990644,
  91.72241746,
  88.63043389,
  85.44216416,
  83.09659958,
  80.7461688,
  77.99745659,
  75.43755054,
  72.53298554,
  70.30310472,
  67.71506702,
  65.53063581,
  63.51647332,
  61.49193881,
  59.39769145,
  57.24811211,
  55.56974549,
  53.96628612,
  52.39858882,
  50.94286582,
  49.55873832,
  47.99340839,
  46.35543865,
  45.11640663,
  43.75374359,
  42.46741487,
  41.1950428,
  39.93375405,
  38.7480202,
  37.63257797,
  36.52968828,
  35.48372942,
  34.51571377,
  33.5041102,
  32.62925225,
  31.80035805,
  30.98128654,
  30.16775831,
  29.32709974,
  28.56074168,
  27.8298174,
  27.0453247,
  26.30808675,
  25.51810387,
  24.75010497,
  24.00573968,
  23.24760491,
  22.51761852,
  21.78398871,
  21.06792047,
  20.39822233,
  19.7458807,
  19.11661541,
  18.44061437,
  17.83250529,
  17.26068394,
  16.65126453,
  16.11545704,
  15.61912435,
  15.1210474,
  14.62910738,
  14.16359209,
  13.72237684,
  13.31430194,
  12.94713935,
  12.56233275,
  12.18239943,
  11.79722098,
  11.38810049,
  11.04636914,
  10.71621297,
  10.38904988,
  10.06620698,
  9.753295821,
  9.46418631,
  9.201075776,
  8.960974818,
  8.732115834,
  8.508712424,
  8.28861478,
  8.070068082,
  7.850866176,
  7.629585176,
  7.417896212,
  7.21399149,
  7.014245694,
  6.819995994,
  6.637200746,
  6.463212542,
  6.291676014,
  6.122400975,
  5.952327234,
  5.785907458,
  5.631916792,
  5.48221029,
  5.338864421,
  5.183886388,
  5.053359936,
  4.941756508,
  4.835098184,
  4.719922707,
  4.619729215,
  4.511137419,
  4.407240202,
  4.306184976,
  4.210413629,
  4.117013411,
  4.012368768,
  3.918726643,
  3.824014432,
  3.725826304,
  3.646586732,
  3.564719937,
  3.488199195,
  3.397463341,
  3.32250234,
  3.262984894,
  3.190955311,
  3.122692223,
  3.056477464,
  2.991274348,
  2.926566072,
  2.864612339,
  2.802940836,
  2.743157021,
  2.685370618,
  2.628641884,
  2.571929704,
  2.517226294,
  2.465127643,
  2.414375576,
  2.365285234,
  2.316701141,
  2.26923212,
  2.222564505,
  2.178496705,
  2.135290025,
  2.092826765,
  2.051565701,
  2.010893773,
  1.971470582,
  1.932492639,
  1.893925453,
  1.853239032,
  1.814419696,
  1.780829606,
  1.751599126,
  1.715922793,
  1.680125966,
  1.647791753,
  1.621454182,
  1.593640531,
  1.560460708,
  1.532378246,
  1.507178355,
  1.480349348,
  1.454525518,
  1.426003985,
  1.40026592,
  1.376814112,
  1.351395724,
  1.327241488,
  1.303320437,
  1.279240078,
  1.255715058,
  1.232621586,
  1.209534773,
  1.186777237,
  1.163774025,
  1.141839466,
  1.121354795,
  1.102697582,
  1.084984542,
  1.06779729,
  1.050654559,
  1.034116451,
  1.018239678,
  1.003106371,
  0.987228033,
  0.971082552,
  0.954532246,
  0.938549781,
  0.922761605,
  0.90746215,
  0.892772367,
  0.876952832,
  0.86169586,
  0.846904043,
  0.832961745,
  0.820193322,
  0.808495532,
  0.796418017,
  0.784036511,
  0.771772032,
  0.760169612,
  0.74902997,
  0.737997332,
  0.727055348,
  0.716477866,
  0.704633464,
  0.691770452,
  0.681177697,
  0.668685204,
  0.6563386,
  0.643784606,
  0.630929839,
  0.618670348,
  0.605670184,
  0.593191697,
  0.582320158,
  0.571630629,
  0.561438106,
  0.551831735,
  0.542986524,
  0.534529199,
  0.526707332,
  0.518722109,
  0.511109087,
  0.50373316,
  0.496221855,
  0.489530981,
  0.482582186,
  0.475974536,
  0.469794569,
  0.463575699,
  0.458286546,
  0.452850271,
  0.447197638,
  0.441572082,
  0.43580287,
  0.430755766,
  0.425717099,
  0.420589447,
  0.41588213,
  0.410468477,
  0.405233536,
  0.399887123,
  0.394668014,
  0.389642973,
  0.384580319,
  0.379611238,
  0.374544041,
  0.369613524,
  0.364863435,
  0.360132602,
  0.355533758,
  0.350967069,
]).reshape(256,1,1)

TARGET_WAVELENGTHS = target_wavelengths = np.array([
    796.6, 813.4, 830.3, 847.2, 864.0, 880.9, 897.7, 914.6, 931.4, 948.3,
    965.1, 982.0, 998.8, 1015.7, 1032.5, 1049.4, 1066.2, 1083.1, 1099.9, 1116.8,
    1133.6, 1150.5, 1167.3, 1184.2, 1201.1, 1217.9, 1234.8, 1251.6, 1268.5, 1285.3,
    1302.2, 1319.0, 1335.9, 1352.7, 1369.6, 1386.4, 1403.3, 1420.1, 1437.0, 1453.8,
    1470.7, 1487.5, 1504.4, 1521.2, 1538.1, 1555.0, 1571.8, 1588.7, 1605.5, 1622.4,
    1639.2, 1656.1, 1672.9, 1689.8, 1706.6, 1723.5, 1740.3, 1757.2, 1774.0, 1790.9,
    1807.7, 1824.6, 1841.4, 1858.3, 1875.1, 1892.0, 1908.9, 1925.7, 1942.6, 1959.4,
    1976.3, 1993.1, 2010.0, 2026.8, 2043.7, 2060.5, 2077.4, 2094.2, 2111.1, 2127.9,
    2144.8, 2161.6, 2178.5, 2195.3, 2212.2, 2229.0, 2245.9, 2262.8, 2279.6, 2296.5,
    2313.3, 2330.2, 2347.0, 2363.9, 2380.7, 2397.6, 2414.4, 2431.3, 2448.1, 2465.0,
    2481.8, 2498.7, 2515.5, 2532.4, 2549.2, 2566.1, 2582.9, 2599.8, 2616.7
])

def read_xml_file(xml_file_path):
    """
        Helper function: Reads and extarct useful information from the xml file.
        Args-> XML file path
        Returns -> image shape, gain, exposure
    """
    
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    axis_elements = {}
    namespaces = {
        'ns': 'http://pds.nasa.gov/pds4/pds/v1'
    }

    namespaces_exposure_gain = {
        'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'
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

    gain = root.find('.//isda:gain', namespaces_exposure_gain).text
    exposure = root.find('.//isda:exposure', namespaces_exposure_gain).text

    return (256, axis_elements.get('LINE', 0), axis_elements.get('SAMPLE', 0)), exposure, gain

def read_image(data_folder_path):
    """
        Reading the images from disk: The function here uses the directory structure to extract the image_file from .qub file into a numpy array.
        Args -> data_folder_path : str, IMAGE_NAME/data/calibrated/.../
        Returns -> Image object
    """
    def convert_to_reflectance(image, solar_zenith_angle=45):
        return (np.pi * image)/(np.cos(solar_zenith_angle * np.pi / 180) * F0 * D_AU**2)
    
    image_shape=None
    image=-1
    for _file in os.listdir(data_folder_path):
        if _file[-4:] == ".xml":
            image_shape, _, _ = read_xml_file(os.path.join(data_folder_path, _file))
        elif _file[-4:] == ".qub":
            with open(os.path.join(data_folder_path, _file), 'rb') as f:
                image = np.reshape(np.frombuffer(f.read(), dtype=np.float32), newshape=image_shape) 
                return image

def extract_pixel_arrays(image):
    """
        Extract pixel arrays: Used to extract the pixels in an array to run the functions on.
        Args -> Image cube : np.ndarray
        Returns -> 2D array of each pixel with its corresponding array across axis=-1
    """
    pixel_values = []
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            pixel_values.append(image[VECTOR_START:VECTOR_END, i, j])

    return np.asarray(pixel_values, dtype=np.float32)

def normalize_pixel_arrays(pixel_vector_array):
    """
        Normalization: Takes in a pixel array and normalizes the array wrt _lambda = 1500nm. This value is chosen because there are no absorption drops in this region.
        Args -> 2D array of pixel vectors, image_normalization_value cube.
        Return -> 2D array of normalized pixel_vectors
    """
    for i in range(pixel_vector_array.shape[0]):
        pixel_vector_array[i] = pixel_vector_array[i]/pixel_vector_array[i][RELATIVE_BAND_WITH_1500_WAVELENGTH]

    return pixel_vector_array

def denoising_pixel_arrays(pixel_values, window_size=WINDOW_SIZE, polyorder=POLY_ORDER, axis=-1):
    """
        Denoising: Takes in a pixel array and runs Savinsky_Golay filter on pixel array axis.
        Args -> 2D array of pixel values, window_size, polynomial order, axis
        Return -> Smoothened 2D array of pixel values
    """
    return savgol_filter(pixel_values, window_length=window_size, polyorder=polyorder, axis=axis)

def interpolate_osf_bands(data_folder_path, pixel_values):
    """
        Interpolating OSF bands: Removes overlap in the file and interpolates lost data back
        Args -> 2D array of pixel vectors, data_folder_path
        Return -> 2D array of normalized pixel_vectors
    """
    for _file in os.listdir(data_folder_path):
        if _file[-4:] == ".xml":
            _, exposure, gain = read_xml_file(os.path.join(data_folder_path, _file))

    if exposure+gain == "e1g2" or exposure+gain=="e2g2" or exposure+gain=="e3g2":
        combined_array = np.concatenate((pixel_values[:, VECTOR_START:29], pixel_values[:,35:69], pixel_values[:, 76:VECTOR_END]), axis=-1)
        original_idx = np.concatenate([np.arange(VECTOR_START, 29), np.arange(35,69), np.arange(76, VECTOR_END)])
        interpolated_idx = np.arange(VECTOR_START, VECTOR_END)

        cubic_spline = CubicSpline(original_idx, combined_array)
        return cubic_spline(interpolated_idx)
    
    else:
        raise ValueError("E4G2 is not a valid sensor configuration for this software.")
    
def main(image_folder_path_that_has_4_sub_folders):
    data_folder_path_before_files = os.path.join(image_folder_path_that_has_4_sub_folders, "data/calibrated/")
    data_folder_path_files = os.path.join(data_folder_path_before_files, os.listdir(data_folder_path_before_files)[0])

    image = read_image(data_folder_path_files)
    if isinstance(image,int):
        return -1
    if len(image.shape) == 3:
        try:
            pixel_values = interpolate_osf_bands(data_folder_path_files, denoising_pixel_arrays(normalize_pixel_arrays(extract_pixel_arrays(image))))
        except:
            "Couldn't interpolate OSF."
            pixel_values = denoising_pixel_arrays(normalize_pixel_arrays(extract_pixel_arrays(image)))

        return pixel_values
    else:
        pass

def process_into_array(input_path, output_path):
    try:
        dataset = np.load(output_path)
    except:
        np.save(output_path, np.zeros((1, VECTOR_END-VECTOR_START), dtype=np.float32))
        dataset = np.load(output_path)
        
    dataset = np.vstack((dataset, main(input_path)))
    np.save(output_path, dataset)
    del dataset

for i in range(1,2):
    INPUT = f"DATASET_NEW/moon_site_{i}/train"
    OUTPUT = r"pixel_vector_data_train_files_transformed.npy"

    for _folder in os.listdir(INPUT):
        process_into_array(os.path.join(INPUT, _folder), OUTPUT)

# all_paths = []
# dataset = None
# solved = [0]

# INPUT_PATH = "DATASET_NEW"
# OUTPUT_PATH = "pixel_vector_data_train_files_transformed.npy"

# try:
#     dataset = np.load(OUTPUT_PATH)
# except:
#     np.save(OUTPUT_PATH, np.zeros((1, VECTOR_END-VECTOR_START), dtype=np.float32))
#     dataset = np.load(OUTPUT_PATH)
    
# i = 1
# for moon_site in os.listdir(INPUT_PATH):
#     for _folders in os.listdir(os.path.join(INPUT_PATH, moon_site)):
#         if _folders == "train":
#             for image_folder in os.listdir(os.path.join(INPUT_PATH, moon_site, _folders)):
#                 all_paths.append((i, os.path.join(INPUT_PATH, moon_site, _folders, image_folder)))
#                 i += 1
                
# all_paths = sorted(all_paths)

# threads = []

# def solve(index, path):
#     global solved, dataset

#     current = main(path)
#     if current == -1:
#         solved.append(index)
#         print(f"{index} : Skipped: {path}")

#     else:
#         print(f"{index} Processed")
#         while (True):
#             if (index-1 in solved and len(solved) >= index):
#                 break
            
#             continue
        
#         dataset = np.vstack((dataset, current))
#         solved.append(index)
        
#         print(f"{index} : Done for: {path}")


# for index_path in all_paths:    
#     t = threading.Thread(target=solve, args=(index_path[0], index_path[1]))
#     t.start()
#     threads.append(t)

# for t in threads:
#     t.join()

# np.save(OUTPUT_PATH, dataset)