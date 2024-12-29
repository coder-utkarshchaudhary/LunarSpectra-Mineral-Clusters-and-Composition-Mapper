import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from matplotlib import pyplot as plt

VECTOR_START = 7
VECTOR_END = 115
NUM_CLASSES = 163
NUM_EPOCHS = 30

def match_spectrums(pixel_arrays, look_up_table, ground_truth_for_cnn_path):
    def similarity_algorithm(sample_array_1, sample_array_2):    
        def CPRMS(pixel_vector, reference_vector):
            mean_pixel = np.mean(pixel_vector)
            mean_reference = np.mean(reference_vector)
            centered_pixel = pixel_vector - mean_pixel
            centered_reference = reference_vector - mean_reference
            squared_diff = (centered_pixel - centered_reference) ** 2
            val = np.sqrt(np.mean(squared_diff))
            
            return val

        def ABS(f_n, r_n):
            return np.sum(np.abs(f_n - r_n))

        return CPRMS(sample_array_1, sample_array_2) + 100*ABS(sample_array_1, sample_array_2)

    
    for i in range(pixel_arrays.shape[0]):
        similarity_vector = np.zeros((NUM_CLASSES,), dtype=np.float32)
        for j in range(look_up_table.shape[0]):
            similarity_vector[j] = similarity_algorithm(pixel_arrays[i], look_up_table[j])

        np.save(os.path.join(ground_truth_for_cnn_path, f"{i}.npy"), similarity_vector)

    return

def plot_spectra_in_pixel_array(pixel_arrays, output_path):
    for i in range(pixel_arrays.shape[0]):
        _ = plt.figure(figsize=(50, 30))
        plt.plot(pixel_arrays[i], color="orange", linewidth=2.5)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, f"{i}.png"))
        plt.close()


def cnn_model(input_shape):
    def hybrid_pooling(_input, pool_size=(2, 2)):
        avg_pool = layers.AveragePooling2D(pool_size=pool_size, padding="same", strides=(1,1))(_input)
        max_pool = layers.MaxPooling2D(pool_size=pool_size, padding="same", strides=(1,1))(_input)
        return layers.Concatenate()([avg_pool, max_pool])

    def inception_block(_input, num_filter):
        x1 = layers.Conv2D(num_filter, (1,1), activation=tf.nn.gelu, padding="same")(_input)
        x2 = layers.Conv2D(num_filter, (3,3), activation=tf.nn.gelu, padding="same")(_input)
        x3 = layers.Conv2D(num_filter, (5,5), activation=tf.nn.gelu, padding="same")(_input)
        x4 = hybrid_pooling(_input=_input)
        x4 = layers.Conv2D(num_filter, (1,1), activation=tf.nn.gelu, padding="same")(x4)
        output_for_inception_block = layers.Concatenate()([x1,x2,x3,x4])
        return output_for_inception_block

    _inputs = layers.Input(shape=input_shape)
    x1 = layers.Conv2D(16, (1,1), activation=tf.nn.gelu, padding="same")(_inputs)
    x2 = layers.Conv2D(16, (3,3), activation=tf.nn.gelu, padding="same")(_inputs)
    x3 = layers.Conv2D(16, (5,5), activation=tf.nn.gelu, padding="same")(_inputs)
    x4 = hybrid_pooling(_input=_inputs)
    x = layers.Concatenate()([x1,x2,x3,x4])

    x = layers.Dropout(0.3)
    
    x = inception_block(x, 32)
    x = hybrid_pooling(x)

    x = layers.Dropout(0.3)
    
    x = inception_block(x, 16)
    x = hybrid_pooling(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation=tf.nn.gelu, kernel_regularizer=tf.keras.Regularizer.L2(0.03))(x)
    x = layers.Dense(64, activation=tf.nn.gelu, kernel_regularizer=tf.keras.Regularizer.L2(0.01))(x)
    _outputs = layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)(x)

    model = Model(_inputs, _outputs)
    return model

def custom_loss_function(y_true, y_pred):
    kl_loss = KLDivergence()(y_true, y_pred)
    categorical_loss = CategoricalCrossentropy()(y_true, y_pred)

    return 0.4*kl_loss+0.6*categorical_loss

def create_X_for_model(path_to_spectral_images):
    X = []
    for _file in os.listdir(path_to_spectral_images):
        X.append(cv.imread(os.path.join(path_to_spectral_images, _file)))

    return np.asarray(X)

def create_y_for_model(path_to_spectral_mapping_to_table):
    y = []
    for _file in os.listdir(path_to_spectral_mapping_to_table):
        y.append(np.load(os.path.join(path_to_spectral_mapping_to_table, _file)))

    return np.asarray(y)

class CustomDataset(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        
    def custom_preprocessing_x_y(self, img, label):
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            label = tf.image.flip_left_right(label)
        return img, label
    
    def flow(self, x, y, shuffle=True):
        batches = super().flow(x, y, batch_size=self.batch_size, shuffle=shuffle)

        while True:
            batch_x, batch_y = next(batches)
            yield batch_x, batch_y

if __name__ == "__main__":
    PIXEL_ARRAY_PATH = r"pixel_vector_data_train_files_transformed.npy"
    SPECTRAL_IMAGES_PATH = r"spectral_files"
    LOOK_UP_TABLE = r"combined_output.npy"
    SPECTRAL_MAPPING_PATH = r"ground_truth_for_cnn"

    pixel_arrays = np.load(PIXEL_ARRAY_PATH)

    pixel_arrays = pixel_arrays[:12851*250]

    print("Got pixel_arrays...")

    plot_spectra_in_pixel_array(pixel_arrays, SPECTRAL_IMAGES_PATH)
    print("Plotted Spectra in pixel array...")
    match_spectrums(pixel_arrays, np.load(LOOK_UP_TABLE), SPECTRAL_MAPPING_PATH)
    print("Spectrum Matching Done ...")

    # X and y for model training
    X = create_X_for_model(SPECTRAL_IMAGES_PATH)
    y = create_y_for_model(SPECTRAL_MAPPING_PATH)


    np.save("X.npy", X)
    np.save("y.npy", y)
    del X
    del y

    print("X, y saved as .npy file")
    X = np.load("X.npy")
    y = np.load("y.npy")


    X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)

    create_generator = CustomDataset(3)
    train_dataset = create_generator.flow(X_train, y_train)
    valid_dataset = create_generator.flow(X_valid, y_valid)

    # CNN model instantiate
    cnn = cnn_model(input_shape=X[0].shape)
    cnn.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3), loss=custom_loss_function)
    print("Training Started...")
    at_training = cnn.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=valid_dataset, steps_per_epoch=len(X_train)//3, validation_steps=len(X_valid)//3)

    # Plot training & validation loss values
    plt.plot(at_training.history['loss'])
    plt.plot(at_training.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    cnn.save("CNN_model.keras")