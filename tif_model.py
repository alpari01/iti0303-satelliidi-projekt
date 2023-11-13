import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tifffile
from datetime import datetime
import numpy as np
import pandas
import os
import sys
import psutil


def get_time_and_code(image_path: str):
    path, filename = os.path.split(image_path)
    time = filename.split('_')[3]
    code = filename.split('_')[0]
    return datetime.strptime(time, '%Y%m%dT%H%M%S'), code


def find_hs_class(hs: np.float64) -> int:
    if 0 <= hs < 0.5:
        return 0
    if 0.5 <= hs < 1.0:
        return 1
    if 1.0 <= hs < 1.5:
        return 2
    if 1.5 <= hs < 2.0:
        return 3
    if 2.0 <= hs < 2.5:
        return 4
    if 2.5 <= hs:
        return 5


def find_hs_measurement(image_path: str, root_path: str) -> np.float64:
    """
    - root_path/
      - measurements/
      - shapefiles/
      - temp/
      - tudengid_imgs/
    """
    time, code = get_time_and_code(image_path)
    csv_data = pandas.read_csv(root_path + '/measurements/format_' + code + '.csv')
    csv_data['Format time (UTC)'] = pandas.to_datetime(csv_data['Format time (UTC)'])
    closest_time_index = (csv_data['Format time (UTC)'] - time).abs().idxmin()
    hs = csv_data.loc[closest_time_index, 'HS']
    return hs


def get_image_size(image_path: str) -> float:
    """Returns image size in MB"""
    image_size = os.path.getsize(image_path)
    return image_size / 1024 / 1024


def read_image(image_path: str, square_size: int):
    """
      Read a TIFF image file and return a cropped portion along with metadata.

      Args:
          image_path (str): The path to the TIFF image file to be read.
          square_size (int): The size of the square area to be cropped around the center.

      Returns:
          np.array: A numpy array containing the cropped image data.
    """
    tif = tifffile.imread(image_path)
    dim, row, col = tif.shape

    row_mid = row // 2
    col_mid = col // 2

    lon = tif[3, row_mid, col_mid]
    lat = tif[4, row_mid, col_mid]
    time, code = get_time_and_code(image_path)

    meta = [lon, lat, time]

    square_size = int(square_size / 2)

    return tif[0:3, row_mid - square_size:row_mid + square_size, col_mid - square_size:col_mid + square_size], meta


def get_data(root_path: str, square_size: int, max_image_size: int, dataset_size):
    images = []
    measurements = []

    station_folders = os.listdir(root_path + '/tif_images')

    for station_folder in station_folders:
        image_paths = os.listdir(os.path.join(root_path + '/tif_images', station_folder))

        for image_path in image_paths:
            image_size = get_image_size(root_path + '/tif_images/' + station_folder + '/' + image_path)

            if image_size > max_image_size:
                try:
                    images.append(
                        read_image(root_path + '/tif_images/' + station_folder + '/' + image_path, square_size)[0])
                    measurements.append(find_hs_class(find_hs_measurement(image_path, root_path)))

                    print(len(images), len(measurements), image_size, psutil.cpu_percent(),
                          f" | RAM in use: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB",
                          f" | RAM avail: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
                    sys.stdout.flush()

                    if len(images) == dataset_size:
                        # for testing
                        return np.array(images), np.array(measurements)
                except Exception as e:
                    print("Could not read image, skipping.")
    return None


class TifModel:
    def __init__(self):
        self.features = None
        self.labels = None
        self.model = None
        self.root_path = None

    def build_dataset(self, square_size: int = 64, max_image_size: int = 40, dataset_size: int = 10):
        print(f"\nBuilding dataset with parameters: square size: {square_size}px, max image size: {max_image_size}MB, dataset size: {dataset_size}")
        self.features, self.labels = get_data(self.root_path, square_size, max_image_size, dataset_size)
        self.labels = keras.utils.to_categorical(self.labels, num_classes=6)
        print("Successfully built dataset")
        print(f"Features size is: {len(self.features)}, labels size is: {len(self.labels)}")

    def model_build(self) -> None:
        print("\nBuilding CNN model...")
        model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(8, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(6, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        print("Successfully built model")
        print(self.model.summary())

    def model_fit(self) -> None:
        print("\nSplitting dataset into training, testing and validation...")
        X_train, X_temp, y_train, y_temp = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])
        X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])
        X_val = tf.transpose(X_test, perm=[0, 2, 3, 1])
        print("Successfully prepared dataset")
        print(f'test size={len(X_train)}')
        print(f'train size={len(X_test)}')
        print(f'val size={len(X_val)}')
        self.model.fit(X_train, y_train,
                       epochs=64, batch_size=64,
                       validation_data=(X_test, y_test))

    def predict_value(self, image_path: str, square_size: int = 64) -> str:
        hs_classes_dict = {0: "0-0.5m", 1: "0.51-1m", 2: "1.01-1.5m", 3: "1.51-2m", 4: "2.01-2.5m", 5: "2.51+m"}
        print(f"\nPredicting HS value based on image..., square_size={square_size}")
        x = read_image(image_path, square_size)[0]
        x = x.reshape(1, 64, 64, 3)
        prediction = self.model.predict(x)
        print(f"Actual HS value is: {find_hs_measurement(image_path, self.root_path)}")
        print(f"Prediction 6 classes probability: {prediction}")
        print(f"HS classes dict: {hs_classes_dict}")
        print(f"Predicted HS class: {np.argmax(prediction)} ({hs_classes_dict[np.argmax(prediction)]})")
