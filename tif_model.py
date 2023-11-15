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
import traceback
import csv


def get_time_and_code(image_path: str):
    path, filename = os.path.split(image_path)
    time = filename.split('_')[3]
    code = filename.split('_')[0]
    return datetime.strptime(time, '%Y%m%dT%H%M%S'), code


def find_hs_class(hs: np.float64) -> int:
    if hs < 0.5:
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


def find_hs_measurement(image_path: str, measurements_root_path: str) -> np.float64:
    """
    - root_path/
      - measurements/
      - shapefiles/
      - temp/
      - tudengid_imgs/
    """
    # /content/drive/MyDrive/TalTech/Tellimus/measurements
    time, code = get_time_and_code(image_path)
    csv_data = pandas.read_csv(measurements_root_path + '/format_' + code + '.csv')
    # csv_data = pandas.read_csv(root_path + '/measurements/format_' + code + '.csv')
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

    return tif[0:3, row_mid - square_size:row_mid + square_size, col_mid - square_size:col_mid + square_size]


class TifModel:
    def __init__(self):
        self.features = []
        self.labels = []
        self.hs_classes_counter = {}
        self.model = None
        self.tif_images_root_path = None
        self.measurements_root_path = None

    def get_dataset_info(self) -> str:
        return f"features size is: {len(self.features)}, labels size is: {len(self.labels)}"

    def add_image_class_to_counter(self, hs_class: int) -> None:
        if hs_class in self.hs_classes_counter.keys():
            self.hs_classes_counter[hs_class] += 1
        else:
            self.hs_classes_counter[hs_class] = 1

    def get_images_class_counter_stats(self) -> str:
        res = "Images per HS class (0 - 5):\n"
        for hs_class, amount in self.hs_classes_counter.items():
            if hs_class == 0:
                res += f"0m - 0.5m: {amount}\n"
            if hs_class == 1:
                res += f"0.51m - 1.0m: {amount}\n"
            if hs_class == 2:
                res += f"1.01m - 1.5m: {amount}\n"
            if hs_class == 3:
                res += f"1.51m - 2.0m: {amount}\n"
            if hs_class == 4:
                res += f"2.01m - 2.5m: {amount}\n"
            if hs_class == 5:
                res += f"2.51m - inf: {amount}\n"
        return res

    def build_dataset(self, square_size: int, max_image_size_mb: int, dataset_size: int):
        """
        This function builds a dataset for a CNN model. It reads .tif images and their measurements,
        then converts each image to np.array and saves the data class variables.

        tif_images_root_path/
            - gof_gcp_2/
                - gof_gcp_2_xxxxx1.tif
                - gof_gcp_2_xxxxx2.tif
                - ...
            - knolls_gcp_2/
            - nbp_gcp_2/
            - selka_gcp_2/

        measurements_root_path/
            - format_gof.csv
            - format_knolls.csv
            - format_nbp.csv
            - format_selks.csv

        square size: int - image size to cut (32px, 64px, 128px, 256px, or 512px)
        max_image_size_mb: int - all images below this size (in MB) are discarded
        dataset_size: int - amount of images to convert
        """
        prepared_images = 0

        measurements_folders = os.listdir(self.tif_images_root_path)
        for measurement_folder in measurements_folders:

            image_paths = os.listdir(os.path.join(self.tif_images_root_path, measurement_folder))
            for image_path in image_paths:

                image_size = get_image_size(self.tif_images_root_path + "/" + measurement_folder + "/" + image_path)
                if image_size >= max_image_size_mb:
                    try:
                        measurement = find_hs_class(find_hs_measurement(image_path, self.measurements_root_path))
                        if measurement:
                            image = read_image(self.tif_images_root_path + "/" + measurement_folder + "/" + image_path, square_size)

                            self.features.append(image)
                            self.labels.append(measurement)

                            prepared_images += 1
                            print(
                                f"image nr: {prepared_images}",
                                f"img size: {round(image_size, 2)} MB,",
                                f"HS class: {measurement}",
                                f" | RAM in use: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB",
                                f" RAM avail: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
                                f"image path: {image_path}"
                            )
                            sys.stdout.flush()

                        if prepared_images == dataset_size:
                            print("Dataset built!")
                            return
                            # return np.array(images), np.array(measurements)
                    except Exception as e:
                      traceback.print_exc()
                      print(f"Could not read image, skipping.\nError: {e}")
                      return

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
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.labels = keras.utils.to_categorical(self.labels, num_classes=6)
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
        x = read_image(image_path, square_size)
        x = x.reshape(1, 64, 64, 3)
        prediction = self.model.predict(x)
        print(f"Actual HS value is: {find_hs_measurement(image_path, self.measurements_root_path)}")
        print(f"Prediction 6 classes probability: {prediction}")
        print(f"HS classes dict: {hs_classes_dict}")
        print(f"Predicted HS class: {np.argmax(prediction)} ({hs_classes_dict[np.argmax(prediction)]})")
