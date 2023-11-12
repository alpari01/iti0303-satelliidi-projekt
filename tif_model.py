import tensorflow as tf
from sklearn.model_selection import train_test_split
import tifffile
from datetime import datetime
import numpy as np
import pandas
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

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

def get_image_size(image_path: str) -> int:
  "Returns image size in MB"
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

  row_kesk = row // 2
  col_kesk = col // 2

  lon = tif[3, row_kesk, col_kesk]
  lat = tif[4, row_kesk, col_kesk]
  time, code = get_time_and_code(image_path)

  meta = [lon, lat, time]

  square_size = int(square_size / 2)

  return tif[0:3, row_kesk-square_size:row_kesk+square_size, col_kesk-square_size:col_kesk+square_size], meta

def get_data(root_path: str, square_size: int, max_image_size: int, dataset_size):
  images = []
  measurements = []

  station_folders = os.listdir(root_path + '/tudengid_imgs')

  for station_folder in station_folders:
    image_paths = os.listdir(os.path.join(root_path + '/tudengid_imgs', station_folder))

    for image_path in image_paths:
        image_size = get_image_size(root_path + '/tudengid_imgs/' + station_folder + '/' + image_path);

        if image_size > max_image_size:
          images.append(read_image(root_path + '/tudengid_imgs/' + station_folder + '/' + image_path, square_size)[0])
          measurements.append(find_hs_class(find_hs_measurement(image_path, root_path)))

          print(len(images), len(measurements), image_size)

          if len(images) == dataset_size:
            # for testing
            return np.array(images), np.array(measurements)
  return None


class TifModel:
  def __init__(self):
    self.features = None
    self.labels = None
    self.model = None
    self.root_path = None

  def set_root_path(self, root_path: str) -> None:
    self.root_path = root_path
    print(f"Updated root path: {self.get_root_path()}")

  def get_root_path(self) -> str:
    return self.root_path

  def get_dataset_info(self) -> str:
    return f"features size is: {len(self.features)}, labels size is: {len(self.labels)}"

  def build_dataset(self, square_size: int=64, max_image_size: int=40, dataset_size: int=10):
    print(f"\nBuilding dataset with parameters: square size: {square_size}px, max image size: {max_image_size}MB, dataset size: {dataset_size}")
    self.features, self.labels = get_data(self.root_path, square_size, max_image_size, dataset_size)
    print("Successfully built dataset")
    print(self.get_dataset_info())

  def prepare_dataset(self) -> None:
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

  def get_summary(self) -> str:
    return self.model.summary()

  def build_model(self) -> None:
    print("\nBuilding CNN model...")
    model_1 = Sequential()
    model_1.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
    model_1.add(MaxPool2D((2, 2)))
    model_1.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model_1.add(MaxPool2D((2, 2)))
    model_1.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model_1.add(Flatten())
    model_1.add(Dense(128, activation='relu'))
    model_1.add(Dense(6, activation='softmax'))
    model_1.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    self.model = model_1
    print("Successfully built model")
    print(self.get_summary())

  def predict_value(self, image_path: str, square_size: int=64) -> str:
    hs_classes_dict = {0: "0-0.5m", 1: "0.51-1m", 2: "1.01-1.5m", 3: "1.51-2m", 4: "2.01-2.5m", 5: "2.51+m"}
    print(f"\nPredicting HS value based on image..., square_size={square_size}")
    x = read_image(image_path, square_size)[0]
    x = x.reshape(1, 64, 64, 3)
    prediction = self.model.predict(x)
    print(f"Actual HS value is: {find_hs_measurement(image_path, self.root_path)}")
    print(f"Prediction 6 classes probability: {prediction}")
    print(f"HS classes dict: {hs_classes_dict}")
    print(f"Predicted HS class: {np.argmax(prediction)} ({hs_classes_dict[np.argmax(prediction)]})")
