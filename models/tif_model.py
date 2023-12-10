import tifffile
from datetime import datetime
import numpy as np
import pandas
import os
import sys
import traceback
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def get_time_and_code(image_path: str) -> tuple:
    path, filename = os.path.split(image_path)
    time = filename.split('_')[3]
    code = filename.split('_')[0]
    return datetime.strptime(time, '%Y%m%dT%H%M%S'), code


def find_hs_class(hs: np.float64) -> int:
    if hs <= 0.5:
        return 0
    if 0.5 < hs <= 1.0:
        return 1
    if 1.0 < hs <= 1.5:
        return 2
    if 1.5 < hs <= 2.0:
        return 3
    if 2.0 < hs <= 2.5:
        return 4
    if 2.5 < hs:
        return 5


def find_hs_measurement(image_path: str, measurements_root_path: str) -> np.float64:
    """
    - root_path/
      - measurements/
      - shapefiles/
      - temp/
      - tudengid_imgs/
    """
    time, code = get_time_and_code(image_path)
    csv_data = pandas.read_csv(measurements_root_path + '/format_' + code + '.csv')
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


def calculate_image_metrics(image_array: np.ndarray) -> np.ndarray:
    mean_value = np.mean(image_array)
    std_value = np.std(image_array)
    percentile_25 = np.percentile(image_array, 25)
    percentile_75 = np.percentile(image_array, 75)
    return np.array([mean_value, std_value, percentile_25, percentile_75])


def plot_confusion_matrix(y_true, y_pred, square_size: any, model_type: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
    class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({model_type},  {square_size})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix-{square_size}px-{model_type}.png", dpi=300)
    plt.show()
    plt.clf()


def save_to_pickle(tif_metrics: np.array, measurements: np.ndarray, pickle_path: str) -> None:
    with open(pickle_path, "wb") as file:
        pickle.dump({'tif_metrics': tif_metrics, 'measurements': measurements}, file)


def read_from_pickle(pickle_path: str) -> tuple:
    with open(pickle_path, "rb") as file:
        data = pickle.load(file)
        tif_metrics = data['tif_metrics']
        measurements = data['measurements']
    return tif_metrics.tolist(), measurements.tolist()


class TifModel:
    def __init__(self):
        self.hs_classes_counter = {}
        self.model = None
        self.tif_images_root_path = None
        self.measurements_root_path = None
        self.pickle_path = None
        self.tif_metrics = []
        self.measurements = []
        self.model_type = None

    def build_dataset(self, square_size: int, max_image_size_mb: int, dataset_size: int):
        """
        This function builds a dataset for a model. It reads .tif images and their measurements,
        then converts each image to np.ndarray and saves the data class variables.

        After dataset is built, this function will save it to .pkl file.

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
        dataset_size: int - amount of images to include to dataset
        """
        prepared_images = 0
        read_images = 0
        checked_images = 0

        classes_counter = {}

        measurements_folders = os.listdir(self.tif_images_root_path)
        for measurement_folder in measurements_folders:

            image_paths = os.listdir(os.path.join(self.tif_images_root_path, measurement_folder))
            for image_path in image_paths:
                checked_images += 1
                image_size = get_image_size(self.tif_images_root_path + "/" + measurement_folder + "/" + image_path)
                if image_size >= max_image_size_mb:
                    read_images += 1
                    try:
                        measurement = find_hs_class(find_hs_measurement(image_path, self.measurements_root_path))
                        if isinstance(measurement, int):

                            if measurement in classes_counter:
                                if classes_counter[measurement] < dataset_size / 6:
                                    classes_counter[measurement] += 1
                                    image = read_image(self.tif_images_root_path + "/" + measurement_folder + "/" + image_path, square_size)
                                    self.tif_metrics.append(calculate_image_metrics(image))
                                    self.measurements.append(measurement)
                                    prepared_images += 1
                            else:
                                classes_counter[measurement] = 1
                                image = read_image(self.tif_images_root_path + "/" + measurement_folder + "/" + image_path, square_size)
                                self.tif_metrics.append(calculate_image_metrics(image))
                                self.measurements.append(measurement)
                                prepared_images += 1

                            print(
                                f"image nr: {prepared_images} - {read_images} - {checked_images}",
                                f"img size: {round(image_size, 2)} MB,",
                                f"HS class: {measurement}",
                                f"image path: {image_path}"
                            )

                            print(classes_counter)

                            sys.stdout.flush()  # this is needed for ai-lab to display prints during runtime

                        if prepared_images == dataset_size:
                            save_to_pickle(np.array(self.tif_metrics), np.array(self.measurements), self.pickle_path + f"/data-{square_size}px.pkl")
                            print("Dataset built and saved!")
                            return

                    except Exception as e:
                        traceback.print_exc()
                        print(f"Could not read image, skipping.\nError: {e}")

        save_to_pickle(np.array(self.tif_metrics), np.array(self.measurements), self.pickle_path + f"/data-{square_size}px.pkl")
        print(f"Didn't reach {dataset_size}, saving anyway. Total images: {prepared_images}")
        print("Dataset built and saved!")

    def model_build_lr(self) -> None:
        print("\nBuilding Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        self.model = model
        self.model_type = "lr"
        print("Successfully built model")

    def model_build_rf(self) -> None:
        print("\nBuilding Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = model
        self.model_type = "rf"
        print("Successfully built model")

    def model_build_gb(self) -> None:
        print("\nBuilding Gradient Boosting model...")
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model = model
        self.model_type = "gb"
        print("Successfully built model")

    def model_build_knn(self) -> None:
        print("\nBuilding k-Nearest Neighbors model...")
        model = KNeighborsClassifier(n_neighbors=5)
        self.model = model
        self.model_type = "knn"
        print("Successfully built model")

    def model_fit(self, square_size: int, datasets_path: str) -> None:
        """
        This method fits model with dataset of images of one particular square size
        (e.g. dataset that only contains images of 32x32).
        Method finds dataset file automatically based on square_size.
        """
        print("\nSplitting dataset into training, testing and validation...")
        features, labels = read_from_pickle(datasets_path + f"/data-{square_size}px.pkl")
        features = pandas.DataFrame(features, columns=["mean_value", "std_value", "percentile_25", "percentile_75"])
        labels = pandas.Series(labels, name="HS_class")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        print("Successfully prepared dataset")
        print(f'test size={len(X_test)}')
        print(f'train size={len(X_train)}')

        self.model.fit(X_train, y_train)

        y_true = y_test
        y_pred = self.model.predict(X_test)
        plot_confusion_matrix(y_true, y_pred, square_size, self.model_type)
        print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], zero_division=1))

    def model_fit_multiple(self, datasets_path: str) -> None:
        """
        Same method as model_fit, but this method fits model with ALL datasets it finds in the path.
        """
        print("\nSplitting dataset into training, testing and validation...")

        dataset_files = os.listdir(datasets_path)
        features = []
        labels = []
        for dataset_file in dataset_files:
          data = read_from_pickle(datasets_path + "/" + dataset_file)
          features += data[0]
          labels += data[1]

        features = pandas.DataFrame(features, columns=["mean_value", "std_value", "percentile_25", "percentile_75"])
        labels = pandas.Series(labels, name="HS_class")
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        print("Successfully prepared dataset")
        print(f'test size={len(X_test)}')
        print(f'train size={len(X_train)}')

        self.model.fit(X_train, y_train)

        y_true = y_test
        y_pred = self.model.predict(X_test)
        plot_confusion_matrix(y_true, y_pred, "32px - 512px", self.model_type)
        print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5], zero_division=1))
