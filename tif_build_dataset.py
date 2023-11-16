import tif_model
import time

start_time = time.time()

model = tif_model.TifModel()
model.tif_images_root_path = "/illukas/data/projects/iti_wave_2023/tif_images"
model.measurements_root_path = "/illukas/data/projects/iti_wave_2023/measurements"
model.pickle_path = "/illukas/data/projects/iti_wave_2023/iti0303-satelliidi-projekt"
model.build_dataset(square_size=64, max_image_size_mb=40, dataset_size=100)

end_time = time.time()

print(f'Algorithm execution time {round(end_time - start_time, 2)} sec.')
