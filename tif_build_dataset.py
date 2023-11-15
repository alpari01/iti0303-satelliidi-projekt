import tif_model
import time

start_time = time.time()

model = tif_model.TifModel()
model.tif_images_root_path = "/illukas/data/projects/iti_wave_2023/tif_images"
model.measurements_root_path = "/illukas/data/projects/iti_wave_2023/measurements"
model.pickle_path = "/illukas/data/projects/iti_wave_2023/iti0303-satelliidi-projekt/data.pkl"

model.build_dataset(64, 40, 100)

end_time = time.time()

print(f'Algorithm execution time {round(end_time - start_time, 2)} sec.')
