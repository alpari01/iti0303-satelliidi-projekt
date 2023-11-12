import tif_model
import time

start_time = time.time()

model = tif_model.TifModel()
model.root_path = "/illukas/data/projects/iti_wave_2023"

model.build_dataset(64, 40, 10)
model.model_build()
model.model_fit()
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/gof_gcp_2/gof_gcp_2_20211008T045036_20211008T045101_S1A_3916_S1A_IW_GRDH_1SDV_20211008T045036_20211008T045101_040023_04BCCC_3916.tif", 64)

end_time = time.time()

print(f'Algorithm execution time {round(end_time - start_time, 2)} sec.')