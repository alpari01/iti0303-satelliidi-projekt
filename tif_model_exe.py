import tif_model
import time

start_time = time.time()

model = tif_model.TifModel()
model.root_path = "/illukas/data/projects/iti_wave_2023"

model.build_dataset(64, 40, 900)
model.model_build()
model.model_fit()
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/gof_gcp_2/gof_gcp_2_20211008T045036_20211008T045101_S1A_3916_S1A_IW_GRDH_1SDV_20211008T045036_20211008T045101_040023_04BCCC_3916.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/gof_gcp_2/gof_gcp_2_20220124T045032_20220124T045057_S1A_21D4_S1A_IW_GRDH_1SDV_20220124T045032_20220124T045057_041598_04F2AC_21D4.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/gof_gcp_2/gof_gcp_2_20220224T044225_20220224T044250_S1A_7AED_S1A_IW_GRDH_1SDV_20220224T044225_20220224T044250_042050_050247_7AED.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/gof_gcp_2/gof_gcp_2_20220325T045032_20220325T045057_S1A_1E38_S1A_IW_GRDH_1SDV_20220325T045032_20220325T045057_042473_0510A6_1E38.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/gof_gcp_2/gof_gcp_2_20181223T160459_20181223T160524_S1A_C476_S1A_IW_GRDH_1SDV_20181223T160459_20181223T160524_025155_02C74E_C476.tif", 64)


end_time = time.time()

print(f'Algorithm execution time {round(end_time - start_time, 2)} sec.')
