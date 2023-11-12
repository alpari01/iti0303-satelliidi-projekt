import tif_model

model = tif_model.TifModel()
model.set_root_path("/illukas/data/projects/iti_wave_2023")

model.build_dataset(64, 40, 10)
model.build_model()
model.model_fit()
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/gof_gcp_2/gof_gcp_2_20211008T045036_20211008T045101_S1A_3916_S1A_IW_GRDH_1SDV_20211008T045036_20211008T045101_040023_04BCCC_3916.tif", 64)