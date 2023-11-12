import tif_model
import time

start_time = time.time()

model = tif_model.TifModel()
model.root_path = "/illukas/data/projects/iti_wave_2023"

model.build_dataset(64, 40, 900)
model.model_build()
model.model_fit()
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20180909T162907_20180909T162932_S1A_6FBB_S1A_IW_GRDH_1SDV_20180909T162907_20180909T162932_023624_029305_6FBB.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20160919T162830_20160919T162855_S1A_6A27_S1A_IW_GRDH_1SDV_20160919T162830_20160919T162855_013124_014D70_6A27.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20161019T162755_20161019T162820_S1B_B03F_S1B_IW_GRDH_1SDV_20161019T162755_20161019T162820_002578_0045AA_B03F.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20200911T050656_20200911T050721_S1B_1442_S1B_IW_GRDH_1SDV_20200911T050656_20200911T050721_023323_02C4AE_1442.tif", 64)
model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20211222T162859_20211222T162924_S1A_4C48_S1A_IW_GRDH_1SDV_20211222T162859_20211222T162924_041124_04E2DD_4C48.tif", 64)


end_time = time.time()

print(f'Algorithm execution time {round(end_time - start_time, 2)} sec.')
