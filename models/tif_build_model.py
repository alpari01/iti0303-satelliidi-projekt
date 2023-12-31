import tif_model
import time

start_time = time.time()

model = tif_model.TifModel()
model.tif_images_root_path = "/illukas/data/projects/iti_wave_2023/tif_images"
model.measurements_root_path = "/illukas/data/projects/iti_wave_2023/measurements"
model.pickle_path = "/illukas/data/projects/iti_wave_2023/iti0303-satelliidi-projekt/datasets"
SQUARE_SIZE = 32

model.model_build_lr()
model.model_fit(SQUARE_SIZE)

model.model_build_rf()
model.model_fit(SQUARE_SIZE)

model.model_build_gb()
model.model_fit(SQUARE_SIZE)

model.model_build_knn()
model.model_fit(SQUARE_SIZE)

# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20180909T162907_20180909T162932_S1A_6FBB_S1A_IW_GRDH_1SDV_20180909T162907_20180909T162932_023624_029305_6FBB.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20160919T162830_20160919T162855_S1A_6A27_S1A_IW_GRDH_1SDV_20160919T162830_20160919T162855_013124_014D70_6A27.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20161019T162755_20161019T162820_S1B_B03F_S1B_IW_GRDH_1SDV_20161019T162755_20161019T162820_002578_0045AA_B03F.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20200911T050656_20200911T050721_S1B_1442_S1B_IW_GRDH_1SDV_20200911T050656_20200911T050721_023323_02C4AE_1442.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20211222T162859_20211222T162924_S1A_4C48_S1A_IW_GRDH_1SDV_20211222T162859_20211222T162924_041124_04E2DD_4C48.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20150104T162835_20150104T162900_S1A_4AAE_S1A_IW_GRDH_1SDV_20150104T162835_20150104T162900_004024_004D92_4AAE.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20150305T162834_20150305T162859_S1A_2DB3_S1A_IW_GRDH_1SDV_20150305T162834_20150305T162859_004899_0061B9_2DB3.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20160529T050703_20160529T050728_S1A_1826_S1A_IW_GRDH_1SDV_20160529T050703_20160529T050728_011469_011790_1826.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20160716T050706_20160716T050731_S1A_0AD5_S1A_IW_GRDH_1SDV_20160716T050706_20160716T050731_012169_012DE3_0AD5.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20170512T050708_20170512T050733_S1A_242C_S1A_IW_GRDH_1SDV_20170512T050708_20170512T050733_016544_01B6D2_242C.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20170529T162855_20170529T162920_S1A_9CFC_S1A_IW_GRDH_1SDV_20170529T162855_20170529T162920_016799_01BEB0_9CFC.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20170909T050719_20170909T050744_S1A_64C4_S1A_IW_GRDH_1SDV_20170909T050719_20170909T050744_018294_01EC47_64C4.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20180822T162801_20180822T162826_S1B_6A32_S1B_IW_GRDH_1SDV_20180822T162801_20180822T162826_012378_016D21_6A32.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20181009T162802_20181009T162827_S1B_BBB1_S1B_IW_GRDH_1SDV_20181009T162802_20181009T162827_013078_01829F_BBB1.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20181120T162907_20181120T162932_S1A_A663_S1A_IW_GRDH_1SDV_20181120T162907_20181120T162932_024674_02B648_A663.tif", 64)
# model.predict_value("/illukas/data/projects/iti_wave_2023/tif_images/knolls_gcp_2/knolls_gcp_2_20190102T050726_20190102T050751_S1A_D1CF_S1A_IW_GRDH_1SDV_20190102T050726_20190102T050751_025294_02CC4C_D1CF.tif", 64)

end_time = time.time()

print(f'Algorithm execution time {round(end_time - start_time, 2)} sec.')
