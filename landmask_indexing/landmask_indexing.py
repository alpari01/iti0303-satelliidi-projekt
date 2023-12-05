import read_geotiff
import time
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from models import tif_model


def read_image(original_tif, top_left, bottom_right):
    return original_tif[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, 0:3]


def calculate_image_metrics(image_array):
    mean_value = np.mean(image_array)
    std_value = np.std(image_array)
    percentile_25 = np.percentile(image_array, 25)
    percentile_75 = np.percentile(image_array, 75)
    return np.array([mean_value, std_value, percentile_25, percentile_75])


def get_indices(original_tif, data, model, max_square, min_square, filling=0.999, prev_row_start=0, prev_col_start=0):
    height = len(data)
    width = len(data[0])
    if max_square < min_square:
        return

    for row_start in range(0, height - (height % max_square), max_square):
        for col_start in range(0, width - (width % max_square), max_square):
            sub_list = []
            for i in range(row_start, row_start + max_square):
                sub_list.append(data[i][col_start:col_start + max_square])

            amount = sum(line.count(1) for line in sub_list)

            if amount > (max_square * max_square * filling):
                top_left = (row_start + prev_row_start, col_start + prev_col_start)
                bottom_right = (row_start + prev_row_start + max_square - 1,
                                col_start + prev_col_start + max_square - 1)

                image = read_image(original_tif, top_left, bottom_right)
                features = calculate_image_metrics(image).tolist()
                X = pd.DataFrame([features], columns=["mean_value", "std_value", "percentile_25", "percentile_75"])
                indices[top_left, bottom_right] = model.model.predict(X)

            else:
                get_indices(original_tif, sub_list, model, int(max_square / 2), min_square, filling,
                            row_start + prev_row_start, col_start + prev_col_start)


if __name__ == '__main__':
    tif_file = r'landmask_indexing/tiff/S1A_IW_GRDH_1SDV_20221027T160500_20221027T160525_045630_0574C2_211B_Cal_Spk_TC.tif'
    shapefile = r'landmask_indexing/tiff/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp'
    datasets_folder = r'models/datasets'

    print('Program started...')
    start_time = time.time()

    tif = tifffile.imread(tif_file)
    print(f'...successfully read .tif file with shape {tif.shape}...')

    landmask = read_geotiff.get_landmask_and_error(tif_file, shapefile)
    landmask = np.array(landmask).tolist()
    print('...successfully received landmask...')

    model_class = tif_model.TifModel()
    model_class.model_build_rf()
    # model_class.model_fit_multiple(datasets_folder)
    model_class.model_fit(512, datasets_folder)
    print('...successfully trained model...')

    indices = {}
    get_indices(tif, landmask, model_class, 512, 32)
    print('...successfully found indices...')

    # for key, value in indices.items():
    #     print(f'{key}: {value},')

    landmask = np.array(landmask)
    aspect_ratio = landmask.shape[1] / landmask.shape[0]
    fig = plt.figure(figsize=(6 * aspect_ratio, 6))
    heatmap = sns.heatmap(landmask, cmap='gray', vmin=0, vmax=1, cbar=False)
    heatmap.set_xticks([])
    heatmap.set_yticks([])

    for key, value in indices.items():
        square_size = key[1][0] - key[0][0] + 1
        x = int((key[0][1] + key[1][1]) / 2) + 1
        y = int((key[0][0] + key[1][0]) / 2) + 1
        fontsize = int(math.log2(square_size)) - 4
        plt.text(x, y, str(value[0]), color='black', ha='center', va='center', fontsize=fontsize, fontweight='normal')
        colors = {5: '#eb3434', 4: '#eb6834', 3: '#eb9934', 2: '#ebcd34', 1: '#d9eb34', 0: '#83eb34'}
        plt.gca().add_patch(plt.Rectangle((x - square_size // 2, y - square_size // 2), square_size, square_size, linewidth=0.1, edgecolor='white', facecolor=colors[int(value[0])]))
    print('...HS classes and colors successfully plotted...')

    plt.savefig('plot.png', dpi=2500, bbox_inches='tight', pad_inches=0)
    plt.show()
    print('...successfully drawn plot.')

    print(f'...algorithm execution time {round(time.time() - start_time, 2)} sec.')