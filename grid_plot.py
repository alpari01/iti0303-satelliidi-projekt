import read_geotiff
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_grid(data, max_square, min_square, filling=0.999, prev_row_start=0, prev_col_start=0):
    """Divides the landmask into a grid and marks squares that meet a specified land coverage threshold."""
    height = len(data)
    width = len(data[0])
    if max_square < min_square:
        return

    result_lists = []
    for row_start in range(0, height - (height % max_square), max_square):
        for col_start in range(0, width - (width % max_square), max_square):
            sub_list = []
            for i in range(row_start, row_start + max_square):
                sub_list.append(data[i][col_start:col_start + max_square])

            amount = sum(line.count(1) for line in sub_list)

            if amount > (max_square * max_square * filling):
                matrix = np.array(sub_list, dtype=float)
                matrix[0, :] = 0.5
                matrix[-1, :] = 0.5
                matrix[:, 0] = 0.5
                matrix[:, -1] = 0.5
                sub_list = matrix.tolist()
                top_left = (row_start + prev_row_start, col_start + prev_col_start)
                bottom_right = (row_start + prev_row_start + max_square - 1,
                                col_start + prev_col_start + max_square - 1)
                square_indices.append([top_left, bottom_right])
            else:
                draw_grid(sub_list, int(max_square / 2), min_square, filling,
                          row_start + prev_row_start, col_start + prev_col_start)
            result_lists.append(sub_list)

    for row_start in range(0, height - (height % max_square), max_square):
        for col_start in range(0, width - (width % max_square), max_square):
            sub_list = result_lists.pop(0)
            for i in range(max_square):
                for j in range(max_square):
                    data[row_start + i][col_start + j] = sub_list[i][j]


def transfer_indices(from_where, to_where):
    """Transfers the marked indices from the landmask to the raster band for visualization."""
    from_where_array = np.array(from_where, dtype=float)
    to_where_array = np.array(to_where, dtype=float)

    indices = np.where(from_where_array == 0.5)
    to_where_array[indices] = 3000

    return to_where_array


if __name__ == '__main__':
    tif_file = r'./s1a-iw-grd-vv-20221027t160500-20221027t160525-045630-0574c2-001.tiff'
    shapefile = r'./gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp'

    print('Program started...')

    landmask = read_geotiff.get_landmask_and_error(tif_file, shapefile)
    landmask = np.array(landmask).tolist()
    print('...successfully received landmask...')

    raster_band = read_geotiff.get_raster_band(tif_file)
    raster_band = np.array(raster_band).tolist()
    print('...successfully received raster_band...')

    square_indices = []
    draw_grid(landmask, 512, 32)
    print('...successfully drawn grid...')

    result_data = transfer_indices(landmask, raster_band)
    print('...successfully transferred indices...')

    aspect_ratio = result_data.shape[1] / result_data.shape[0]
    fig = plt.figure(figsize=(6 * aspect_ratio, 6))
    sns.heatmap(result_data, cmap='gray', vmin=0, vmax=1000, cbar=False)
    plt.savefig('plot.png', dpi=5000)
    plt.show()
    print('...successfully created plot.')
