import read_geotiff
import time
import numpy


def get_indices(data, max_square, min_square, filling=0.999, prev_row_start=0, prev_col_start=0):
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
                indices.append([top_left, bottom_right])
            else:
                get_indices(sub_list, int(max_square / 2), min_square, filling,
                            row_start + prev_row_start, col_start + prev_col_start)


if __name__ == '__main__':
    tif_file = r'tiff/s1b-ew-grd-hh-20191106t160337-20191106t160441-018809-023761-001.tiff'
    shapefile = r'tiff/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp'

    print('Program started...')

    landmask = read_geotiff.get_landmask_and_error(tif_file, shapefile)
    landmask = numpy.array(landmask).tolist()
    print('...successfully received landmask...')

    start_time = time.time()
    indices = []
    get_indices(landmask, 512, 32)
    # indices.sort(key=lambda x: (x[1][0] - x[0][0] + 1) * (x[1][1] - x[0][1] + 1))
    print('...successfully found indices...')
    end_time = time.time()

    print(f'...algorithm execution time {round(end_time - start_time, 2)} sec.')
