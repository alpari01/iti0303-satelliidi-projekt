import csv
import time


def read_csv(filename):
    result_list = []
    with open(filename, 'r', newline='') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            int_row = [int(cell) for cell in row]
            result_list.append(int_row)
    return result_list


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

            amount = 0
            for line in sub_list:
                amount += line.count(1)

            if amount > (max_square * max_square * filling):
                top_left = (row_start + prev_row_start, col_start + prev_col_start)
                bottom_right = (row_start + prev_row_start + max_square - 1,
                                col_start + prev_col_start + max_square - 1)
                indices.append([top_left, bottom_right])
            else:
                get_indices(sub_list, int(max_square / 2), min_square, filling,
                            row_start + prev_row_start, col_start + prev_col_start)


if __name__ == '__main__':
    landmask_filename = 'sigma_cro/error_and_landmask.csv'

    print('Program started...')

    landmask = read_csv(landmask_filename)
    print('...successfully read landmask csv...')

    start_time = time.time()
    indices = []
    get_indices(landmask, 512, 32)
    print('...successfully found indices...')
    end_time = time.time()

    print(f'...algorithm execution time {round(end_time - start_time, 2)} sec.')
