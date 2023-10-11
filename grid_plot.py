import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def read_csv(filename):
    result_list = []
    with open(filename, 'r', newline='') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            int_row = [int(cell) for cell in row]
            result_list.append(int_row)
    return result_list


def draw_grid(data, max_square, min_square, filling=0.999, prev_row_start=0, prev_col_start=0):
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

            amount = 0
            for line in sub_list:
                amount += line.count(1)

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
    from_where_array = np.array(from_where, dtype=float)
    to_where_array = np.array(to_where, dtype=float)

    indices = np.where(from_where_array == 0.5)
    to_where_array[indices] = 3000

    return to_where_array


if __name__ == '__main__':
    landmask_filename = 'sigma_cro/error_and_landmask.csv'
    sigma_cro_filename = 'sigma_cro/sigma_cro.csv'

    print('Program started...')

    landmask = read_csv(landmask_filename)
    print('...successfully read landmask csv...')

    sigma_cro = read_csv(sigma_cro_filename)
    print('...successfully read sigma_cro csv...')

    square_indices = []
    draw_grid(landmask, 512, 32)
    print('...successfully drawn grid...')

    result_data = transfer_indices(landmask, sigma_cro)
    print('...successfully transferred indices...')

    sns.heatmap(result_data, cmap='grey', vmin=0, vmax=3000)
    plt.savefig('plot.png', dpi=3000)
    plt.show()
    print('...successfully created plot.')
