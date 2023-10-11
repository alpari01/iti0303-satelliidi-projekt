import os
import time
import tif_utils


def process(filein, mode, filetype, path_to_results, get_train_data=False):
    path, filename = os.path.split(filein)
    product_name = filename[:-4]

    print("Started reading product info")
    start = time.time()
    if filetype == 'GEOTIFF':
        product = tif_utils.read_geotiff(filein, mode, product_name, landmask=True, iceraster=get_train_data)
    elif filetype == 'ENVI':
        product = tif_utils.read_envi_dimap(filein, mode, product_name, landmask=True, iceraster=get_train_data)
    else:
        exit('Filetype not specified or unsupported filetype!')
    print("Finished reading product info in {}s".format(str(round(time.time() - start, 2))))
    return product


if __name__ == '__main__':
    filein = r'C:\some\path\to\file.tif'
    path_to_results = r'../../../Downloads'
    product = process(filein, 'IW', 'GEOTIFF', path_to_results)
