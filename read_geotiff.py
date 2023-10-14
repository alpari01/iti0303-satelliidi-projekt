from osgeo import gdal
import os
import shutil
import subprocess


def get_landmask_and_error(filein, shapefile):
    path, filename = os.path.split(filein)
    product_name = filename[:-5]

    gdal.AllRegister()

    tif_copy = 'temp/' + product_name + '_temp.tiff'
    shutil.copy2(filein, tif_copy)

    cmd = 'gdal_rasterize -burn 0 ' + shapefile + ' ' + tif_copy
    subprocess.call(cmd, shell=True)

    raster = gdal.Open(tif_copy)
    raster_band = raster.GetRasterBand(1).ReadAsArray()
    error_and_landmask = (raster_band != 0).astype(int)

    return error_and_landmask


def get_raster_band(filein):
    raster = gdal.Open(filein)
    raster_band = raster.GetRasterBand(1).ReadAsArray()
    return raster_band
