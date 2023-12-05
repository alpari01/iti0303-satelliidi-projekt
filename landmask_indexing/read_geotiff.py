from osgeo import gdal
import os
import shutil
import subprocess


def get_landmask_and_error(filein, shapefile):
    """Generates landmask and error binary map by rasterizing the shapefile on the GeoTIFF file."""
    path, filename = os.path.split(filein)
    product_name = filename[:-4]

    gdal.AllRegister()

    tif_copy = 'landmask_indexing/temp/' + product_name + '_temp.tif'

    shutil.copy2(filein, tif_copy)

    cmd = 'gdal_rasterize -burn 0 ' + shapefile + ' ' + tif_copy
    subprocess.call(cmd, shell=True)

    raster = gdal.Open(tif_copy)
    if raster is None:
        raise Exception(f"Could not open the raster file: {filein}")

    raster_band = raster.GetRasterBand(1).ReadAsArray()
    error_and_landmask = (raster_band != 0).astype(int)

    os.remove(tif_copy)

    return error_and_landmask


def get_raster_band(filein):
    """Reads a GeoTIFF file and returns the raster band as a NumPy array."""
    raster = gdal.Open(filein)
    raster_band = raster.GetRasterBand(1).ReadAsArray()
    return raster_band
