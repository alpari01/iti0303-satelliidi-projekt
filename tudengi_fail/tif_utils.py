"""Utils code that holds functions necessary for geotiff file reading and saving."""
from osgeo import gdal, osr, ogr
from osgeo.gdalconst import GA_ReadOnly
from skimage.transform import rescale
import logging
import os
import shutil
import numpy as np
import concurrent.futures
import array_utils
import subprocess
from scipy.ndimage import zoom
import re


class Product:
    """
    Class, that hold product matrixes and info about imaging mode
    """
    IW = 'IW'
    EW = 'EW'

    def __init__(self, mode, sigma_co_db, sigma_cro_db, inci, error_and_landmask, depth, ice_class, time, product_name, filein):
        self.mode = mode
        self.sigma_co_db = sigma_co_db
        self.sigma_cro_db = sigma_cro_db
        self.inci = inci
        self.error_and_landmask = error_and_landmask
        self.depth = depth
        self.ice_class = ice_class
        self.time = time
        self.product_name = product_name
        self.filein = filein


def get_depth(filein):
    depthfile = './landmask/ice_area_depth_lest97_v2.tif'

    path, filename = os.path.split(filein)
    product_name = filename[:-4]

    data = gdal.Open(filein, GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    data_arr = data.GetRasterBand(1).ReadAsArray()
    # data_arr = data_band
    og_height, og_width = data_arr.shape

    gdal.AllRegister()
    tif_copy = './temp/' + product_name + "_depthtemp.tif"
    shutil.copy2(filein, tif_copy)

    cmd = 'gdal_translate -projwin ' + ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) + ' -of GTiff ' + \
          depthfile + ' ' + tif_copy
    subprocess.call(cmd, shell=True)

    raster = gdal.Open(tif_copy)
    band = raster.GetRasterBand(1)
    depth = band.ReadAsArray()
    old_height, old_width = depth.shape
    depth_zoom = zoom(depth, [og_height / old_height, og_width / old_width])

    raster = None
    os.remove(tif_copy)

    return depth_zoom


def reproject_geotiff_arr(filein, project_to, outresXY):
    """
    :param array_in: absolute path to the file
    :param project_to: EPSG number to which the data should be projected
    :param outresXY: resolution of reprojected file in both X and Y axis
    """
    logging.info(' Reprojecting data to EPSG {} projection and reading data.'.format(project_to))
    print('Reprojecting data to EPSG {} projection and reading data.'.format(project_to))

    path, filename = os.path.split(filein)
    product_name = filename[:-4]
    # reprojecting into EPSG:3301
    ds1 = gdal.Warp('temp/' + product_name + '_reprojected.tif', filein, dstSRS=project_to, srcNodata=0, dstNodata=0,
                    xRes=outresXY, yRes=outresXY, multithread=True, copyMetadata=True)
    ds1 = None

    raster_reproject = gdal.Open(product_name + '_reprojected.tif')
    band1 = raster_reproject.GetRasterBand(1).ReadAsArray()
    band2 = raster_reproject.GetRasterBand(2).ReadAsArray()
    band3 = raster_reproject.GetRasterBand(3).ReadAsArray()
    raster_reproject = None

    return band1, band2, band3, product_name + '_reprojected.tif'




def get_error_and_landmask(filein):
    """
    Objective is to create matrix with dimensions and projection as filein and fill
    values of 0 and 1, where 0 is land and/or error.
    NB! Creates temporary copy of file in to be used to modify original file.

    :param filein: SAR file.
    :return: land and error mask
    """
    print('Finding land mask.')
    logging.info(' Finding land mask.')

    # ESRI shapefile containing land polygons.
    # NB! GSHSS file include larger lakes.
    shapefile = 'landmask/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp'
    # shapefile = 'landmask/Laanemeri/Laanemeri.shp'

    path, filename = os.path.split(filein)
    product_name = filename[:-4]

    gdal.AllRegister()
    tif_copy = 'temp/' + product_name + "_temp.tif"
    shutil.copy2(filein, tif_copy)

    # see käsk kirjutab failis band 1 andmed üle. pmst võiks nüüd faili uuesti sisse lugeda ja sealt nullid maskiks võtta.
    cmd = 'gdal_rasterize -burn 0 ' + shapefile + ' ' + tif_copy
    subprocess.call(cmd, shell=True)

    raster = gdal.Open(tif_copy)
    bandvh = raster.GetRasterBand(1)
    maskfail = bandvh.ReadAsArray()
    error_and_landmask = (maskfail != 0).astype(int)
    raster = None
    os.remove(tif_copy)

    return error_and_landmask


def read_geotiff(filein, mode, product_name, landmask=True, iceraster=False):
    """
    Function to read processed SAR geotiff files. Assumption is that the file is projected into Estonian 97 projection
    EPSG:3301 and includes 3 channels (cross-pol, co-pol, incidence angle) in decibel scale.
    :param filein: absoulte path of the file
    :param mode: either IW or EW
    :param product_name: tif file name extracted from absolute path
    :param landmask: if true (default), land mask is found using GSHHS landmask shapefile for the specific file
    :param iceraster: if true (not default), program searches for ice class raster (annotated SAR file with ice class values).
    :return: product which includes mode (str), sigma_co_db (np.array), sigma_cro_db (np.array), incidence angle (np.array),
    error and landmask (np.array), iceclass raster (np.array), time (str with the coding 'YYYYMMDDThhmmss').
    """

    logging.info(' Reading product using GDAL.')
    print('Reading files using GDAL.')

    if mode == 'IW':
        mode = Product.IW
        outresXY = 20
        time = product_name.split("_")[4]
        raster = gdal.Open(filein)
        #if not re.search(raster.GetProjection(), "3301"):
        #    sigma_cro, sigma_co, inci, processing_file = reproject_geotiff_arr(filein, "EPSG:3301", outresXY)
        #else:
        sigma_cro = raster.GetRasterBand(1).ReadAsArray()
        sigma_co = raster.GetRasterBand(2).ReadAsArray()
        inci = raster.GetRasterBand(3).ReadAsArray()
        processing_file = filein
    elif mode == 'EW':
        mode = Product.EW
        outresXY = 40
        time = product_name.split("_")[4]
        raster = gdal.Open(filein)
        #if not re.search(raster.GetProjection(), "3301"):
        #    sigma_co, sigma_cro, inci, processing_file = reproject_geotiff_arr(filein, "EPSG:3301", outresXY)
        #else:
        sigma_cro = raster.GetRasterBand(2).ReadAsArray()
        sigma_co = raster.GetRasterBand(1).ReadAsArray()
        inci = raster.GetRasterBand(3).ReadAsArray()
        processing_file = filein

    # sigma_co = band_co.ReadAsArray()
    # sigma_cro = band_cro.ReadAsArray()
    # inci = bandinci.ReadAsArray()

    # check if log or linear
    if np.nanmean(sigma_co) < 0:
        sigma_co_db = sigma_co
        sigma_cro_db = sigma_cro
        inci = array_utils.log_to_linear(inci)
    else:
        sigma_co_db = array_utils.linear_to_log(sigma_co)
        sigma_cro_db = array_utils.linear_to_log(sigma_cro)

    # filter bands
    inci[inci < 16.36] = 0
    inci[inci > 47] = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [sigma_co_db, sigma_cro_db]
        sigma_co_db, sigma_cro_db = executor.map(array_utils.filter_band, args)

    # find landmask
    if landmask:
        error_and_landmask = get_error_and_landmask(processing_file)
    else:
        error_and_landmask = np.ones((sigma_co.shape[0], sigma_co.shape[1]), dtype=np.float32)

    # find depth
    depth = get_depth(filein)
    depth[error_and_landmask == 0] = 0

    # load ice class raster for training data collection
    if iceraster:
        # read iceclass file as well
        logging.warning(" Reading iceclass raster. Make sure the file exists!")
        iceclass_filename = product_name[:-10] + "_annsar_resampled.tif"
        iceclass_file_path = r'G:\Project_data\2019-KEMIT-ice\icetype_kaur_80\ew\icetif' + '\\' + iceclass_filename
        try:
            iceclass_raster = gdal.Open(iceclass_file_path)
            bandice = iceclass_raster.GetRasterBand(1)
            ice_class = bandice.ReadAsArray()
        except:
            logging.warning(" Iceraster not found!")
            exit('Iceraster not found!')
    else:
        ice_class = 0




    return Product(mode, sigma_co_db, sigma_cro_db, inci, error_and_landmask, depth, ice_class, time, product_name, processing_file)


def read_envi_dimap(pathin, mode, product_name, landmask=True, iceraster=False):
    """
    Function to read processed ENVI .hdr + .img files. Assumption is that the file is projected into Estonian 97 projection
    EPSG:3301 and includes 3 channels (cross-pol, co-pol, incidence angle) in decibel scale - should check the assumption
    as well as if image is in linear on log scale.
    :param pathin: absolute path of the folder that contains <filename>.dim file and <filename>.data folder
    :param mode: either IW or EW
    :param product_name: tif file name extracted from absolute path
    :param landmask: if true (default), land mask is found using GSHHS landmask shapefile for the specific file
    :param iceraster: if true (not default), program searches for ice class raster (annotated SAR file with ice class values).
    :return: product which includes mode (str), sigma_co_db (np.array), sigma_cro_db (np.array), incidence angle (np.array),
    error and landmask (np.array), iceclass raster (np.array), time (str with the coding 'YYYYMMDDThhmmss').
    """

    logging.info('Reading product using GDAL.')
    print('Reading files using GDAL.')

    if mode == 'IW':
        mode = Product.IW
        outresXY = 20
    elif mode == 'EW':
        mode = Product.EW
        outresXY = 40

    # find *.img files in <filepath>.data directory
    for files_in_folder in os.listdir(pathin):
        # file should have dimensions (2,)
        if files_in_folder.endswith('.data'):
            time = files_in_folder.split("_")[4]
            og_filename = files_in_folder.replace('calibrated', '').replace('thermalnoiseremoved', '').replace('speckle-filtered', '').replace('terraincorrected', '').replace('-', '').replace('.data', '').replace('.zip', '')
            datafolder = os.path.join(pathin, files_in_folder + '/')

    for file in os.listdir(datafolder):
        if file.endswith('.img'):
            filename_clean = file.replace('.img', '')
            try:
                polarization = filename_clean.split("_")[1]
                if polarization == 'VH' or polarization == 'HV':
                    cropol_file = os.path.join(datafolder, file)
                elif polarization == 'VV' or polarization == 'HH':
                    copol_file = os.path.join(datafolder, file)
                else:
                    logging.warning("Unsupported polarization in BEAM-DIMAP file: ", datafolder + '/' + file)
                    exit('Unsupported polarization in BEAM-DIMAP file!')
            except IndexError:
                inci_file = os.path.join(datafolder, file)

    # reprojecting into EPSG:3301
    ds1 = gdal.Warp('temp/' + og_filename + '_co.tif', copol_file, dstSRS='EPSG:3301', srcNodata=0, dstNodata=0, xRes=outresXY, yRes=outresXY, multithread=True, copyMetadata=True)
    ds2 = gdal.Warp('temp/' + og_filename + '_cro.tif', cropol_file, dstSRS='EPSG:3301', srcNodata=0, dstNodata=0, xRes=outresXY, yRes=outresXY, multithread=True, copyMetadata=True)
    ds3 = gdal.Warp('temp/' + og_filename + '_inci.tif', inci_file, dstSRS='EPSG:3301', srcNodata=0, dstNodata=0, xRes=outresXY, yRes=outresXY, multithread=True, copyMetadata=True)
    ds1 = ds2 = ds3 = None

    raster_co = gdal.Open('temp/' + og_filename + '_co.tif')
    raster_cro = gdal.Open('temp/' + og_filename + '_cro.tif')
    raster_inci = gdal.Open('temp/' + og_filename + '_inci.tif')

    sigma_co = raster_co.GetRasterBand(1).ReadAsArray()
    sigma_cro = raster_cro.GetRasterBand(1).ReadAsArray()
    inci = raster_inci.GetRasterBand(1).ReadAsArray()
    raster_co = raster_cro = raster_inci = None

    # assumption that in linear scale the mean is greater than zero (should be)
    # in case of beam-dimap files from esthub, they are in linear scale
    if np.nanmean(sigma_co) > 0:
        sigma_co_db = array_utils.linear_to_log(sigma_co)
        sigma_cro_db = array_utils.linear_to_log(sigma_cro)
    else:
        sigma_co_db = sigma_co
        sigma_cro_db = sigma_cro

    # filter bands
    inci[inci < 16.36] = 0
    inci[inci > 47] = 0
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            args = [sigma_co_db, sigma_cro_db]
            sigma_co_db, sigma_cro_db = executor.map(array_utils.filter_band, args)
    except MemoryError:
        sigma_co_db = array_utils.filter_band(sigma_co_db)
        sigma_cro_db = array_utils.filter_band(sigma_cro_db)

    # find landmask
    if landmask:
        error_and_landmask = get_error_and_landmask('temp/' + og_filename + '_co.tif')
    else:
        error_and_landmask = np.ones((sigma_co.shape[0], sigma_co.shape[1]), dtype=np.float32)

    # load ice class raster for training data collection
    if iceraster:
        # read iceclass file as well
        logging.warning("Reading iceclass raster. Make sure the file exists!")
        iceclass_filename = product_name[:-9] + "annsar.tif"
        iceclass_file_path = "shptif/" + iceclass_filename
        try:
            iceclass_raster = gdal.Open(iceclass_file_path)
            bandice = iceclass_raster.GetRasterBand(1)
            ice_class = bandice.ReadAsArray()
        except:
            logging.warning("Iceraster not found!")
            exit('Iceraster not found!')
    else:
        ice_class = 0

    return Product(mode, sigma_co_db, sigma_cro_db, inci, error_and_landmask, ice_class, time, 'temp/' + og_filename + '_co.tif')


def array2raster_gdal(original_input, array_to_raster, resize_factor, product_name, path_to_results):
    """
    original from: https://gist.github.com/jkatagi/a1207eee32463efd06fb57676dcf86c8
    :param original_input: file from which the predictions were made; necessary here to get the SRS information (product class)
    :param array_to_raster: prediction matrix (numpy array)
    :param resize_factor: factor for resizing output array to save space. e.g. resize_factor = 3 means array_to_raster.shape / 3
    :param product_name: name for the file (str)
    :param path_to_results: path for file saving
    save GTiff file from numpy.array
    input:
        dataset : original tif file
        newRasterfn: save file name
        array : numpy.array
    :return: None
    """
    logging.info(" Writing product of {} to raster.".format(product_name))
    path, filename = os.path.split(original_input.filein)
    input_name = filename[:-4]

    nodataval = 0.0
    ds = gdal.Open(original_input.filein)
    prj = ds.GetProjection()
    driver = gdal.GetDriverByName('GTiff')
    band_num = 1

    if resize_factor > 1:
        originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
        array_to_raster = rescale(array_to_raster.astype(np.float), 1/resize_factor, anti_aliasing=False)
        array_to_raster[array_to_raster < 0.01] = 0
        cols = array_to_raster.shape[1]
        rows = array_to_raster.shape[0]
        array_to_raster.resize((rows, cols), refcheck=False)
        outRaster = driver.Create(path_to_results + '/' + input_name + '_' + product_name + '.tif', cols, rows, band_num,
                                  gdal.GDT_Float32, options=['COMPRESS=LZW'])
        outRaster.SetGeoTransform((originX, pixelWidth * resize_factor, 0, originY, 0, pixelHeight * resize_factor))
    else:
        originX, pixelWidth, b, originY, d, pixelHeight = ds.GetGeoTransform()
        cols = array_to_raster.shape[1]
        rows = array_to_raster.shape[0]
        outRaster = driver.Create(path_to_results + '/' + input_name + '_' + product_name + '.tif', cols, rows, band_num,
                                  gdal.GDT_Float32, options=['COMPRESS=LZW'])
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array_to_raster)
            outband.SetNoDataValue(nodataval)
        else:
            outband.WriteArray(array_to_raster[:, :, b])
            outband.SetNoDataValue(nodataval)

    # setteing srs from input tif file.
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

    return None


def gdal_vectorize_predictions_with_pred_vals(original_input, array_to_vector, product_name, path_to_results):
    """
    Function to save prediction matrix into ESRI shapefile with polygons.
    - Aim is to give prediction values to the polygons in shape file.

    :param original_input: file from which the predictions were made; necessary here to get the SRS information (product class)
    :param array_to_vector: prediction matrix (numpy array)
    :param product_name: name for the file (str)
    :param path_to_results: path for file saving
    :return: None
    """
    logging.info(" Writing product of {} to ESRI shapefile.".format(product_name))
    original_input = original_input.filein

    path, filename = os.path.split(original_input)
    input_name = filename[:-4]
    tif_copy = input_name + "_temp.tif"

    shutil.copy2(original_input, tif_copy)
    og_raster = gdal.Open(original_input)
    proj = og_raster.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    nodataval = 0.0

    input_data = gdal.Open(tif_copy, gdal.GA_Update)
    band = input_data.GetRasterBand(1)
    band.WriteArray(np.array(array_to_vector), 0, 0)
    band.SetNoDataValue(nodataval)
    band.FlushCache()

    outShapefile = path_to_results + "/" + input_name + '_' + product_name

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outShapefile + ".shp"):
        driver.DeleteDataSource(outShapefile + ".shp")
    outDatasource = driver.CreateDataSource(outShapefile + ".shp")
    outLayer = outDatasource.CreateLayer(outShapefile, srs=srs)
    newField = ogr.FieldDefn('type', ogr.OFTInteger)
    outLayer.CreateField(newField)
    gdal.Polygonize(band, band, outLayer, 0, [], callback=None)
    outDatasource.Destroy()
    input_data = None
    os.remove(tif_copy)

    return None
