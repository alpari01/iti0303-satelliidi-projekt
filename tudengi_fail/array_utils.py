"""Utils code that holds functions necessary for array manipulations."""
import numpy as np
from scipy.sparse import csr_matrix


def remap_array(img_array,
                current_min_value_in_array,
                current_max_value_in_array,
                new_min_value_in_array,
                new_max_value_in_array):
    # range check
    if current_min_value_in_array >= current_max_value_in_array:
        exit("Warning: Invalid input range")

    if new_min_value_in_array >= new_max_value_in_array:
        exit("Warning: Invalid output range")

    portion = (img_array - current_min_value_in_array) * (new_max_value_in_array - new_min_value_in_array) \
              / (current_max_value_in_array - current_min_value_in_array)

    result = portion + new_min_value_in_array

    return result


def get_remapped_array(band, new_min_value, new_max_value, selected_min=None, selected_max=None):
    # default_max = 0.
    # default_min = -35.
    if selected_min is None:
        current_max_value = 0.
        current_min_value = -35.
    else:
        current_max_value = selected_max
        current_min_value = selected_min

    # current_max_value_in_img_array = 0
    # current_min_value_in_img_array = -35

    remapped_image = remap_array(band,
                                 current_min_value,
                                 current_max_value,
                                 new_min_value,
                                 new_max_value)

    return remapped_image


def multiply_arrays(sigma_h, sigma_v):
    sigma_hv = []
    for i in range(len(sigma_h)):
        hv_row = []
        for j in range(len(sigma_h[0])):
            current_h_value = sigma_h[i][j]
            current_v_value = sigma_v[i][j]
            hv_row.append(abs(current_h_value * current_v_value * 100))
        sigma_hv.append(np.asarray(hv_row))
    return np.asarray(sigma_hv)


def filter_band(band):
    band[np.isneginf(band)] = 0
    band = np.nan_to_num(band)
    band[band < -35] = -35
    band[band > 0] = 0
    return band


def preserve_zeros(sigma0, sigma1):
    """
    Separate function for applying only error areas. Leaves land areas as is.
    """
    sigma1_with_zeros = np.where(sigma0 == 0, sigma0, sigma1)
    return sigma1_with_zeros


def apply_error_and_landmask(array_in, error_and_landmask):
    """
    Apply error and land mask to matrix.
    :param array_in: array from e.g. segmentation where land should be zerod out
    :param error_and_landmask: error and landmask from rasterfile.
    :return: array, where errors and landareas are marked as 0
    """

    # where error == 0 -> zero, otherwise prediction value
    array_with_land = np.where(error_and_landmask == 0, error_and_landmask, array_in)

    return array_with_land


def log_to_linear(array):
    """
    Convert array in logarithmic scale to linear. From product, all physical arrays are in logarithmic scale.
    Mostly necessary for incidence angle.
    """

    return np.power(10, array / 10)


def linear_to_log(array):
    """
    Convert array in linear scale to logarithmic. From esthub processing, arrays are in linear scale.
    """

    return 10 * np.log10(array)


def list_to_arrays_w_nan(listin):
    b = np.zeros([len(listin), len(max(listin, key=lambda x: len(x)))])
    b[:] = np.NAN
    for i, j in enumerate(listin):
        b[i][0:len(j)] = j
    return b


def compute_matrix(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    matrix = compute_matrix(data)
    res = []
    for row in matrix:
        shape = np.unravel_index(row.data, data.shape)
        if len(shape[0]) > 0 and len(shape[1]) > 0:
            res.append(shape)

    return res
