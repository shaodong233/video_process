'''
https://github.com/guptapraful/niqe
'''
import numpy as np
import scipy.io
from os.path import dirname, join
import scipy.ndimage
import scipy.special
import math
from PIL import Image

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = np.sqrt(np.average(left_data)) if len(left_data) > 0 else 0
    right_mean_sqrt = np.sqrt(np.average(right_data)) if len(right_data) > 0 else 0

    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf
    imdata2_mean = np.mean(imdata2)
    r_hat = (np.average(np.abs(imdata)) ** 2) / np.average(imdata2) if imdata2_mean != 0 else np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    N = (br - bl) * (gam2 / gam1)
    return alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return H_img, V_img, D1_img, D2_img

def gen_gauss_window(lw, sigma):
    lw = int(lw)
    weights = [np.exp(-0.5 * (ii ** 2) / (sigma ** 2)) for ii in range(-lw, lw + 1)]
    weights /= np.sum(weights)
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = image.astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image

def _niqe_extract_subband_feats(mscncoefs):
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, bl3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, bl4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                     alpha1, N1, bl1, br1,
                     alpha2, N2, bl2, br2,
                     alpha3, N3, bl3, bl3,
                     alpha4, N4, bl4, bl4])

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = [img[j:j + patch_size, i:i + patch_size] for j in range(0, h - patch_size + 1, patch_size) for i in range(0, w - patch_size + 1, patch_size)]
    patch_features = [_niqe_extract_subband_feats(p) for p in patches]
    return np.array(patch_features)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = img.shape
    hoffset = h % patch_size
    woffset = w % patch_size

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    img2 = np.array(Image.fromarray(img).resize((w // 2, h // 2), Image.BICUBIC))

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn2, _, _ = compute_image_mscn_transform(img2)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size // 2)

    return np.hstack((feats_lvl1, feats_lvl2))

def niqe(inputImgData):
    patch_size = 96
    module_path = dirname(__file__)
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    print("Loaded MAT file content keys:", params.keys())  # Add this line for debugging
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N = inputImgData.shape

    assert M > (patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov) / 2.0
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score