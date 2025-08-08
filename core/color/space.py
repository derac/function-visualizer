from core.nd import xp as np


def rgb_to_hsv(r, g, b):
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc + 1e-8
    s = deltac / (maxc + 1e-8)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac
    h = (rc - gc) * (r == maxc) + (2.0 + bc - rc) * (g == maxc) + (4.0 + gc - bc) * (b == maxc)
    h = (h / 6.0) % 1.0
    return h, s, v


def hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    r = (i_mod == 0) * v + (i_mod == 1) * q + (i_mod == 2) * p + (i_mod == 3) * p + (i_mod == 4) * t + (i_mod == 5) * v
    g = (i_mod == 0) * t + (i_mod == 1) * v + (i_mod == 2) * v + (i_mod == 3) * q + (i_mod == 4) * p + (i_mod == 5) * p
    b = (i_mod == 0) * p + (i_mod == 1) * p + (i_mod == 2) * t + (i_mod == 3) * v + (i_mod == 4) * v + (i_mod == 5) * q
    return r, g, b


def apply_vibrance(rgb_r, rgb_g, rgb_b, vibrance=1.0, saturation=1.0):
    r = np.clip(rgb_r, 0.0, 1.0)
    g = np.clip(rgb_g, 0.0, 1.0)
    b = np.clip(rgb_b, 0.0, 1.0)
    h, s, v = rgb_to_hsv(r, g, b)
    s = np.clip(s * vibrance * saturation, 0.0, 1.0)
    return hsv_to_rgb(h, s, v)


def enforce_min_variance(rgb_r, rgb_g, rgb_b, min_lum_range=0.35, min_sat_mean=0.35):
    r = np.clip(rgb_r, 0.0, 1.0)
    g = np.clip(rgb_g, 0.0, 1.0)
    b = np.clip(rgb_b, 0.0, 1.0)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    lum_range = np.max(luminance) - np.min(luminance)
    h, s, v = rgb_to_hsv(r, g, b)
    sat_mean = np.mean(s)
    if (lum_range < min_lum_range) or (sat_mean < min_sat_mean):
        v = np.clip((v - v.min()) / (v.max() - v.min() + 1e-6), 0.0, 1.0)
        s = np.clip(s * 1.3 + 0.1, 0.0, 1.0)
        r, g, b = hsv_to_rgb(h, s, v)
    return r, g, b


