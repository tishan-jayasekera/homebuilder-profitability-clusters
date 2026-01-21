import numpy as np
from shapely.ops import transform


def compute_affine(src_pts, dst_pts):
    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)
    if len(src) < 3:
        return None
    A = []
    B = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(X)
        B.append(Y)
    A = np.array(A)
    B = np.array(B)
    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return params


def apply_affine(polygon, params):
    if params is None:
        return None
    a, b, c, d, e, f = params

    def _transform(x, y, z=None):
        X = a * x + b * y + c
        Y = d * x + e * y + f
        return X, Y

    return transform(_transform, polygon)


def alignment_rms(src_pts, dst_pts, params):
    if params is None or len(src_pts) < 4:
        return None
    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)
    a, b, c, d, e, f = params
    X = a * src[:, 0] + b * src[:, 1] + c
    Y = d * src[:, 0] + e * src[:, 1] + f
    err = np.sqrt((X - dst[:, 0]) ** 2 + (Y - dst[:, 1]) ** 2)
    return float(np.mean(err))
