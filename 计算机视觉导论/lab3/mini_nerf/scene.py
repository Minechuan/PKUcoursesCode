import numpy as np


def get_camera_config():
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return {
        "H": 64,
        "W": 64,
        "focal": 70.0,
        "near": 1.5,
        "far": 5.0,
        "num_samples": 64,
        "c2w": c2w,
    }


def query_radiance_field(points):
    centers = np.array(
        [
            [-0.55, -0.15, 2.6],
            [0.45, 0.15, 3.4],
            [0.00, 0.55, 3.0],
        ],
        dtype=np.float32,
    )
    scales = np.array([0.28, 0.35, 0.22], dtype=np.float32)
    colors = np.array(
        [
            [0.95, 0.35, 0.25],
            [0.20, 0.60, 0.95],
            [0.95, 0.85, 0.20],
        ],
        dtype=np.float32,
    )

    diff = points[..., None, :] - centers[None, None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    density_components = np.exp(-sq_dist / (2.0 * scales[None, None, :] ** 2))
    sigmas = 18.0 * np.sum(density_components, axis=-1)

    weights = density_components / (np.sum(density_components, axis=-1, keepdims=True) + 1e-8)
    rgbs = np.sum(weights[..., None] * colors[None, None, :, :], axis=-2)

    floor_mask = points[..., 1] < -0.65
    sigmas = sigmas + floor_mask.astype(np.float32) * 8.0 * np.exp(
        -((points[..., 1] + 0.75) ** 2) / 0.02
    )
    rgbs = np.where(floor_mask[..., None], np.array([0.75, 0.75, 0.78], dtype=np.float32), rgbs)
    return sigmas.astype(np.float32), np.clip(rgbs, 0.0, 1.0).astype(np.float32)


def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    return -10.0 * np.log10(mse + 1e-8)

