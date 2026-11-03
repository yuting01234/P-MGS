"""Camera, checkpoint resolution and output helpers for the Lego example."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def protocol():
    return json.loads((ROOT / 'protocol.json').read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checkpoint(path, iteration=30000):
    path = Path(path).expanduser().resolve()
    if path.name == 'point_cloud.ply':
        path = path.parent
    if (path / 'point_cloud.ply').is_file():
        return path
    candidate = path / 'point_cloud' / ('iteration_' + str(iteration))
    if (candidate / 'point_cloud.ply').is_file():
        return candidate
    raise FileNotFoundError('Expected point_cloud.ply in ' + str(path) + ' or ' + str(candidate))


def binding_path(weights, explicit=None):
    if explicit:
        target = Path(explicit).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        return target
    name = 'binding_affine_normal_v1.pt'
    candidates = [PROJECT / 'bindings' / 'lego' / name, weights / name]
    for target in candidates:
        if target.is_file():
            return target.resolve()
    raise FileNotFoundError('No affine-normal binding found; pass --binding-path. Searched: ' +
                            ', '.join(str(p) for p in candidates))


def camera(config=None):
    import numpy as np
    import torch
    from scene.cameras import Camera
    p = config or protocol()
    c2w = np.array(p['camera_c2w'], dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    size = p['resolution']
    return Camera(colmap_id=0, uid=0, R=w2c[:3, :3].T.astype(np.float32),
                  T=w2c[:3, 3].astype(np.float32), FoVx=p['camera_angle_x'],
                  FoVy=p['camera_angle_x'], image=torch.zeros((3, size, size)),
                  image_name='lego_bucket_fixed_side', gt_alpha_mask=None,
                  trans=np.zeros(3), scale=1., data_device='cpu')


def finite(name, value):
    import torch
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError('Non-finite ' + name)


def save_rgb(path, value):
    import numpy as np
    from PIL import Image
    finite('RGB', value)
    pixels = (value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + .5).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def image_error(a, b):
    delta = (a.clamp(0, 1) - b.clamp(0, 1)).abs()
    return dict(mean_absolute_rgb_error=float(delta.mean()), maximum_absolute_rgb_error=float(delta.max()))
