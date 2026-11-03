"""Read calibrated IDR DTU cameras and masked RGB using the official split."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov


def readDTUCameras(path, white_background):
    path = Path(path)
    scene = json.loads((path / 'sfm_scene.json').read_text())
    bbox = np.asarray(scene['bbox']['transform'], dtype=np.float64).reshape(4, 4)
    if not np.allclose(bbox, np.diag([2.0, 2.0, 2.0, 1.0])):
        raise ValueError('DTU reader expects the supplied IDR-normalized bbox diag(2,2,2,1)')
    validation = (path / 'val_names.txt').read_text().split()
    if not validation or len(validation) != len(set(validation)):
        raise ValueError('Missing or duplicate official DTU test names')
    held_out = set(validation)
    image_paths = scene['image_path']['file_paths']
    cameras = scene['camera_track_map']['images']
    train, test, seen = [], [], set()
    for key in sorted(cameras, key=int):
        entry = cameras[key]
        if entry['flg'] != 2:
            continue
        image_path = path / image_paths[key]
        name = image_path.name
        if name in seen:
            raise ValueError('Duplicate DTU image: ' + name)
        seen.add(name)
        with Image.open(image_path) as image:
            rgb = np.array(image.convert('RGB'))
        with Image.open(path / 'pmasks' / name) as mask:
            foreground = np.asarray(mask.convert('L')) > 127
        if foreground.shape != rgb.shape[:2] or not foreground.any() or foreground.all():
            raise ValueError('Invalid DTU foreground mask: ' + name)
        rgb[~foreground] = 255 if white_background else 0
        image = Image.fromarray(rgb, mode='RGB')
        width, height = image.size
        intrinsic = entry['camera']['intrinsic']
        fx, fy = map(float, intrinsic['focal'])
        cx, cy = map(float, intrinsic['ppt'])
        if min(fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
            raise ValueError('Invalid DTU intrinsics')
        w2c = np.asarray(entry['camera']['extrinsic'], dtype=np.float64).reshape(4, 4)
        if not np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-6):
            raise ValueError('Invalid DTU camera rotation')
        info = CameraInfo(uid=int(key), R=w2c[:3, :3].T.copy(), T=w2c[:3, 3].copy(),
                          FovY=focal2fov(fy, height), FovX=focal2fov(fx, width),
                          image=image, image_path=str(image_path), image_name=image_path.stem,
                          width=width, height=height, intrinsics=(fx, fy, cx, cy))
        (test if name in held_out else train).append(info)
    if not held_out <= seen or not train:
        raise ValueError('Official test names missing from calibrated DTU data')
    print('DTU split: {} training / {} official test views'.format(len(train), len(test)))
    return train, test
