#!/usr/bin/env python3
"""Signed-normal binding, local-affine editing and rest-pose verification.

Create separate affine-normal bindings for completed PMGS checkpoints.
Trained checkpoints are read only.

  python scripts/post_bind_affine.py --scenes all --bind-only
  CUDA_VISIBLE_DEVICES=0 python scripts/post_bind_affine.py --scenes chair lego mic

All generated files live under --output-root, separately from training runs.
The prescribed unit-normal column fixes normal thickness/offset under mesh
scaling. Tangential stretch and shear are transported by the full local affine
map. Covariances go directly to the renderer, without a quaternion conversion.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

LOG = logging.getLogger('post_bind_affine')
GIB = 1024 ** 3
_DTU_CAMERAS = {}


def torch_load(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=str(path.parent),
                                     prefix='.' + path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        except BaseException:
            temporary.unlink()
            raise
    try:
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_binding(path, binding):
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix='.affine_binding.',
                                     suffix='.pt', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(binding, str(temporary))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_checkpoint(checkpoint):
    ply = PlyData.read(str(checkpoint / 'point_cloud.ply'))
    data = ply['vertex'].data
    def columns(names):
        return torch.from_numpy(np.column_stack([data[name] for name in names]).astype(np.float32))
    snapshot = dict(xyz=columns(('x', 'y', 'z')),
                    rotation=columns(('rot_0', 'rot_1', 'rot_2', 'rot_3')),
                    log_scales=columns(('scale_0', 'scale_1', 'scale_2')))
    params = torch_load(checkpoint / 'model_params.pt')
    snapshot['vertices'] = torch.as_tensor(params['vertices']).detach().cpu().float().contiguous()
    snapshot['faces'] = torch.as_tensor(params['faces']).detach().cpu().long().contiguous()
    for name, tensor in snapshot.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite checkpoint field: ' + name)
    if not len(snapshot['xyz']):
        raise ValueError('Checkpoint has no Gaussians')
    if snapshot['vertices'].ndim != 2 or snapshot['vertices'].shape[1] != 3:
        raise ValueError('Invalid mesh vertex dimensions')
    if snapshot['faces'].ndim != 2 or snapshot['faces'].shape[1] != 3:
        raise ValueError('Invalid mesh face dimensions')
    if int(snapshot['faces'].min()) < 0 or int(snapshot['faces'].max()) >= len(snapshot['vertices']):
        raise ValueError('Mesh face index out of range')
    return snapshot


def original_factor(rotation, log_scales):
    rotation = rotation.double()
    norm = rotation.norm(dim=1, keepdim=True)
    if bool((norm == 0).any()):
        raise ValueError('Zero rotation quaternion in original weights')
    w, x, y, z = (rotation / norm).unbind(1)
    matrix = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), dim=1).reshape(-1, 3, 3)
    return matrix * log_scales.double().exp()[:, None, :]


def verify_new_binding(binding, snapshot, affine, chunk_size):
    if not torch.equal(binding['faces'].cpu(), snapshot['faces']):
        raise ValueError('New binding face topology differs from the checkpoint')
    if not torch.equal(binding['reference_vertices'].float().cpu(), snapshot['vertices']):
        raise ValueError('New binding mesh differs from the checkpoint')
    if not torch.equal(binding['xyz_rest'].float().cpu(), snapshot['xyz']):
        raise ValueError('New binding centers differ from the checkpoint')
    xyz, factor = affine.deform_gaussians(binding, snapshot['vertices'], chunk_size=chunk_size)
    errors = dict(rest_xyz_max_abs_error=float((xyz-snapshot['xyz'].double()).abs().max()),
                  rest_factor_max_abs_error=0., rest_covariance_max_abs_error=0.,
                  rest_covariance_max_abs_value=0., barycentric_sum_max_abs_error=float(
                      (binding['barycentric'].sum(1)-1).abs().max()),
                  negative_barycentric_gaussians=int((binding['barycentric'] < 0).any(1).sum()),
                  normal_offset_min=float(binding['normal_offset'].min()),
                  normal_offset_max=float(binding['normal_offset'].max()))
    for start in range(0, len(xyz), chunk_size):
        sl = slice(start, start+chunk_size)
        truth = original_factor(snapshot['rotation'][sl], snapshot['log_scales'][sl])
        cov_truth = truth @ truth.transpose(1, 2)
        rebuilt = factor[sl] @ factor[sl].transpose(1, 2)
        errors['rest_factor_max_abs_error'] = max(errors['rest_factor_max_abs_error'],
            float((factor[sl]-truth).abs().max()))
        errors['rest_covariance_max_abs_error'] = max(errors['rest_covariance_max_abs_error'],
            float((rebuilt-cov_truth).abs().max()))
        errors['rest_covariance_max_abs_value'] = max(errors['rest_covariance_max_abs_value'],
            float(cov_truth.abs().max()))
    errors['rest_covariance_relative_max_error'] = errors['rest_covariance_max_abs_error'] / max(
        errors['rest_covariance_max_abs_value'], 1e-300)
    if errors['rest_xyz_max_abs_error'] > 1e-7 or errors['rest_covariance_relative_max_error'] > 1e-8:
        raise ValueError('Rest reconstruction failed: ' + json.dumps(errors))
    errors['status'] = 'passed'
    return errors, xyz, factor


def resolve_dataset(run, scene, dataset_root):
    def has_cameras(path):
        return any((path / name).is_file() for name in ('transforms_test.json', 'sfm_scene.json'))

    for name in ('qgain_full_status.json', 'run_status.json'):
        path = run / name
        if path.is_file():
            metadata = json.loads(path.read_text(encoding='utf-8'))
            for key in ('original_dataset', 'original_data', 'source_path'):
                if metadata.get(key):
                    candidate = Path(metadata[key]).expanduser()
                    if has_cameras(candidate):
                        return candidate
    candidate = dataset_root / scene
    if not has_cameras(candidate):
        raise FileNotFoundError('Cannot locate test camera metadata: ' + str(candidate))
    return candidate


def make_camera(dataset, view_index, resolution, framing_scale=1.):
    from scene.cameras import Camera
    if (dataset / 'sfm_scene.json').is_file():
        from scene.dtu_dataset import readDTUCameras
        from utils.camera_utils import loadCam
        key = str(dataset.resolve())
        if key not in _DTU_CAMERAS:
            _, _DTU_CAMERAS[key] = readDTUCameras(dataset, False)
        cameras = _DTU_CAMERAS[key]
        if not 0 <= view_index < len(cameras):
            raise ValueError('view-index outside DTU test camera range 0..%d' % (len(cameras)-1))
        info = cameras[view_index]
        if framing_scale != 1.:
            fx, fy, cx, cy = info.intrinsics
            width, height = info.width, info.height
            intrinsics = (fx / framing_scale, fy / framing_scale,
                          (cx + .5 - width / 2) / framing_scale + width / 2 - .5,
                          (cy + .5 - height / 2) / framing_scale + height / 2 - .5)
            info = info._replace(FovX=2*math.atan(framing_scale*math.tan(info.FovX/2)),
                                 FovY=2*math.atan(framing_scale*math.tan(info.FovY/2)),
                                 intrinsics=intrinsics)
        return loadCam(SimpleNamespace(resolution=resolution, data_device='cpu'), view_index, info, 1.)
    metadata = json.loads((dataset / 'transforms_test.json').read_text(encoding='utf-8'))
    frames = metadata['frames']
    if not 0 <= view_index < len(frames):
        raise ValueError('view-index outside test camera range 0..%d' % (len(frames)-1))
    frame = frames[view_index]
    c2w = np.asarray(frame['transform_matrix'], dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    fovx = float(metadata['camera_angle_x'])
    fovy = float(metadata.get('camera_angle_y', fovx))
    fovx = 2*math.atan(framing_scale*math.tan(fovx/2))
    fovy = 2*math.atan(framing_scale*math.tan(fovy/2))
    return Camera(colmap_id=view_index, R=w2c[:3, :3].T.astype(np.float32),
                  T=w2c[:3, 3].astype(np.float32), FoVx=fovx, FoVy=fovy,
                  image=torch.zeros(3, resolution, resolution), gt_alpha_mask=None,
                  image_name=Path(frame['file_path']).name, uid=view_index,
                  data_device='cpu')


def cuda_budget():
    if not torch.cuda.is_available():
        raise RuntimeError('Rendering requires CUDA; --bind-only performs CPU binding and validation')
    free, total = torch.cuda.mem_get_info()
    reserve = 2 * GIB
    budget = free-reserve
    if budget < GIB:
        raise RuntimeError('Insufficient free CUDA memory: %.2f GiB free; need at least 3 GiB '
                           'including a 2 GiB reserve for other workloads' % (free/GIB))
    torch.cuda.set_per_process_memory_fraction(min(float(budget)/total, 0.95))
    torch.cuda.reset_peak_memory_stats()
    return dict(device=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                free_before_gib=free/GIB, total_gib=total/GIB, reserved_for_other_work_gib=2.,
                per_process_budget_gib=budget/GIB)


def packed_covariance(factor, chunk_size):
    packed = torch.empty((len(factor), 6), dtype=torch.float32)
    for start in range(0, len(factor), chunk_size):
        sl = slice(start, start+chunk_size)
        cov = factor[sl].double() @ factor[sl].double().transpose(1, 2)
        packed[sl] = cov[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)].float()
    if not bool(torch.isfinite(packed).all()):
        raise ValueError('Nonfinite covariance during float32 renderer conversion')
    return packed


class CovarianceProxy:
    def __init__(self, model, xyz, packed):
        self.model = model
        self.xyz = xyz.to(device='cuda', dtype=torch.float32).contiguous()
        self.packed = packed.to(device='cuda', dtype=torch.float32).contiguous()

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def get_xyz(self):
        return self.xyz

    def get_covariance(self, scaling_modifier=1.):
        return self.packed if scaling_modifier == 1. else self.packed * scaling_modifier ** 2


def render_tensor(renderer, camera, model, pipe, background):
    rendered = renderer(camera, model, pipe, background)['render'].detach()
    if not bool(torch.isfinite(rendered).all()):
        raise ValueError('Renderer produced nonfinite pixels')
    return rendered.cpu()


def save_image(path, tensor):
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy()*255 + .5).astype(np.uint8)
    Image.fromarray(array).save(str(path))


def edited_meshes(snapshot, twist_axis):
    vertices = snapshot['vertices'].double()
    points = snapshot['xyz'].double().numpy()
    low, high = np.quantile(points, [.05, .95], axis=0)
    center_np = np.median(points, axis=0)
    center = torch.from_numpy(center_np)
    relative = vertices-center
    stretch = relative * torch.tensor([1.35, .85, 1.], dtype=torch.float64) + center
    shear = relative.clone()
    shear[:, 0] += .35*relative[:, 1]
    shear += center
    axis = 2 if twist_axis == 'z' else 1
    radial_axes = (0, 1) if axis == 2 else (0, 2)
    height = max(float(high[axis]-low[axis]), 1e-6)
    angle = math.radians(40.) * torch.tanh(relative[:, axis]/(.5*height))
    c, s = angle.cos(), angle.sin()
    first, second = radial_axes
    twist = relative.clone()
    twist[:, first] = c*relative[:, first]-s*relative[:, second]
    twist[:, second] = s*relative[:, first]+c*relative[:, second]
    twist += center
    metadata = dict(pivot=center_np.tolist(), robust_gaussian_bounds=[low.tolist(), high.tolist()],
                    stretch_diagonal=[1.35, .85, 1.], shear_x_from_y=.35,
                    twist_axis=twist_axis, twist_angle_limit_degrees=40.,
                    twist_height=height, normal_scale_factor=1.)
    return dict(stretch=stretch, shear=shear, twist=twist), metadata


def preview_sheet(output, resolution, scene):
    label_height = max(32, resolution//16)
    sheet = Image.new('RGB', (resolution*2, (resolution+label_height)*3), (242, 242, 242))
    draw = ImageDraw.Draw(sheet)
    font_size = max(14, resolution//32)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()
    for row, mode in enumerate(('stretch', 'shear', 'twist')):
        filenames = ('original.png', mode+'.png')
        labels = (scene+' / original', 'PMGS / '+mode)
        for col, (filename, label) in enumerate(zip(filenames, labels)):
            x, y = col*resolution, row*(resolution+label_height)
            path = output / filename
            if path.is_file():
                with Image.open(str(path)) as image:
                    sheet.paste(image.convert('RGB'), (x, y+label_height))
            else:
                draw.rectangle((x, y+label_height, x+resolution-1, y+label_height+resolution-1), fill=(0, 0, 0))
                label = 'Image unavailable'
            draw.text((x+12, y+max(4, (label_height-font_size)//2)), label, fill=(20, 20, 20), font=font)
    sheet.save(str(output / 'preview.png'))


def render_scene(snapshot, binding, checkpoint, dataset, output, args, affine,
                 identity_xyz, identity_factor):
    from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
    from renderer.gaussian_renderer import render
    memory = cuda_budget()
    model = None
    proxy = None
    try:
        with torch.no_grad():
            model = GaussianMeshModel(3)
            # Load the original trained Gaussian fields for the rest control.
            model._load_ply(str(checkpoint / 'point_cloud.ply'))
            camera = make_camera(dataset, args.view_index, args.resolution, args.framing_scale)
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=True, debug=False)
            background = torch.zeros(3, device='cuda')
            original = render_tensor(render, camera, model, pipe, background)
            save_image(output / 'original.png', original)
            packed = packed_covariance(identity_factor, args.chunk_size)
            proxy = CovarianceProxy(model, identity_xyz, packed)
            identity = render_tensor(render, camera, proxy, pipe, background)
            save_image(output / 'identity.png', identity)
            difference = (original-identity).abs()
            identity_report = dict(mean_pixel_error=float(difference.mean()),
                                   max_pixel_error=float(difference.max()),
                                   status='passed', mean_tolerance=1e-4, max_tolerance=.05)
            if identity_report['mean_pixel_error'] >= 1e-4 or identity_report['max_pixel_error'] >= .05:
                identity_report['status'] = 'failed'
                raise ValueError('Native original versus new identity render failed: ' + json.dumps(identity_report))
            del proxy, packed, identity, difference
            proxy = None
            edited, deformation_metadata = edited_meshes(snapshot, args.twist_axis)
            for mode, vertices in edited.items():
                LOG.info('%s: rendering %s', output.name, mode)
                xyz, factor = affine.deform_gaussians(binding, vertices, chunk_size=args.chunk_size)
                packed = packed_covariance(factor, args.chunk_size)
                proxy = CovarianceProxy(model, xyz, packed)
                new_image = render_tensor(render, camera, proxy, pipe, background)
                save_image(output / (mode+'.png'), new_image)
                # Save the exact edited geometry so this comparison is repeatable.
                torch.save(dict(vertices=vertices, faces=snapshot['faces']), str(output / (mode+'_mesh.pt')))
                del proxy, xyz, factor, packed
                proxy = None
                del new_image
                gc.collect()
                torch.cuda.empty_cache()
            preview_sheet(output, args.resolution, output.name)
            memory['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/GIB
            memory['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/GIB
            return dict(status='complete', native_original_identity=identity_report,
                        test_view_index=args.view_index, resolution=args.resolution,
                        framing_scale=args.framing_scale,
                        background='black', dataset=str(dataset), deformation=deformation_metadata,
                        full_test_psnr_ssim_lpips_recomputed=False, cuda_memory=memory,
                        covariance_rendering='direct packed J Sigma0 J^T; no affine-to-quaternion conversion')
    finally:
        del proxy, model
        gc.collect()
        torch.cuda.empty_cache()


def process_scene(scene, args, affine, paths):
    if args.model_path:
        checkpoint = paths.resolve_checkpoint(args.model_path, args.iteration)
        run = paths.run_directory(checkpoint)
    else:
        run = paths.resolve_scene_run(args.run_root, scene, args.iteration)
        checkpoint = paths.resolve_checkpoint(run, args.iteration)
    output = args.output_root / scene
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(scene=scene, status='running', checkpoint=str(checkpoint), output=str(output),
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha256(__file__),
                  method='signed_normal_local_affine_v1', normal_scale_factor=1.,
                  input_checkpoints_read_only=True)
    metrics_path = output / 'metrics.json'
    handler = logging.FileHandler(str(output / 'post_bind_affine.log'), encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOG.addHandler(handler)
    protected = {}
    try:
        for name in ('point_cloud.ply', 'model_params.pt'):
            path = checkpoint / name
            if not path.is_file():
                raise FileNotFoundError('Required original checkpoint file is missing: ' + str(path))
            protected[name] = sha256(path)
        report['original_sha256_before'] = protected.copy()
        LOG.info('%s: reading trained weights', scene)
        snapshot = read_checkpoint(checkpoint)
        report.update(gaussian_count=len(snapshot['xyz']), face_count=len(snapshot['faces']),
                      vertex_count=len(snapshot['vertices']))
        binding_path = output / 'binding_affine_normal_v1.pt'
        if args.reuse_new_binding and binding_path.exists():
            binding = torch_load(binding_path)
            report['binding_action'] = 'reused_new_format'
        else:
            LOG.info('%s: binding with signed barycentrics and signed normal offsets', scene)
            binding = affine.bind_gaussians(snapshot['xyz'], snapshot['vertices'], snapshot['faces'],
                snapshot['rotation'], snapshot['log_scales'],
                chunk_size=args.chunk_size, threads=args.threads)
            save_binding(binding_path, binding)
            del binding
            binding = torch_load(binding_path)
            report['binding_action'] = 'created_new_format'
        report['face_association'] = 'exact_nearest_face_search'
        report['binding_file'] = str(binding_path)
        report['binding_sha256'] = sha256(binding_path)
        report['save_reload_verified'] = True
        geometry, identity_xyz, identity_factor = verify_new_binding(binding, snapshot, affine, args.chunk_size)
        report['geometry_check'] = geometry
        write_json(metrics_path, report)
        if args.bind_only:
            report['render'] = dict(status='not_requested')
        else:
            dataset = args.source_path or resolve_dataset(run, scene, args.dataset_root)
            report['render'] = render_scene(snapshot, binding, checkpoint, dataset,
                output, args, affine, identity_xyz, identity_factor)
        report['status'] = 'complete'
        LOG.info('%s: COMPLETE (%d Gaussians)', scene, len(snapshot['xyz']))
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        LOG.exception('%s: FAILED', scene)
    finally:
        after = {name: sha256(checkpoint / name) for name in protected if (checkpoint / name).is_file()}
        report['original_sha256_after'] = after
        report['original_files_unchanged'] = after == protected and bool(protected)
        if protected and after != protected:
            report.update(status='failed', original_preservation_error='An original file changed during the experiment')
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(metrics_path, report)
        LOG.removeHandler(handler)
        handler.close()
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return report


def main():
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--scene', help='One scene label; arbitrary dataset names are supported')
    selection.add_argument('--scenes', nargs='+', help='Scene labels, or all to discover complete outputs')
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--run-root', type=Path,
                        help='Scene output root (default: <project>/output); latest complete checkpoint per scene')
    target.add_argument('--model-path', type=Path,
                        help='One run directory, checkpoint directory, or point_cloud.ply')
    parser.add_argument('--source-path', type=Path, help='Dataset directory override for --model-path rendering')
    parser.add_argument('--output-root', type=Path, default=repo / 'bindings')
    parser.add_argument('--dataset-root', type=Path, default=repo / 'data/nerf_synthetic')
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--view-index', type=int, default=100)
    parser.add_argument('--framing-scale', type=float, default=1.,
                        help='Expand the preview camera FOV by this positive tangent-space factor')
    parser.add_argument('--resolution', type=int, default=800)
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--twist-axis', choices=['z', 'y'], default='z')
    parser.add_argument('--bind-only', action='store_true', help='CPU binding, save/reload and geometry validation only')
    parser.add_argument('--reuse-binding', '--reuse-new-binding', dest='reuse_new_binding', action='store_true',
                        help="Validate and reuse this output directory's saved affine binding")
    args = parser.parse_args()
    if min(args.iteration, args.resolution, args.chunk_size, args.threads) <= 0 or args.view_index < 0:
        parser.error('iteration, resolution, chunk-size and threads must be positive; view-index must be nonnegative')
    if not math.isfinite(args.framing_scale) or args.framing_scale <= 0:
        parser.error('framing-scale must be finite and positive')
    if args.source_path and not args.model_path:
        parser.error('--source-path is only valid with --model-path')
    requested = [args.scene] if args.scene is not None else (args.scenes or ['all'])
    discover = args.scene is None and requested == ['all']
    if args.scene is None and 'all' in requested and not discover:
        parser.error('Use --scenes all or explicit scene labels')
    args.run_root = (args.run_root or repo / 'output').expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    if args.source_path:
        args.source_path = args.source_path.expanduser().resolve()
    paths = import_file('_pmgs_checkpoints', repo / 'pmgs/checkpoints.py')
    try:
        if args.model_path:
            args.model_path = args.model_path.expanduser().resolve()
            checkpoint = paths.resolve_checkpoint(args.model_path, args.iteration)
            selected = [paths.infer_scene(checkpoint)] if discover else list(dict.fromkeys(requested))
            if len(selected) != 1:
                parser.error('--model-path accepts exactly one scene label')
        else:
            selected = paths.discover_scenes(args.run_root, args.iteration) if discover else list(dict.fromkeys(requested))
        selected = [paths.scene_label(scene) for scene in selected]
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    # Keep generated bindings and previews outside source and input directories.
    protected_inputs = [args.run_root, args.dataset_root]
    if args.model_path:
        protected_inputs.append(args.model_path.parent if args.model_path.is_file() else args.model_path)
    if args.source_path:
        protected_inputs.append(args.source_path)
    for protected in protected_inputs:
        if (args.output_root == protected or protected in args.output_root.parents
                or args.output_root in protected.parents):
            parser.error('output-root must be separate from input directories and their parents/children')
    for reserved in ('scripts', 'games', 'scene', 'utils', 'pmgs', 'tools', 'configs'):
        protected = (repo / reserved).resolve()
        if args.output_root == protected or protected in args.output_root.parents:
            parser.error('output-root must not be inside source directories')
    module_path = repo / 'games/mesh_splatting/scene/affine_normal_binding.py'
    if not module_path.is_file():
        parser.error('Missing PMGS affine binding module')
    sys.path.insert(0, str(repo))
    affine = import_file('_pmgs_affine_normal_binding', module_path)
    torch.set_num_threads(args.threads)
    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    summary = dict(status='running', method='signed_normal_local_affine_v1',
                   output_root=str(args.output_root), run_root=str(args.run_root), scenes=[],
                   timestamp_utc=datetime.now(timezone.utc).isoformat(), bind_only=args.bind_only)
    with torch.no_grad():
        for scene in selected:
            try:
                report = process_scene(scene, args, affine, paths)
            except (FileNotFoundError, ValueError) as error:
                LOG.exception('%s: cannot open checkpoint', scene)
                report = dict(scene=scene, status='failed', error=repr(error))
            summary['scenes'].append(report)
            write_json(args.output_root / 'summary.json', summary)
    failures = sum(row['status'] != 'complete' for row in summary['scenes'])
    summary.update(status='failed' if failures else 'complete',
                   passed=len(summary['scenes'])-failures, failed=failures)
    write_json(args.output_root / 'summary.json', summary)
    LOG.info('Finished: %d complete, %d failed; %s', summary['passed'], failures, args.output_root / 'summary.json')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
