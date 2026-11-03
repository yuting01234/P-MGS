"""Render an edit animation with the bundled Blender rig trajectory."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from common import PROJECT, ROOT, binding_path, camera, checkpoint, digest, image_error, protocol, save_rgb


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--model-path', type=Path, required=True, help='Compatible training run, iteration directory, or PLY')
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--binding-path', type=Path,
                   help='Affine-normal binding; default: bindings/lego/binding_affine_normal_v1.pt, then beside the PLY')
    p.add_argument('--output-root', type=Path, default=PROJECT / 'animations' / 'lego_bucket_lift')
    p.add_argument('--gpu', help='CUDA device selection; otherwise retain CUDA_VISIBLE_DEVICES')
    p.add_argument('--preview', action='store_true', help='Render original keyframes 0, 15, 29 and 59')
    p.add_argument('--memory-fraction', type=float, default=.70)
    return p


def main():
    args = parser().parse_args()
    if not 0 < args.memory_fraction <= 1:
        raise ValueError('--memory-fraction must be in (0,1]')
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.path.insert(0, str(PROJECT))
    import torch
    from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
    from renderer.gaussian_renderer import render
    from utils.general_utils import strip_symmetric
    from motion import RigField
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to render the animation')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    weights = checkpoint(args.model_path, args.iteration)
    bound = binding_path(weights, args.binding_path)
    inputs = [weights / 'point_cloud.ply', weights / 'model_params.pt', bound]
    before = {str(path): digest(path) for path in inputs}
    output = args.output_root.expanduser().resolve()
    if output == weights or weights in output.parents:
        raise ValueError('Choose an animation output directory outside the checkpoint directory')
    frames = output / ('preview' if args.preview else 'frames')
    frames.mkdir(parents=True, exist_ok=True)
    config = protocol()
    started = time.monotonic()
    with torch.no_grad():
        model = GaussianMeshModel(3)
        model.load_ply(str(weights / 'point_cloud.ply'))
        view = camera(config)
        background = torch.tensor(config['background'], dtype=torch.float32, device='cuda')
        direct_pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        baseline = render(view, model, direct_pipe, background)['render']
        field = RigField()
        from games.mesh_splatting.scene.affine_normal_binding import FORMAT, deform_gaussians
        try:
            binding = torch.load(str(bound), map_location='cuda', weights_only=False)
        except TypeError:
            binding = torch.load(str(bound), map_location='cuda')
        if binding.get('format_version') != FORMAT:
            raise ValueError('The selected file is not an affine-normal binding; check --binding-path')
        if binding['xyz_rest'].shape != model.get_xyz.shape:
            raise ValueError('Affine binding and checkpoint have different Gaussian counts')
        if not torch.allclose(binding['xyz_rest'], model.get_xyz.double(), rtol=0, atol=1e-7):
            raise ValueError('Affine binding rest positions do not match the selected checkpoint')
        if (not torch.equal(binding['faces'], model.faces.long())
                or not torch.equal(binding['reference_vertices'].float(), model.vertices)):
            raise ValueError('Affine binding reference mesh does not match the selected checkpoint')
        original, faces, labels, partition = field.prepare_bound(binding['reference_vertices'], binding['faces'])
        binding = dict(binding, reference_vertices=original, faces=faces)

        def deform(vertices):
            xyz, factor = deform_gaussians(binding, vertices)
            return xyz, factor @ factor.transpose(-2, -1)

        class Proxy:
            def __getattr__(self, name):
                return getattr(model, name)

            @property
            def get_xyz(self):
                return self.xyz

            def get_covariance(self, scaling_modifier=1.):
                return strip_symmetric(self.covariance * scaling_modifier ** 2)

        proxy = Proxy()
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=True, debug=False)

        def apply(index):
            vertices = field.apply_bound(original, labels, index)
            xyz, covariance = deform(vertices)
            proxy.xyz, proxy.covariance = xyz.float().contiguous(), covariance.float().contiguous()

        def image():
            return render(view, proxy, pipe, background)['render']

        save_rgb(output / 'rest.png', baseline)
        apply(None)
        identity = image()
        identity_error = image_error(baseline, identity)
        if identity_error['mean_absolute_rgb_error'] > 1e-4:
            raise RuntimeError('Reference-pose render validation failed: ' + repr(identity_error))
        save_rgb(output / 'control_rest.png', identity)
        del identity, baseline
        indices = [0, 15, 29, 59] if args.preview else list(range(config['frame_count']))
        for index in indices:
            apply(index)
            save_rgb(frames / ('frame_%04d.png' % index), image())
            if index % 10 == 0 or args.preview:
                print('PMGS_FRAME', index, 'ARM_Z', config['controller_z'][index], flush=True)
        after = {str(path): digest(path) for path in inputs}
        if before != after:
            raise RuntimeError('An input checkpoint or binding changed during rendering')
        metadata = dict(method='PMGS', binding_kind='affine', model_path=str(weights), binding_path=str(bound),
                        input_sha256=before, input_files_unchanged=True, gaussian_count=len(model.get_xyz),
                        motion_source=config['motion_source'], camera_c2w=config['camera_c2w'],
                        resolution=config['resolution'], fps=config['fps'], controller_z=config['controller_z'],
                        frame_indices=indices, frame_count=len(indices), preview=args.preview,
                        protocol_sha256=digest(ROOT / 'protocol.json'),
                        motion_assets_sha256=digest(ROOT / config['motion_assets']),
                        identity_render_error=identity_error, mesh_partition=partition,
                        elapsed_seconds=time.monotonic() - started,
                        peak_cuda_MiB=torch.cuda.max_memory_allocated() / 2 ** 20)
        (output / ('preview_metadata.json' if args.preview else 'metadata.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    print('PMGS_ANIMATION_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
