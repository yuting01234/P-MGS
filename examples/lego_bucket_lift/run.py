"""Render an edit animation with the bundled rig and reuse matching completed output."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from common import PROJECT, ROOT, binding_path, checkpoint, digest, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-path', type=Path, required=True, help='Checkpoint aligned with the bundled rig reference geometry')
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--binding-path', type=Path,
                        help='Affine-normal binding; default: bindings/lego/binding_affine_normal_v1.pt, then beside the PLY')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'animations' / 'lego_bucket_lift')
    parser.add_argument('--gpu', help='CUDA device selection; otherwise retain CUDA_VISIBLE_DEVICES')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--force', action='store_true', help='Overwrite previously rendered frames')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without loading checkpoints')
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    command = [sys.executable, str(ROOT / 'render_rig.py'), '--model-path', str(args.model_path.expanduser().resolve()),
               '--iteration', str(args.iteration), '--output-root', str(output)]
    if args.binding_path:
        command += ['--binding-path', str(args.binding_path.expanduser().resolve())]
    if args.gpu is not None:
        command += ['--gpu', args.gpu]
    if args.preview:
        command.append('--preview')
    if args.dry_run:
        print(json.dumps(dict(command=command, output_root=str(output)), indent=2))
        return
    weights = checkpoint(args.model_path, args.iteration)
    bound = binding_path(weights, args.binding_path)
    metadata_path = output / ('preview_metadata.json' if args.preview else 'metadata.json')
    frame_directory = output / ('preview' if args.preview else 'frames')
    config = protocol()
    indices = [0, 15, 29, 59] if args.preview else list(range(config['frame_count']))
    if metadata_path.is_file() and not args.force:
        metadata = json.loads(metadata_path.read_text())
        expected = {str(p): digest(p) for p in [weights / 'point_cloud.ply', weights / 'model_params.pt', bound]}
        if (metadata.get('input_sha256') == expected and metadata.get('frame_indices') == indices
                and metadata.get('binding_kind') == 'affine'
                and metadata.get('protocol_sha256') == digest(ROOT / 'protocol.json')
                and metadata.get('motion_assets_sha256') == digest(ROOT / config['motion_assets'])
                and all((frame_directory / ('frame_%04d.png' % i)).is_file() for i in indices)):
            print('REUSE_COMPLETE', output)
            return
        raise RuntimeError('Existing output has different or incomplete provenance; use a new --output-root or --force')
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'MAX_JOBS']:
        env.setdefault(name, '4')
    env['PYTHONUNBUFFERED'] = '1'
    log = output / ('preview.log' if args.preview else 'render.log')
    print('START_PMGS', 'LOG', log, flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit('Rendering failed (%s); inspect %s' % (result.returncode, log))
    print('COMPLETE_PMGS', output, flush=True)


if __name__ == '__main__':
    main()
