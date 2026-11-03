"""Locate completed PMGS checkpoints without loading model or CUDA code."""
import json
from pathlib import Path


def scene_label(value):
    value = str(value)
    if not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError('Scene labels must be nonempty directory names without path separators')
    return value


def resolve_checkpoint(path, iteration=30000):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.name == 'point_cloud.ply':
        checkpoint = path.parent
    elif (path / 'point_cloud.ply').is_file():
        checkpoint = path
    else:
        checkpoint = path / 'point_cloud' / ('iteration_' + str(iteration))
    for name in ('point_cloud.ply', 'model_params.pt'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(str(checkpoint / name))
    return checkpoint


def run_directory(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.parent.name == 'point_cloud':
        return checkpoint.parent.parent
    for candidate in (checkpoint, *checkpoint.parents):
        if any((candidate / name).is_file() for name in
               ('run_status.json', 'qgain_full_status.json', 'REPLAY_PROTOCOL.json')):
            return candidate
    return checkpoint


def resolve_scene_run(run_root, scene, iteration=30000):
    """Find the latest complete requested iteration beneath a scene directory."""
    scene_root = Path(run_root) / scene_label(scene)
    candidates = [scene_root]
    if scene_root.is_dir():
        candidates.extend(path for path in scene_root.iterdir() if path.is_dir())
    available = []
    for candidate in candidates:
        try:
            checkpoint = resolve_checkpoint(candidate, iteration)
        except FileNotFoundError:
            continue
        modified = max((checkpoint / name).stat().st_mtime_ns
                       for name in ('point_cloud.ply', 'model_params.pt'))
        available.append((modified, str(candidate), candidate))
    if not available:
        raise FileNotFoundError('No complete iteration_%s checkpoint under %s; '
                                'use --model-path for another layout' % (iteration, scene_root))
    return max(available)[2]


def discover_scenes(run_root, iteration=30000):
    """Discover valid scene output directories, without a fixed scene list."""
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError('Run root does not exist: ' + str(root))
    found = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith('.'):
            continue
        try:
            resolve_scene_run(root, candidate.name, iteration)
        except FileNotFoundError:
            continue
        found.append(candidate.name)
    if not found:
        raise FileNotFoundError('No complete scene checkpoints found under ' + str(root))
    return found


def infer_scene(checkpoint):
    """Read a scene label from run metadata, falling back to the run name."""
    run = run_directory(checkpoint)
    for name in ('run_status.json', 'qgain_full_status.json', 'REPLAY_PROTOCOL.json'):
        path = run / name
        if path.is_file():
            metadata = json.loads(path.read_text(encoding='utf-8'))
            if metadata.get('scene'):
                return scene_label(metadata['scene'])
    return scene_label(run.name)
