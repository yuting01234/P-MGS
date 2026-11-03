"""Validate and replay the published Qgain per-face counts.

The full-project E4 experiments changed only the adaptive sampler's face
selection. Replacing that one expression preserves the original within-face
sampling and its random-number stream, scales, rotations, and initial colors.
"""
import hashlib
import inspect

from .workflow import ROOT, allocation_manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_allocation(scene):
    import numpy as np

    entry = allocation_manifest()["scenes"][scene]
    path = ROOT / "configs" / "allocations" / entry["file"]
    if digest(path) != entry["sha256"]:
        raise ValueError("Allocation checksum mismatch: " + str(path))
    with np.load(path, allow_pickle=False) as archive:
        counts, lower, upper = (archive[name].copy() for name in ("counts", "lower", "upper"))
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Allocation counts must be integers")
    if counts.shape != (entry["num_faces"],) or int(counts.sum()) != entry["gaussians"]:
        raise ValueError("Allocation shape or budget differs from its manifest")
    if lower.shape != counts.shape or upper.shape != counts.shape:
        raise ValueError("Allocation bounds have the wrong shape")
    if not ((counts >= lower).all() and (counts <= upper).all()):
        raise ValueError("Allocation violates the saved coverage/support bounds")
    if int(counts.max()) != entry["maximum_per_face"]:
        raise ValueError("Allocation maximum differs from its manifest")
    return counts, entry


def matching_allocation(signature):
    """Find a plan only when every mesh/camera/image signature matches."""
    matches = [scene for scene, entry in allocation_manifest()["scenes"].items()
               if all(signature.get(key) == value for key, value in entry["signature"].items())]
    if len(matches) > 1:
        raise ValueError("More than one bundled allocation has the same input signature")
    return matches[0] if matches else None


def camera_signature(mesh, cameras):
    import numpy as np

    signature = hashlib.sha256()
    for camera in cameras:
        signature.update(camera.image_name.encode())
        signature.update(np.asarray(camera.R, np.float32).tobytes())
        signature.update(np.asarray(camera.T, np.float32).tobytes())
        signature.update(np.asarray(
            [camera.FovX, camera.FovY, camera.width, camera.height], np.float64
        ).tobytes())
        intrinsics = getattr(camera, "intrinsics", None)
        if intrinsics is not None:
            signature.update(np.asarray(intrinsics, np.float64).tobytes())
        signature.update(np.asarray(camera.image.convert("RGB"), np.uint8).tobytes())
    return dict(
        vertices_sha256=hashlib.sha256(np.asarray(mesh.vertices, np.float32).tobytes()).hexdigest(),
        faces_sha256=hashlib.sha256(np.asarray(mesh.faces, np.int64).tobytes()).hexdigest(),
        train_camera_image_sha256=signature.hexdigest(), train_views=len(cameras),
    )


def patch_sampler(module, counts):
    original = module.adaptive_gaussian_sampling
    source = inspect.getsource(original)
    expression = "    sampled_face = rng.choice(num_faces, size=M, p=face_w)"
    if source.count(expression) != 1:
        raise RuntimeError("Adaptive sampler changed; review Qgain replay before using this source")
    replacement = (
        "    sampled_face = np.repeat(np.arange(num_faces, dtype=np.int64), _qgain_counts)\n"
        '    assert sampled_face.size == M, "Qgain budget differs from the requested budget"'
    )
    namespace = dict(original.__globals__)
    namespace["_qgain_counts"] = counts
    exec(compile(source.replace(expression, replacement),
                 original.__code__.co_filename + ":pmgs_qgain_counts", "exec"), namespace)
    patched = namespace["adaptive_gaussian_sampling"]
    module.adaptive_gaussian_sampling = patched
    return patched


def initial_fingerprints(initialization):
    result = {}
    for key in ("xyz", "scale_init", "rotation_init", "face_id"):
        result[key] = hashlib.sha256(initialization[key].cpu().numpy().tobytes()).hexdigest()
    result["colors"] = hashlib.sha256(initialization["colors"].tobytes()).hexdigest()
    return result

