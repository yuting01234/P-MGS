#
# Mesh-guided Gaussian Splatting - Initialization module
# Mesh surface sampling
#

import numpy as np
import torch
import trimesh


def sample_mesh_surface(mesh, num_samples, seed=None):
    """
    Uniformly sample the mesh surface using area-weighted barycentric
    sampling.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    num_samples : int
        Total surface samples to draw.
    seed : int, optional

    Returns
    -------
    dict
        xyz              : (N, 3) torch.FloatTensor   surface positions
        normal           : (N, 3) torch.FloatTensor   per-point interpolated normals
        face_id          : (N,)   torch.LongTensor    which face each point sits on
        barycentric      : (N, 3) torch.FloatTensor   barycentric coordinates inside the face
        triangles        : (M, 3, 3) torch.FloatTensor cached face vertices (M = num faces)
        face_normals     : (M, 3)   torch.FloatTensor   per-face normals
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces    = np.asarray(mesh.faces,    dtype=np.int64)

    tri         = vertices[faces]                                          # (M, 3, 3)
    face_normals= np.asarray(mesh.face_normals, dtype=np.float32)           # (M, 3)

    # ---- area-weighted face sampling ----
    cross   = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas   = 0.5 * np.linalg.norm(cross, axis=1)
    total   = areas.sum()
    probs   = areas / (total + 1e-12)
    face_id = rng.choice(len(faces), size=num_samples, p=probs)

    # ---- uniform barycentric coordinates (Dirichlet(1,1,1)) ----
    u   = rng.random(num_samples)
    v   = rng.random(num_samples)
    bad = (u + v) > 1.0
    u[bad] = 1.0 - u[bad]
    v[bad] = 1.0 - v[bad]
    w   = 1.0 - u - v
    barycentric = np.stack([u, v, w], axis=1).astype(np.float32)          # (N, 3)

    # ---- surface positions  xyz = u*v0 + v*v1 + w*v2 ----
    chosen_tri = tri[face_id]                                              # (N, 3, 3)
    xyz        = (barycentric[:, :, None] * chosen_tri).sum(axis=1)        # (N, 3)

    # ---- interpolate vertex normals by the barycentric weights ----
    if (hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None
            and len(mesh.vertex_normals) == len(vertices)):
        vnorms       = np.asarray(mesh.vertex_normals, dtype=np.float32)
        chosen_vnorm = vnorms[faces[face_id]]                              # (N, 3, 3)
        normal       = (barycentric[:, :, None] * chosen_vnorm).sum(axis=1)
        normal       = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    else:
        normal = face_normals[face_id]

    return {
        "xyz":           torch.from_numpy(xyz),
        "normal":        torch.from_numpy(normal.astype(np.float32)),
        "face_id":       torch.from_numpy(face_id.astype(np.int64)),
        "barycentric":   torch.from_numpy(barycentric),
        "triangles":     torch.from_numpy(tri),
        "face_normals":  torch.from_numpy(face_normals),
    }
