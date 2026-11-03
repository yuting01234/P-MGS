"""Independent signed-normal binding and local-affine Gaussian deformation.

Implements signed-normal binding and local-affine deformation:
    mu0 = sum(w_k v0_k) + d n0,       sum(w_k) = 1 (signed w allowed)
    J = [e1t,e2t,nt] [e10,e20,n0]^-1
    mut = sum(w_k vt_k) + d nt,       Lt = J L0, Sigma_t = Lt Lt^T

The third basis vector is a UNIT normal. Normal thickness and signed normal
offset consequently remain fixed, including under uniform mesh scaling.
Covariance factors retain the original scales without legacy clamps or an
eigendecomposition of an ill-conditioned covariance matrix. All geometry is
computed in float64; the renderer can receive packed float32 covariance.
"""
import torch

FORMAT = 'signed_normal_local_affine_v1'


def _finite(name, tensor):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError('Nonfinite values in ' + name)


def _quaternion_matrices(quaternions):
    norm = torch.linalg.norm(quaternions, dim=1, keepdim=True)
    if bool((norm == 0).any()):
        raise ValueError('Zero rotation quaternion')
    w, x, y, z = (quaternions / norm).unbind(1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), dim=1).reshape(-1, 3, 3)


def _geometry(vertices, faces):
    tri = vertices[faces]
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    cross = torch.linalg.cross(e1, e2, dim=1)
    area2 = cross.norm(dim=1)
    edge_product = e1.norm(dim=1) * e2.norm(dim=1)
    valid = (edge_product > 0) & (area2 > edge_product * 1e-12)
    normals = torch.zeros_like(cross)
    normals[valid] = cross[valid] / area2[valid, None]
    basis = torch.stack((e1, e2, normals), dim=2)
    return basis, normals, valid


def _nearest_faces(xyz, vertices, faces, valid, chunk_size, threads):
    import numpy as np
    import open3d as o3d
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten().cpu()
    if not len(valid_ids):
        raise ValueError('Reference mesh has no nondegenerate faces')
    scene = o3d.t.geometry.RaycastingScene(nthreads=threads)
    scene.add_triangles(
        o3d.core.Tensor(vertices.detach().cpu().numpy().astype(np.float32)),
        o3d.core.Tensor(faces.detach().cpu()[valid_ids].numpy().astype(np.uint32)))
    found = torch.empty(len(xyz), dtype=torch.int64, device=xyz.device)
    for start in range(0, len(xyz), chunk_size):
        stop = min(start + chunk_size, len(xyz))
        query = xyz[start:stop].detach().cpu().numpy().astype(np.float32)
        hits = scene.compute_closest_points(o3d.core.Tensor(query), nthreads=threads)
        ids = torch.from_numpy(hits['primitive_ids'].numpy().astype(np.int64))
        if bool((ids < 0).any()) or bool((ids >= len(valid_ids)).any()):
            raise ValueError('Nearest-face search returned an invalid triangle')
        found[start:stop] = valid_ids[ids].to(xyz.device)
    return found


@torch.no_grad()
def bind_gaussians(xyz, vertices, faces, rotation_quaternions, log_scales,
                   face_ids=None, chunk_size=100000, threads=4):
    """Bind optimized Gaussians to face planes with signed barycentrics.

    Face association uses the exact Open3D nearest triangle search. The nearest
    POINT is deliberately not used as the new anchor: the Gaussian is projected
    onto that triangle's infinite plane, as required by the new method.
    Optional face_ids reuse an already established face association, useful for
    controlled comparisons and tests. Degenerate assigned faces are rejected.
    """
    if chunk_size <= 0 or threads <= 0:
        raise ValueError('chunk_size and threads must be positive')
    xyz = torch.as_tensor(xyz).detach().to(dtype=torch.float64).clone()
    device = xyz.device
    vertices = torch.as_tensor(vertices, device=device).detach().to(dtype=torch.float64).clone()
    faces = torch.as_tensor(faces, device=device).detach()
    if faces.dtype not in (torch.int32, torch.int64):
        raise ValueError('Mesh face indices must be integers')
    faces = faces.long().clone()
    rotation = torch.as_tensor(rotation_quaternions, device=device).detach().double()
    log_scales = torch.as_tensor(log_scales, device=device).detach().double()
    n = len(xyz)
    if xyz.shape != (n, 3) or not n:
        raise ValueError('Expected nonempty Gaussian positions with shape (N,3)')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError('Invalid reference vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('Invalid reference faces')
    if rotation.shape != (n, 4) or log_scales.shape != (n, 3):
        raise ValueError('Rotation/scale dimensions must match Gaussian count')
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError('Mesh face index outside the vertex range')
    for name, tensor in [('xyz', xyz), ('vertices', vertices),
                         ('rotation', rotation), ('log_scales', log_scales)]:
        _finite(name, tensor)
    basis, normals, valid = _geometry(vertices, faces)
    if not bool(valid.any()):
        raise ValueError('Reference mesh has no nondegenerate faces')
    if face_ids is None:
        assigned = _nearest_faces(xyz, vertices, faces, valid, chunk_size, threads)
    else:
        assigned = torch.as_tensor(face_ids, device=device).detach()
        if assigned.dtype not in (torch.int32, torch.int64):
            raise ValueError('Assigned face indices must be integers')
        assigned = assigned.long().clone()
    if assigned.shape != (n,):
        raise ValueError('Assigned face count must match Gaussian count')
    if int(assigned.min()) < 0 or int(assigned.max()) >= len(faces):
        raise ValueError('Assigned face index out of range')
    if not bool(valid[assigned].all()):
        raise ValueError('A Gaussian is assigned to a degenerate reference face')
    inverse = torch.eye(3, dtype=torch.float64, device=device).repeat(len(faces), 1, 1)
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    for start in range(0, len(valid_ids), chunk_size):
        ids = valid_ids[start:start+chunk_size]
        inverse[ids] = torch.linalg.inv(basis[ids])
    _finite('reference basis inverse', inverse)
    barycentric = torch.empty((n, 3), dtype=torch.float64, device=device)
    normal_offset = torch.empty(n, dtype=torch.float64, device=device)
    factor = torch.empty((n, 3, 3), dtype=torch.float64, device=device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sl = slice(start, stop)
        ids = assigned[sl]
        origin = vertices[faces[ids, 0]]
        delta = xyz[sl] - origin
        d = (delta * normals[ids]).sum(1)
        plane_delta = delta - d[:, None] * normals[ids]
        # Equivalent full-rank 3x3 solve avoids squaring the condition number
        # in the normal equations (A^T A)^-1 A^T from the mathematical sketch.
        local = (inverse[ids] @ plane_delta[:, :, None]).squeeze(-1)
        barycentric[sl] = torch.stack((1-local[:, 0]-local[:, 1], local[:, 0], local[:, 1]), dim=1)
        normal_offset[sl] = d
        factor[sl] = _quaternion_matrices(rotation[sl]) * log_scales[sl].exp()[:, None, :]
    _finite('barycentric coordinates', barycentric)
    _finite('covariance factor', factor)
    return dict(format_version=FORMAT, face_id=assigned,
                barycentric=barycentric, normal_offset=normal_offset,
                reference_basis_inv=inverse, reference_normals=normals,
                reference_vertices=vertices, faces=faces, reference_valid_faces=valid,
                xyz_rest=xyz, covariance_factor_rest=factor)


@torch.no_grad()
def deform_gaussians(binding, new_vertices, chunk_size=100000):
    """Return deformed centers and covariance factors; never mutate rest data.

    The caller must preserve vertex correspondence and face topology. New
    degenerate assigned triangles are rejected rather than silently clamped.
    Unreferenced degenerate triangles are allowed.
    """
    if binding.get('format_version') != FORMAT:
        raise ValueError('This binding is not the independent signed-normal format')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    reference = binding['reference_vertices']
    new_vertices = torch.as_tensor(new_vertices, device=reference.device).detach().double()
    if new_vertices.shape != reference.shape:
        raise ValueError('Edited mesh must preserve the reference vertex count/correspondence')
    _finite('edited vertices', new_vertices)
    basis, normals, valid = _geometry(new_vertices, binding['faces'])
    assigned = binding['face_id']
    if not bool(valid[assigned].all()):
        raise ValueError('Edited mesh has a degenerate face assigned to Gaussians')
    n = len(assigned)
    positions = torch.empty((n, 3), dtype=torch.float64, device=reference.device)
    factors = torch.empty((n, 3, 3), dtype=torch.float64, device=reference.device)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sl = slice(start, stop)
        ids = assigned[sl]
        tri = new_vertices[binding['faces'][ids]]
        anchor = (binding['barycentric'][sl, :, None] * tri).sum(1)
        positions[sl] = anchor + binding['normal_offset'][sl, None] * normals[ids]
        jacobian = basis[ids] @ binding['reference_basis_inv'][ids]
        factors[sl] = jacobian @ binding['covariance_factor_rest'][sl]
    _finite('deformed centers', positions)
    _finite('deformed covariance factors', factors)
    return positions, factors


@torch.no_grad()
def factor_to_rotation_scale(factors):
    """Optional stable conversion of covariance factors to proper R and scales.

    For L=U diag(s) V^T, Sigma=L L^T=U diag(s^2) U^T. Singular values s,
    rather than covariance eigenvalues, are the Gaussian standard deviations.
    One column sign may be flipped to enforce det(R)=+1 without changing Sigma.
    """
    factors = torch.as_tensor(factors).double()
    if factors.ndim != 3 or factors.shape[1:] != (3, 3):
        raise ValueError('Expected covariance factors with shape (N,3,3)')
    _finite('covariance factors', factors)
    rotation, scales, _ = torch.linalg.svd(factors, full_matrices=False)
    signs = torch.where(torch.linalg.det(rotation) < 0, -torch.ones_like(scales[:, 0]), torch.ones_like(scales[:, 0]))
    rotation[:, :, 2] *= signs[:, None]
    if bool((scales <= 0).any()):
        raise ValueError('Singular covariance factor')
    return rotation, scales
