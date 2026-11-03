"""Original Blender rig motion sampled at the native bound-face centroids."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from common import ROOT, protocol, finite


class RigField:
    chunk = 200000

    def __init__(self, device='cuda'):
        with np.load(ROOT / protocol()['motion_assets']) as assets:
            self.centroids = assets['centroids'].copy()
            self.parts = assets['controls'].copy()
            delta = assets['delta_unique'][assets['frame_indices']]
            self.delta = torch.from_numpy(delta).to(device)
        self.tree = cKDTree(self.centroids)
        self.info = dict(source='Original evaluated Blender Controlpanel_Arm rig',
                         association='Nearest valid reference face centroid',
                         reference_faces=len(self.centroids),
                         coordinate_transform='identity',
                         bands='Original evaluated triangle affine frames, including both rubber bands')

    def associate(self, centers):
        hits = []
        for start in range(0, len(centers), self.chunk):
            _, indices = self.tree.query(centers[start:start + self.chunk], workers=4)
            hits.append(indices)
        return np.concatenate(hits)

    def prepare_bound(self, vertices, faces):
        faces = faces.long()
        triangles = vertices[faces].detach()
        indices = self.associate(triangles.mean(1).cpu().numpy())
        parts = self.parts[indices]
        original = triangles.reshape(-1, 3).contiguous()
        edited_faces = torch.arange(len(original), device=vertices.device).reshape(-1, 3)
        labels = torch.from_numpy(np.repeat(parts, 3).astype(np.int64)).to(vertices.device)
        if not torch.equal(original[edited_faces], triangles):
            raise RuntimeError('Splitting face corners changed the ordered rest triangles')
        moving = (self.delta - torch.eye(4, device=self.delta.device)).abs().amax((0, 2, 3)) > 1e-6
        part_tensor = torch.from_numpy(parts.astype(np.int64)).to(self.delta.device)
        info = dict(original_vertices=len(vertices), ordered_faces=len(faces), editing_vertices=len(original),
                    moving_faces=int(moving[part_tensor].sum()),
                    ordered_rest_triangles_preserved=True, face_to_gaussian_assignments_preserved=True,
                    seam_rule='Duplicate face corners; one rig transform per ordered face')
        return original, edited_faces, labels, info

    def apply_bound(self, original, labels, index):
        if index is None:
            return original.clone()
        if index < 0 or index >= len(self.delta):
            raise IndexError(index)
        result = torch.empty_like(original)
        for start in range(0, len(original), self.chunk):
            stop = min(start + self.chunk, len(original))
            transforms = self.delta[index, labels[start:stop]].to(original.dtype)
            result[start:stop] = (transforms[:, :3, :3] @ original[start:stop, :, None]).squeeze(-1) + transforms[:, :3, 3]
        finite('posed bound mesh', result)
        return result
