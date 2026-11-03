#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import torch
import numpy as np

from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud


class GaussianMeshModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self.triangles = None
        self.eps_s0 = 1e-8

        # mesh-guided extras (kept on the model when init comes from
        # `create_from_mesh_guided_init`).  None means legacy init path.
        self.face_id = None
        self.mesh_normal = None
        self.complexity = None
        self.seed_idx = None

    @property
    def get_xyz(self):
        return self._xyz

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):

        self.point_cloud = pcd
        self.triangles = self.point_cloud.triangles
        self.spatial_lr_scale = spatial_lr_scale
        pcd_alpha_shape = pcd.alpha.shape

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()
        scale = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        print("Number of points at initialisation : ",
              alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.vertices = nn.Parameter(
            self.point_cloud.vertices.clone().detach().requires_grad_(True).cuda().float()
        )
        self.faces = torch.tensor(self.point_cloud.faces).cuda()

        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))  # check update_alpha
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = nn.Parameter(scale.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # ------------------------------------------------------------------
    # Mesh-guided initialization (complexity-aware, see
    # games/mesh_splatting/init/pipeline.py for the generating pipeline)
    # ------------------------------------------------------------------
    def create_from_mesh_guided_init(self, init_dict, spatial_lr_scale):
        """
        Initialize the model from a dict returned by
        `init.pipeline.build_mesh_guided_point_cloud`.

        The init dict carries:
            xyz           (M, 3) torch
            colors        (M, 3) numpy float32 in [0, 1]
            scale_init    (M, 3) torch    (isotropic adaptive scale)
            rotation_init (M, 4) torch    (identity quaternion)
            vertices      (V, 3) torch
            faces         (F, 3) torch.long
            triangles     (F, 3, 3) torch
            face_id, complexity, normal, ...   (stored as model attributes)

        The GaMeS `_alpha` tensors are kept around but not used to parameterize
        xyz in this path (we just initialize them to a uniform softmax(1,1,1)
        so that the GaMeS training-loop's `update_alpha()` call is still valid).
        """
        self.spatial_lr_scale = spatial_lr_scale

        xyz    = init_dict["xyz"].float().cuda()
        colors = init_dict["colors"]
        scale_init     = init_dict["scale_init"].float().cuda()
        rotation_init  = init_dict["rotation_init"].float().cuda()
        # stash extras
        self.face_id      = init_dict.get("face_id")
        self.mesh_normal  = init_dict.get("normal")
        self.complexity   = init_dict.get("complexity")
        self.seed_idx     = init_dict.get("seed_idx")

        self.vertices = nn.Parameter(
            init_dict["vertices"].clone().detach()
                 .requires_grad_(True).cuda().float()
        )
        self.faces      = init_dict["faces"].cuda()
        self.triangles  = init_dict["triangles"].cuda()

        # === MeshGS-style precomputed tensors ===
        # face_verts   : (F, 3, 3)  - three vertex positions per face
        # face_normals : (F, 3)     - per-face unit normal in world space
        # We use `self.vertices.detach()` so the buffer never participates
        # in the optimizer (we want mesh to stay fixed).
        with torch.no_grad():
            v = self.vertices.detach()                       # (V, 3)
            fv = v[self.faces]                               # (F, 3, 3)
            e1 = fv[:, 1] - fv[:, 0]
            e2 = fv[:, 2] - fv[:, 0]
            fn = torch.linalg.cross(e1, e2, dim=-1)
            fn = fn / (fn.norm(dim=-1, keepdim=True) + 1e-8)
        self.face_verts   = fv.contiguous()                  # (F, 3, 3)
        self.face_normals = fn.contiguous()                  # (F, 3)
        # Average face circumradius (used as rho in scale reg).
        edge1 = (self.face_verts[:, 1] - self.face_verts[:, 0]).norm(dim=-1)
        edge2 = (self.face_verts[:, 2] - self.face_verts[:, 1]).norm(dim=-1)
        edge3 = (self.face_verts[:, 0] - self.face_verts[:, 2]).norm(dim=-1)
        self.face_avg_radius = float((edge1 + edge2 + edge3).mean().item() / 3.0)
        # Map face_id to GPU long tensor if not already
        if self.face_id is not None and isinstance(self.face_id, torch.Tensor):
            self.face_id = self.face_id.long().cuda()
        # Keep an empty MeshPointCloud-compatible placeholder for save/load
        # and for backward-compat with legacy code (alpha length must equal
        # num_faces for the `update_alpha()` path).
        self.point_cloud = MeshPointCloud(
            alpha=-1,
            points=xyz.detach().cpu(),
            colors=colors,
            normals=init_dict["normal"].numpy(),
            vertices=init_dict["vertices"],
            faces=init_dict["faces"].cpu().numpy(),
            transform_vertices_function=None,
            triangles=self.triangles,
        )

        M = xyz.shape[0]
        print("Number of points at initialisation (mesh-guided): ", M)

        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        features = torch.zeros((M, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(
            0.1 * torch.ones((M, 1), dtype=torch.float, device="cuda")
        )

        # Create a dummy alpha so that the legacy `update_alpha()`
        # call in `train.py` continues to work when the gs_type path
        # for `gs_mesh` is still on.  We never compute `_xyz` from it
        # unless `self._xyz` is None, which only happens if a user
        # explicitly calls `_calc_xyz`.
        num_faces = self.faces.shape[0]
        dummy_alpha = (torch.ones(num_faces, 1, 3) / 3.0).cuda()
        self._alpha = nn.Parameter(dummy_alpha.requires_grad_(True))
        self.alpha  = dummy_alpha

        # === CORE TENSORS ===
        # Use plain tensor (requires_grad=True) instead of nn.Parameter to preserve
        # gradient graph for densification stats.  nn.Parameter breaks the grad
        # connection because it detaches from the original variable.
        self._xyz         = xyz.requires_grad_(True)
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        # Use the adaptive scale_init directly (log space)
        # Must be named _scaling (matching optimizer group "scaling") for
        # densification_postfix / replace_tensor_to_optimizer to work.
        self._scaling = nn.Parameter(
            torch.log(torch.clamp(scale_init, min=1e-6)).requires_grad_(True)
        )

        # === SCALING / ROTATION ===
        # In the mesh-guided path we already have valid per-Gaussian scales
        # (from `scale_init`) and identity rotations.  The legacy
        # `prepare_scaling_rot()` recomputes them from `self._scale * alpha`
        # and therefore crashes when `_scale` is empty.  Keep `_scale` shaped
        # as the GaMeS legacy code expects (M, 1) so any external code that
        # touches it still works.
        self._scale  = nn.Parameter(torch.ones((M, 1), device="cuda"))
        # Build covariance from `scale_init` + identity rotation once at init.
        self._build_scaling_rot_from_init(scale_init, rotation_init)

        self._opacity  = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((M,), device="cuda")

        # Mark this model as mesh-guided so training-time hooks know to skip
        # the legacy alpha-driven update.
        self._is_mesh_guided = True

    def _build_scaling_rot_from_init(self, scale_init, rotation_init):
        """Compute `self._scaling` and `self._rotation` from raw init tensors."""
        s = scale_init
        M = s.shape[0]
        L = torch.zeros((M, 3, 3), device=s.device)
        L[:, 0, 0] = s[:, 0]
        L[:, 1, 1] = s[:, 1]
        L[:, 2, 2] = s[:, 2]
        # IMPORTANT: scaling must be in *log space* for `scaling_activation = exp`.
        self._scaling = nn.Parameter(
            torch.log(torch.clamp(s, min=1e-6)).requires_grad_(True)
        )
        self._rotation = nn.Parameter(rotation_init.requires_grad_(True))

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        _xyz = torch.matmul(
            self.alpha,
            self.triangles
        )
        self._xyz = _xyz.reshape(
                _xyz.shape[0] * _xyz.shape[1], 3
            )
        
    def prepare_scaling_rot(self):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from
        centroid to 2nd vertex onto subspace spanned by v0 and v1.
        """
        # Mesh-guided models already carry per-Gaussian `scale_init` and
        # identity rotations; recomputing from `_scale * alpha` would wipe
        # the init (and crash because `_scale` is a no-op placeholder).
        if getattr(self, "_is_mesh_guided", False):
            return

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = self.triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + self.eps_s0)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + self.eps_s0
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + self.eps_s0)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = self.eps_s0 * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
        self._scaling = torch.log(
            torch.nn.functional.relu(self._scale * scales.flatten(start_dim=0, end_dim=1)) + self.eps_s0
        )
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    def freeze_geometry(self):
        """
        SuGaR-style refinement for mesh-guided model.

        In addition to xyz/scaling/rotation (handled by the parent), also
        freeze `_alpha` (the dummy GaMeS barycentric weights) since the
        mesh-guided code-path no longer reconstructs xyz from alpha. We
        also explicitly freeze the mesh `vertices` parameter if present
        (it is registered on `self.vertices` after the init).
        """
        super().freeze_geometry()
        for group in self.optimizer.param_groups:
            if group.get("name") == "alpha":
                for p in group["params"]:
                    p.requires_grad_(False)

    def rebind_for_freeze(self, scene_info):
        """
        After `load_ply` overwrites the model parameters from a checkpoint
        PLY, restore all the mesh-guided attributes needed by the training
        loop (faces, vertices, triangles, face_id, etc.) so MeshGS losses
        and the GaMeS alpha path still work.
        """
        pcd = scene_info.point_cloud
        # vertices & faces come from the original MeshPointCloud (or the
        # mesh-guided init dict, whichever was attached).
        if hasattr(pcd, "vertices"):
            vertices = torch.as_tensor(np.asarray(pcd.vertices)).float().cuda()
        elif isinstance(scene_info, dict):
            vertices = scene_info["vertices"].float().cuda()
        else:
            raise RuntimeError("rebind_for_freeze: no vertices found in scene_info")
        if hasattr(pcd, "faces"):
            faces = torch.as_tensor(np.asarray(pcd.faces)).long().cuda()
        else:
            raise RuntimeError("rebind_for_freeze: no faces found in scene_info")

        self.faces = faces
        self.vertices = nn.Parameter(vertices, requires_grad=False)  # frozen
        self.triangles = torch.as_tensor(np.asarray(pcd.triangles.cpu())).float().cuda()

        # Re-build face_verts / face_normals (used by MeshGS losses).
        with torch.no_grad():
            v = self.vertices
            fv = v[faces]                                     # (F, 3, 3)
            e1 = fv[:, 1] - fv[:, 0]
            e2 = fv[:, 2] - fv[:, 0]
            fn = torch.linalg.cross(e1, e2, dim=-1)
            fn = fn / (fn.norm(dim=-1, keepdim=True) + 1e-8)
        self.face_verts = fv.contiguous()
        self.face_normals = fn.contiguous()
        # Average face circumradius (used as rho in scale reg).
        edge1 = (self.face_verts[:, 1] - self.face_verts[:, 0]).norm(dim=-1)
        edge2 = (self.face_verts[:, 2] - self.face_verts[:, 1]).norm(dim=-1)
        edge3 = (self.face_verts[:, 0] - self.face_verts[:, 2]).norm(dim=-1)
        self.face_avg_radius = float((edge1 + edge2 + edge3).mean().item() / 3.0)
        # Dummy alpha tensor (mesh-guided path no longer reconstructs xyz
        # from alpha, but legacy code may still touch this attribute).
        if not hasattr(self, "_alpha") or self._alpha is None or self._alpha.numel() == 0:
            num_faces = self.faces.shape[0]
            self._alpha = nn.Parameter(
                (torch.ones(num_faces, 1, 3) / 3.0).cuda(),
                requires_grad=False,
            )
            self.alpha = self._alpha.detach()
        # Restore mesh-guided flag (cleared by PLY load).
        self._is_mesh_guided = True
        print(f"[Freeze][MeshGuided] Rebound: {self.faces.shape[0]} faces, "
              f"{self.vertices.shape[0]} vertices, "
              f"avg_radius={self.face_avg_radius:.4f}, "
              f"model has {self.get_xyz.shape[0]} Gaussians")

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0
        """
        # Mesh-guided: `_alpha` is a dummy placeholder; `_xyz` is already
        # computed in `create_from_mesh_guided_init` and must NOT be
        # overwritten by the legacy barycentric recomputation below.
        if getattr(self, "_is_mesh_guided", False):
            return
        self.alpha = torch.relu(self._alpha) + 1e-8
        self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)
        self.triangles = self.vertices[self.faces]
        self._calc_xyz()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l_params = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        # alpha is mesh-guided dummy; safe to skip optimizing it.
        if self._alpha is not None and self._alpha.numel() > 0:
            l_params.append(
                {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"}
            )

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)
        
        # Store position LR scheduler args for mesh-guided model
        self.position_lr_init = training_args.position_lr_init
        self.position_lr_final = training_args.position_lr_final
        self.position_lr_delay_mult = training_args.position_lr_delay_mult
        self.position_lr_max_steps = training_args.position_lr_max_steps

    def _prune_optimizer(self, mask):
        """
        Override to skip non-Gaussian param groups (e.g. 'vertices', 'alpha').
        Only prune Gaussian-level parameters that have the same size as xyz.
        """
        optimizable_tensors = {}
        gaussian_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in gaussian_names:
                continue  # skip 'vertices', 'alpha', etc.
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

        l_params = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init, "name": "xyz"},
            {'params': [self.vertices], 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
        ]
        # alpha is mesh-guided dummy; safe to skip optimizing it.
        if self._alpha is not None and self._alpha.numel() > 0:
            l_params.append(
                {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"}
            )

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step - exponential decay with delay """
        # Compute delay multiplier based on iteration progress
        delay_mult = self.position_lr_delay_mult
        max_steps = self.position_lr_max_steps
        
        # Delay the start of LR decay until densification is complete
        delay_from_iter = 5000  # Start decay after initial densification phase
        
        if iteration < delay_from_iter:
            lr = self.position_lr_init
        else:
            step = min(iteration - delay_from_iter, max_steps - delay_from_iter)
            delay_weight = min(1.0, (iteration - delay_from_iter) / 500.0)  # Gradual ramp
            lr = self.position_lr_init * (delay_mult + (1 - delay_mult) * delay_weight)
            # Exponential decay
            lr = lr * (self.position_lr_final / self.position_lr_init) ** (step / max_steps)
            lr = max(lr, self.position_lr_final)
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = lr
                break

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Override to skip non-Gaussian param groups (vertices, alpha).
        Only concatenate Gaussian-level parameters.
        """
        optimizable_tensors = {}
        gaussian_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in gaussian_names:
                continue  # skip 'vertices', 'alpha', etc.
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """
        Override to skip 'vertices' group (mesh is fixed; not part of optimizer for mesh-guided).
        Only handles xyz, features, opacity, scaling, rotation.
        """
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        # Filter out groups that aren't in the optimizer
        existing_names = {g["name"] for g in self.optimizer.param_groups}
        d = {k: v for k, v in d.items() if k in existing_names}

        if not d:
            return

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        if "xyz" in optimizable_tensors:
            self._xyz = optimizable_tensors["xyz"]
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]
        if "opacity" in optimizable_tensors:
            self._opacity = optimizable_tensors["opacity"]
        if "scaling" in optimizable_tensors:
            self._scaling = optimizable_tensors["scaling"]
        if "rotation" in optimizable_tensors:
            self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def prune_points(self, mask):
        """Override to skip non-Gaussian param groups (vertices, alpha)."""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        if "xyz" in optimizable_tensors:
            self._xyz = optimizable_tensors["xyz"]
        if "f_dc" in optimizable_tensors:
            self._features_dc = optimizable_tensors["f_dc"]
        if "f_rest" in optimizable_tensors:
            self._features_rest = optimizable_tensors["f_rest"]
        if "opacity" in optimizable_tensors:
            self._opacity = optimizable_tensors["opacity"]
        if "scaling" in optimizable_tensors:
            self._scaling = optimizable_tensors["scaling"]
        if "rotation" in optimizable_tensors:
            self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # Keep face_id aligned with surviving Gaussians so MeshGS losses
        # stay valid after pruning.
        if self.face_id is not None and self.face_id.shape[0] == valid_points_mask.shape[0]:
            self.face_id = self.face_id[valid_points_mask]

    def _extend_face_id_clone(self, selected_pts_mask):
        """Append one face_id per cloned Gaussian (inherits from parent)."""
        if self.face_id is None:
            return
        if self.face_id.shape[0] != self.get_xyz.shape[0] - selected_pts_mask.sum().item():
            # Length mismatch; safest fallback: re-extend at end via _reconcile_face_id
            self._face_id_dirty = True
            return
        new_fid = self.face_id[selected_pts_mask]
        self.face_id = torch.cat([self.face_id, new_fid], dim=0)

    def _extend_face_id_split(self, selected_pts_mask, n_copies):
        """Append n_copies face_ids per split Gaussian (inherits from parent)."""
        if self.face_id is None:
            return
        new_fid = self.face_id[selected_pts_mask].repeat(n_copies)
        self.face_id = torch.cat([self.face_id, new_fid], dim=0)

    def _reconcile_face_id(self):
        """If face_id got out of sync with _xyz, reset to nearest face for each Gaussian.
        Used as a safety net after densify+prune cycles.
        """
        if self.face_id is None or self.face_id.shape[0] == self.get_xyz.shape[0]:
            return
        if not getattr(self, "_face_id_dirty", False):
            return
        # Brute-force: for each Gaussian, find nearest face centroid
        xyz = self.get_xyz.detach()
        centroids = self.face_verts.mean(dim=1)             # (F, 3)
        d = torch.cdist(xyz, centroids)                      # (N, F)
        self.face_id = d.argmin(dim=-1).long()
        self._face_id_dirty = False

    # ------------------------------------------------------------------
    # Override densify_and_clone / densify_and_split so face_id stays
    # aligned with the Gaussian count after densification.
    # ------------------------------------------------------------------
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        if self.face_id is not None:
            n_before = self.face_id.shape[0]
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        if self.face_id is not None and self.face_id.shape[0] == n_before:
            self._extend_face_id_clone(selected_pts_mask)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        if self.face_id is not None:
            n_before = self.face_id.shape[0]
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        from utils.general_utils import build_rotation
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        if self.face_id is not None and self.face_id.shape[0] == n_before:
            self._extend_face_id_split(selected_pts_mask, N)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def save_ply(self, path):
        self.update_alpha()
        self.prepare_scaling_rot()
        self._save_ply(path)

        attrs = self.__dict__
        additional_attrs = [
            '_alpha',
            '_scale',
            'point_cloud',
            'triangles',
            'vertices',
            'faces',
            # Mesh-guided extras (may be None for legacy GaMeS init).
            'face_id',
            'complexity',
            'seed_idx',
            'mesh_normal',
            'face_verts',
            'face_normals',
            'face_avg_radius',
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            v = attrs.get(attr_name, None)
            if v is not None:
                save_dict[attr_name] = v

        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)


    def load_ply(self, path):
        """Load a trained Gaussian model from PLY + model_params.pt.

        Note: PLY contains the trained `_scaling` (log space), `_rotation`,
        and `_xyz`. The `model_params.pt` only has mesh data and alpha
        overrides. Do NOT overwrite `_scaling` with the (linear) `_scale`
        from model_params.pt.
        """
        self._load_ply(path)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        try:
            params = torch.load(path_model, weights_only=False)
        except TypeError:
            params = torch.load(path_model)
        alpha = params['_alpha']
        scale = params['_scale']
        if 'vertices' in params:
            self.vertices = params['vertices']
        if 'triangles' in params:
            self.triangles = params['triangles']
        if 'faces' in params:
            self.faces = params['faces']
        # CRITICAL FIX: do NOT overwrite _scaling with _scale (which is just
        # a placeholder linear scaling tensor, NOT log-scale). The PLY file
        # already contains the correctly trained _scaling in log space.
        # Save _scale only for backward compatibility but do not use it as scaling.
        self._alpha = nn.Parameter(alpha)
        self._scale = nn.Parameter(scale)
        # Mesh-guided render: flag the model so legacy GaMeS alpha/scaling
        # paths (update_alpha, prepare_scaling_rot) short-circuit. Otherwise
        # `prepare_scaling_rot` crashes on size mismatch between the mesh
        # face count and the densified PLY GS count.
        self._is_mesh_guided = True
