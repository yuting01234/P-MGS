"""Read-only geometry measurements of a learned model; no optimizer updates."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import _bootstrap
from pmgs.workflow import atomic_json as atomic, checkpoint_paths, set_memory_limit

def stats(a,weights=None):
    a=np.asarray(a,dtype=np.float64)
    assert a.size and np.isfinite(a).all()
    ans=dict(mean=float(a.mean()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),
             p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),maximum=float(a.max()))
    if weights is not None:ans['opacity_weighted_mean']=float(np.average(a,weights=np.maximum(weights,1e-12)))
    return ans

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-path','--output',dest='model_path',type=Path,required=True)
    ap.add_argument('--destination',type=Path)
    ap.add_argument('--iteration',type=int,default=30000)
    ap.add_argument('--memory-fraction',type=float)
    args=ap.parse_args()
    import torch,open3d as o3d
    from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
    from utils.loss_utils import mesh_gaussian_normal
    set_memory_limit(args.memory_fraction);start=time.time()
    out,weights=checkpoint_paths(args.model_path,args.iteration)
    dest=args.destination or out/'geometry.json';assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        model=GaussianMeshModel(3);model.load_ply(str(weights/'point_cloud.ply'))
        assert model._is_mesh_guided
        # The project's renderer loader intentionally restores only render/mesh
        # tensors. Read stored ancestry/rho explicitly for diagnostic use.
        params_path=weights/'model_params.pt'
        try:params=torch.load(str(params_path),map_location='cpu',weights_only=False)
        except TypeError:params=torch.load(str(params_path),map_location='cpu')
        assert 'face_id' in params and len(params['face_id'])==len(model.get_xyz)
        xyz=model.get_xyz.cpu().numpy();opacity=model.get_opacity.flatten().cpu().numpy()
        faces=model.faces.cpu().numpy().astype(np.int64);vertices=model.vertices.cpu().numpy().astype(np.float32)
        fv=vertices[faces];cross=np.cross(fv[:,1]-fv[:,0],fv[:,2]-fv[:,0]);area=np.linalg.norm(cross,axis=1)
        valid=np.flatnonzero(area>0);normals=cross/np.maximum(area[:,None],1e-12)
        fid=params['face_id'].cpu().numpy();good=area[fid]>0
        center=fv[fid].mean(1);plane=np.abs(((xyz-center)*normals[fid]).sum(1))
        gn=mesh_gaussian_normal(model._rotation,model.get_scaling).cpu().numpy()
        ancestral_cos=np.clip((gn*normals[fid]).sum(1),-1,1)
        ray=o3d.t.geometry.RaycastingScene(nthreads=4)
        ray.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces[valid].astype(np.uint32)))
        distance=[];nearest_cos=[]
        for lo in range(0,len(xyz),100000):
            hit=ray.compute_closest_points(o3d.core.Tensor(xyz[lo:lo+100000].astype(np.float32)),nthreads=4)
            ids=valid[hit['primitive_ids'].numpy().astype(np.int64)]
            distance.append(np.linalg.norm(xyz[lo:lo+100000]-hit['points'].numpy(),axis=1))
            nearest_cos.append(np.clip((gn[lo:lo+100000]*normals[ids]).sum(1),-1,1))
        distance=np.concatenate(distance);nearest_cos=np.concatenate(nearest_cos)
        scales=model.get_scaling.cpu().numpy();minimum=scales.min(1);maximum=scales.max(1)
        rho=float(params['face_avg_radius'])
        value=dict(status='complete',model=str(out),GS_final=len(xyz),rho_global_mean_edge_length=rho,
            original_face_valid_fraction=float(good.mean()),
            ancestral_face_plane_distance=stats(plane[good],opacity[good]),
            ancestral_plane_fraction_above_0_01=float((plane[good]>.01).mean()),
            exact_nearest_triangle_distance=stats(distance,opacity),
            signed_normal_error_ancestral=stats(1-ancestral_cos[good],opacity[good]),
            unsigned_normal_error_ancestral=stats(1-np.abs(ancestral_cos[good]),opacity[good]),
            unsigned_normal_error_nearest=stats(1-np.abs(nearest_cos),opacity),
            shortest_scale=stats(minimum,opacity),longest_scale=stats(maximum,opacity),
            longest_scale_over_rho=stats(maximum/max(rho,1e-12),opacity),
            flatness_shortest_over_longest=stats(minimum/np.maximum(maximum,1e-12),opacity),
            opacity=stats(opacity),elapsed_seconds=time.time()-start,
            interpretation='Geometry diagnostics in dataset world units; unsigned normal error treats n and -n as identical Gaussian axes. These do not establish deformation quality.')
        atomic(dest,value);print('GEOMETRY_COMPLETE',str(dest),len(xyz),flush=True)

if __name__=='__main__':main()
