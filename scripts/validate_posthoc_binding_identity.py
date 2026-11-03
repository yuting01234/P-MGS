#!/usr/bin/env python3
"""Validate saved post-hoc binding at the exact undeformed reference mesh.

No retraining, filtering, checkpoint mutation or rest-pose bypass. Full test
cameras are paired with their original GT images. Metrics use float render
tensors before PNG quantization. Existing source and inputs remain read-only.
"""
import argparse
import csv
import gc
import importlib.util
import json
import logging
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

LOG=logging.getLogger('binding_identity')


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gt_image(dataset,frame,resolution,white_background=False):
    path=dataset/frame['file_path']
    if not path.suffix:
        path=path.with_suffix('.png')
    with Image.open(str(path)) as image:
        if image.size!=(resolution,resolution):
            raise ValueError('Expected native resolution %s, found %s'%(resolution,image.size))
        rgba=np.array(image.convert('RGBA'))/255.0
    background=np.array([1.,1.,1.]) if white_background else np.zeros(3)
    # Exact GT composition/uint8 conversion from scene/dataset_readers.py.
    composed=rgba[:,:,:3]*rgba[:,:,3:4]+background*(1-rgba[:,:,3:4])
    array=np.ascontiguousarray((composed*255.).astype(np.uint8).transpose(2,0,1))
    return torch.from_numpy(array).cuda().float()/255.,path


def psnr64(prediction,gt):
    mse=(prediction.double()-gt.double()).square().mean(dim=(1,2))
    if bool((mse<=0).any()):
        raise ValueError('Zero MSE to GT; PSNR undefined for finite JSON report')
    # Match the archived project evaluation: psnr(CHW).mean() is channel-wise
    # PSNR followed by an arithmetic mean, rather than PSNR of RGB-mean MSE.
    return float((-10*torch.log10(mse)).mean())


def psnr_joint_rgb64(prediction,gt):
    mse=(prediction.double()-gt.double()).square().mean()
    return float(-10*torch.log10(mse))


def center_errors(binding,snapshot,rest_xyz,chunk_size):
    # Independently reconstruct the user's equation from saved w,d,face IDs.
    vertices=snapshot['vertices'].double()
    faces=snapshot['faces']
    errors=[]
    float32_errors=[]
    implementation_max=0.
    for start in range(0,len(rest_xyz),chunk_size):
        sl=slice(start,start+chunk_size)
        triangles=vertices[faces[binding['face_id'][sl]]]
        e1,e2=triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]
        normal=torch.linalg.cross(e1,e2,dim=1)
        normal=normal/normal.norm(dim=1,keepdim=True)
        restored=(binding['barycentric'][sl,:,None]*triangles).sum(1)+binding['normal_offset'][sl,None]*normal
        original=snapshot['xyz'][sl].double()
        errors.append((restored-original).norm(dim=1))
        float32_errors.append((restored.float().double()-original).norm(dim=1))
        implementation_max=max(implementation_max,float((restored-rest_xyz[sl]).abs().max()))
    error=torch.cat(errors)
    error32=torch.cat(float32_errors)
    if implementation_max>1.e-12:
        raise ValueError('Explicit equation and deformation implementation differ')
    diagonal=float((vertices.max(0).values-vertices.min(0).values).norm())
    return dict(e_mu_mean_l2=float(error.mean()),e_mu_median_l2=float(error.median()),
        e_mu_p95_l2=float(torch.quantile(error,.95)),e_mu_max_l2=float(error.max()),
        e_mu_rms_l2=float(error.square().mean().sqrt()),
        e_mu_mean_l2_after_renderer_float32_cast=float(error32.mean()),
        e_mu_max_l2_after_renderer_float32_cast=float(error32.max()),
        e_mu_nonzero_after_renderer_float32_cast=int((error32>0).sum()),
        explicit_equation_vs_deform_max_abs_error=implementation_max,
        mesh_bbox_diagonal=diagonal,e_mu_mean_relative_to_mesh_bbox_diagonal=float(error.mean())/diagonal,
        coordinate_units='Original scene coordinates',geometry_dtype='float64',renderer_dtype='float32')


def run_scene(scene,args,repo,p,affine):
    from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
    from renderer.gaussian_renderer import render
    started=time.monotonic()
    out=args.output_root/scene
    out.mkdir(parents=True,exist_ok=False)
    paths=import_file('_pmgs_identity_paths',repo/'pmgs/checkpoints.py')
    if args.model_path:
        checkpoint=paths.resolve_checkpoint(args.model_path,args.iteration)
        run=paths.run_directory(checkpoint)
    else:
        run=paths.resolve_scene_run(args.run_root,scene,args.iteration)
        checkpoint=paths.resolve_checkpoint(run,args.iteration)
    binding_path=args.binding_root/scene/'binding_affine_normal_v1.pt'
    protected=[checkpoint/name for name in ['point_cloud.ply','model_params.pt']]+[binding_path]
    if (run/'cfg_args').is_file():
        protected.append(run/'cfg_args')
    hashes={str(path):p.sha256(path) for path in protected}
    report=dict(scene=scene,status='running',iteration=args.iteration,sha256_before=hashes)
    model=proxy=None
    try:
        if (run/'cfg_args').is_file():
            cfg=(run/'cfg_args').read_text()
            if 'white_background=False' not in cfg or 'white_background=True' in cfg:
                raise ValueError('Expected black-background evaluation configuration')
        snapshot=p.read_checkpoint(checkpoint)
        binding=p.torch_load(binding_path)
        geometry,rest_xyz,rest_factor=p.verify_new_binding(binding,snapshot,affine,args.chunk_size)
        report['geometry_check']=geometry
        report['center_reconstruction']=center_errors(binding,snapshot,rest_xyz,args.chunk_size)
        report['gaussian_count']=len(snapshot['xyz'])
        if len(binding['face_id'])!=len(snapshot['xyz']):
            raise ValueError('Binding filtered the trained Gaussian set')
        dataset=args.source_path or p.resolve_dataset(run,scene,args.dataset_root)
        is_dtu=(dataset/'sfm_scene.json').is_file()
        metadata_path=dataset/('sfm_scene.json' if is_dtu else 'transforms_test.json')
        if is_dtu:
            from scene.dtu_dataset import readDTUCameras
            _,frames=readDTUCameras(dataset,False)
        else:
            metadata=json.loads(metadata_path.read_text())
            frames=metadata['frames']
        if not frames:
            raise ValueError('Test camera metadata contains no frames')
        indices=list(range(len(frames)))
        if args.max_views:
            indices=indices[:args.max_views]
        report.update(dataset=str(dataset),camera_metadata_sha256=p.sha256(metadata_path),
            full_test_views=len(frames),evaluated_views=len(indices),all_test_views_evaluated=len(indices)==len(frames),
            background='black',requested_resolution=args.resolution,metric_images='float RGB,clamped[0,1],before PNG quantization',
            psnr_mse_dtype='float64',psnr_definition='Match original project: arithmetic mean of R,G,B per-channel PSNR',
            additional_psnr_definition='Standard joint RGB PSNR from mean MSE over all RGB samples',
            gt_preprocessing=('Calibrated DTU masked RGB and project camera resizing' if is_dtu else
                              'scene/dataset_readers.py RGBA-to-RGB alpha composition then uint8 conversion'),
            main_before_covariance_path='Original CUDA scale/quaternion rasterizer path (compute_cov3D_python=False)',
            after_covariance_path='Saved binding rest-pose J*L0 covariance factor,packed as float32',
            same_path_control='Original precomputed float32 covariance before vs binding precomputed covariance after')
        with torch.no_grad():
            report['cuda_memory']=p.cuda_budget()
            model=GaussianMeshModel(3)
            model._load_ply(str(checkpoint/'point_cloud.ply'))
            if not torch.equal(model.get_xyz.detach().cpu(),snapshot['xyz']):
                raise ValueError('Loaded native model centers mismatch PLY')
            proxy=p.CovarianceProxy(model,rest_xyz,p.packed_covariance(rest_factor,args.chunk_size))
            if proxy.model is not model:
                raise ValueError('Appearance model changed')
            report['appearance_shared_by_reference']=True
            report['before_after_gaussian_count_equal']=len(proxy.get_xyz)==len(model.get_xyz)
            del rest_xyz,rest_factor,binding,snapshot
            pipe_native=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            pipe_cov=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=True,debug=False)
            background=torch.zeros(3,device='cuda')
            per_view=[]
            for index in indices:
                camera=p.make_camera(dataset,index,args.resolution,1.)
                if is_dtu:
                    gt,path=camera.original_image.cuda(),Path(frames[index].image_path)
                else:
                    gt,path=gt_image(dataset,frames[index],args.resolution)
                report['resolution']=[camera.image_width,camera.image_height]
                before=render(camera,model,pipe_native,background)['render'].detach().clamp(0,1)
                after=render(camera,proxy,pipe_cov,background)['render'].detach().clamp(0,1)
                controlled=render(camera,model,pipe_cov,background)['render'].detach().clamp(0,1)
                if not bool(torch.isfinite(before).all() and torch.isfinite(after).all() and torch.isfinite(controlled).all()):
                    raise ValueError('Nonfinite rendered pixels')
                psnr_before,psnr_after,psnr_control=psnr64(before,gt),psnr64(after,gt),psnr64(controlled,gt)
                difference=(before-after).abs()
                difference_control=(controlled-after).abs()
                per_view.append(dict(view_index=index,image_path=str(path),psnr_before=psnr_before,psnr_after=psnr_after,
                    delta_bind=psnr_after-psnr_before,psnr_before_precomputed_covariance=psnr_control,
                    delta_bind_same_covariance_path=psnr_after-psnr_control,
                    standard_joint_rgb_psnr_before=psnr_joint_rgb64(before,gt),
                    standard_joint_rgb_psnr_after=psnr_joint_rgb64(after,gt),
                    render_mean_abs_difference=float(difference.mean()),render_max_abs_difference=float(difference.max()),
                    render_same_path_mean_abs_difference=float(difference_control.mean()),
                    render_same_path_max_abs_difference=float(difference_control.max())))
                if index==0:
                    p.save_image(out/'before_view00000.png',before.cpu())
                    p.save_image(out/'after_view00000.png',after.cpu())
                    p.save_image(out/'difference_view00000_x1000.png',(difference*1000).cpu())
                del gt,before,after,controlled,difference,difference_control,camera
                if (index+1)%25==0 or index==indices[-1]:
                    LOG.info('%s: %d/%d test views',scene,index+1,len(indices))
            before_mean=statistics.mean(row['psnr_before'] for row in per_view)
            after_mean=statistics.mean(row['psnr_after'] for row in per_view)
            control_mean=statistics.mean(row['psnr_before_precomputed_covariance'] for row in per_view)
            report['render_metrics']=dict(psnr_before=before_mean,psnr_after=after_mean,
                delta_bind=after_mean-before_mean,
                max_abs_delta_bind_per_view=max(abs(row['delta_bind']) for row in per_view),
                mean_abs_delta_bind_per_view=statistics.mean(abs(row['delta_bind']) for row in per_view),
                psnr_before_precomputed_covariance=control_mean,
                delta_bind_same_covariance_path=after_mean-control_mean,
                render_mean_abs_difference=statistics.mean(row['render_mean_abs_difference'] for row in per_view),
                render_max_abs_difference=max(row['render_max_abs_difference'] for row in per_view),
                render_same_path_mean_abs_difference=statistics.mean(row['render_same_path_mean_abs_difference'] for row in per_view),
                render_same_path_max_abs_difference=max(row['render_same_path_max_abs_difference'] for row in per_view))
            joint_before=statistics.mean(row['standard_joint_rgb_psnr_before'] for row in per_view)
            joint_after=statistics.mean(row['standard_joint_rgb_psnr_after'] for row in per_view)
            report['render_metrics'].update(standard_joint_rgb_psnr_before=joint_before,
                standard_joint_rgb_psnr_after=joint_after,standard_joint_rgb_delta_bind=joint_after-joint_before)
            reference_path=run/'evaluation.json'
            if reference_path.is_file():
                reference=json.loads(reference_path.read_text())
                reference_psnr=float(reference['PSNR'])
                report['previous_evaluation_reference']=dict(path=str(reference_path),psnr=reference_psnr,
                    recomputed_before_minus_reference=before_mean-reference_psnr,
                    matches_within_1e_minus4_db=abs(before_mean-reference_psnr)<1.e-4)
            with (out/'per_view.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(per_view[0]))
                writer.writeheader()
                writer.writerows(per_view)
            report['numerical_acceptance']=dict(scene_abs_delta_bind_below_1e_minus4_db=abs(after_mean-before_mean)<1.e-4,
                mean_center_error_below_1e_minus10=report['center_reconstruction']['e_mu_mean_l2']<1.e-10,
                max_center_error_below_1e_minus7=report['center_reconstruction']['e_mu_max_l2']<1.e-7)
            if len(indices)==len(frames) and 'previous_evaluation_reference' in report:
                report['numerical_acceptance']['matches_original_evaluation']=report['previous_evaluation_reference']['matches_within_1e_minus4_db']
            report['numerical_acceptance']['passed']=all(report['numerical_acceptance'].values())
            report['status']='complete' if report['numerical_acceptance']['passed'] else 'numerical_mismatch'
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        LOG.exception('%s: validation failed',scene)
    finally:
        after_hashes={str(path):p.sha256(path) for path in protected if path.is_file()}
        report.update(sha256_after=after_hashes,inputs_unchanged=hashes==after_hashes,
            elapsed_seconds=time.monotonic()-started)
        if hashes!=after_hashes:
            report.update(status='failed',error='Input files changed')
        p.write_json(out/'identity_metrics.json',report)
        del proxy,model
        gc.collect()
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
    return report


def main():
    repo=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--scene',help='One scene label')
    selection.add_argument('--scenes',nargs='+',help='Scene labels, or all to discover complete outputs')
    target=parser.add_mutually_exclusive_group()
    target.add_argument('--run-root',type=Path)
    target.add_argument('--model-path',type=Path)
    parser.add_argument('--source-path',type=Path,help='Dataset directory override for a single model')
    parser.add_argument('--binding-root',type=Path,default=repo/'bindings')
    parser.add_argument('--dataset-root',type=Path,default=repo/'data/nerf_synthetic')
    parser.add_argument('--output-root',type=Path,default=repo/'validation/binding_identity')
    parser.add_argument('--iteration',type=int,default=30000)
    parser.add_argument('--resolution',type=int,default=800)
    parser.add_argument('--chunk-size',type=int,default=100000)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-views',type=int,default=0,help='0=full test set; positive is explicitly a pilot')
    args=parser.parse_args()
    if min(args.iteration,args.resolution,args.chunk_size,args.threads)<=0 or args.max_views<0:
        parser.error('Invalid numeric argument')
    if args.source_path and not args.model_path:
        parser.error('--source-path requires --model-path')
    args.run_root=args.run_root or repo/'output'
    for name in ['run_root','binding_root','dataset_root','output_root']:
        setattr(args,name,getattr(args,name).expanduser().resolve())
    paths=import_file('_pmgs_identity_main_paths',repo/'pmgs/checkpoints.py')
    requested=[args.scene] if args.scene is not None else (args.scenes or ['all'])
    discover=args.scene is None and requested==['all']
    if args.scene is None and 'all' in requested and not discover:
        parser.error('Use --scenes all or explicit scene labels')
    try:
        if args.model_path:
            args.model_path=args.model_path.expanduser().resolve()
            checkpoint=paths.resolve_checkpoint(args.model_path,args.iteration)
            selected=[paths.infer_scene(checkpoint)] if discover else list(dict.fromkeys(requested))
            if len(selected)!=1:parser.error('--model-path accepts exactly one scene label')
        else:
            selected=paths.discover_scenes(args.run_root,args.iteration) if discover else list(dict.fromkeys(requested))
        args.scenes=[paths.scene_label(scene) for scene in selected]
    except (FileNotFoundError,ValueError) as error:
        parser.error(str(error))
    if args.source_path:args.source_path=args.source_path.expanduser().resolve()
    protected_paths=[args.run_root,args.binding_root,args.dataset_root,repo/'scripts',repo/'games',repo/'pmgs',repo/'tools']
    if args.model_path:protected_paths.append(args.model_path.parent if args.model_path.is_file() else args.model_path)
    if args.source_path:protected_paths.append(args.source_path)
    for protected in protected_paths:
        if args.output_root==protected or protected in args.output_root.parents or args.output_root in protected.parents:
            parser.error('output-root must be separate from existing input/code directories')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error('Choose a new empty output directory')
    args.output_root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo))
    p=import_file('_identity_helpers_readonly',repo/'scripts/post_bind_affine.py')
    affine=import_file('_identity_affine_readonly',repo/'games/mesh_splatting/scene/affine_normal_binding.py')
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',stream=sys.stdout)
    summary=dict(status='running',experiment='Post-hoc binding identity on undeformed mesh',scenes=[],
        is_full_test_set=not bool(args.max_views),model_iteration=args.iteration,
        definitions=dict(delta_bind='mean_test_PSNR_after - mean_test_PSNR_before',
            e_mu='mean_i ||sum_k(w_ik v_fik) + d_i n_fi - mu_i||_2'))
    for scene in args.scenes:
        try:
            row=run_scene(scene,args,repo,p,affine)
        except (FileNotFoundError,ValueError) as error:
            LOG.exception('%s: cannot open validation inputs',scene)
            row=dict(scene=scene,status='failed',error=repr(error))
        summary['scenes'].append(row)
        p.write_json(args.output_root/'summary.json',summary)
    summary['passed']=sum(row['status']=='complete' for row in summary['scenes'])
    summary['failed']=len(summary['scenes'])-summary['passed']
    summary['status']='complete' if not summary['failed'] else 'failed'
    p.write_json(args.output_root/'summary.json',summary)
    return int(bool(summary['failed']))


if __name__=='__main__':
    sys.exit(main())
