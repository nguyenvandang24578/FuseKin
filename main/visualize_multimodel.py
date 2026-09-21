"""Visualize intermediate activations of `Pose2Mesh` from lib/models/Multimodel.py

Usage examples:
python tools/visualize_multimodel.py --checkpoint path/to/checkpoint.pth.tar --device cuda
python tools/visualize_multimodel.py --use-dummy --seq-len 15 --device cpu

The script registers forward hooks on several submodules, runs one forward pass
with either a dummy batch or random inputs, saves activations to numpy and
saves simple visualizations (heatmaps / line plots / 2D PCA of vertices) to
`visualizations/`.
"""
import os
import sys
sys.path.append('./lib')
sys.path.append('./smplpytorch')
sys.path.append('./data')
# sys.path.append('./PoseMamba')

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def import_model():
    # Import here so workspace modules resolve
    from lib.models.ARTS import get_model
    model = get_model(num_joint=19, embed_dim=256, depth=4)
    return model

def register_hooks(model, layers, storage):
    def get_module(root, path):
        parts = path.split('.')
        cur = root
        for p in parts:
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    def make_hook(name):
        def hook(module, input, output):
            try:
                storage[name] = output.detach().cpu().numpy()
            except Exception:
                try:
                    storage[name] = [o.detach().cpu().numpy() if hasattr(o, 'detach') else o for o in output]
                except Exception:
                    storage[name] = str(type(output))
        return hook

    handles = []
    for name in layers:
        mod = get_module(model, name)
        if mod is None:
            print(f"Warning: module {name} not found on model")
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))
    return handles

def plot_activation(name, arr, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"{name}.png"
    try:
        # arr -> numpy
        a = np.array(arr)
        # collapse batch if present
        if a.ndim >= 4 and a.shape[0] == 1:
            a = a[0]

        if a.ndim == 4:
            # (T, J, D) or (T, J, D, ...) plot mean magnitude per joint over time
            # reshape to (T, J, -1)
            T = a.shape[0]
            J = a.shape[1]
            mag = np.linalg.norm(a.reshape(T, J, -1), axis=-1)
            plt.figure(figsize=(6,4))
            plt.imshow(mag.T, aspect='auto', origin='lower')
            plt.colorbar()
            plt.title(name + ' (norm per joint over time)')
            plt.xlabel('time')
            plt.ylabel('joint')
            plt.tight_layout()
            plt.savefig(fn)
            plt.close()
            return

        if a.ndim == 3:
            # (T, J, D) or (B, T, D) -> compute mean abs over feature dim -> line per joint or single line
            if a.shape[1] <= 64:
                # treat as (T, J, D)
                mag = np.linalg.norm(a, axis=-1)  # (T, J)
                plt.figure(figsize=(8,4))
                plt.imshow(mag.T, aspect='auto', origin='lower')
                plt.colorbar()
                plt.title(name + ' (norm per feature over time)')
                plt.xlabel('time')
                plt.ylabel('index')
                plt.tight_layout()
                plt.savefig(fn)
                plt.close()
                return
            else:
                # (B, T, D) plot mean feature magnitude over time
                mag = np.linalg.norm(a, axis=-1).mean(axis=0)
                plt.plot(mag)
                plt.title(name + ' mean feature norm over time')
                plt.xlabel('time')
                plt.tight_layout()
                plt.savefig(fn)
                plt.close()
                return

        if a.ndim == 2:
            plt.figure(figsize=(6,3))
            plt.plot(np.linalg.norm(a, axis=-1))
            plt.title(name + ' norm per row')
            plt.tight_layout()
            plt.savefig(fn)
            plt.close()
            return

        if a.ndim == 1:
            plt.figure(figsize=(6,3))
            plt.plot(a)
            plt.title(name)
            plt.tight_layout()
            plt.savefig(fn)
            plt.close()
            return

        # fallback: save array image via imshow on flattened 2d
        a2 = a.reshape(a.shape[0], -1) if a.ndim >=2 else a
        plt.figure(figsize=(6,4))
        plt.imshow(a2, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(name + ' (fallback)')
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Failed plotting {name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-dummy', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=15)
    parser.add_argument('--num-joints', type=int, default=19)
    parser.add_argument('--outdir', type=str, default='visualizations')
    args = parser.parse_args()

    device = torch.device(args.device)

    # import_model() returns an INSTANCE of ARTS model
    model = import_model()
    model.to(device)
    model.eval()

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=device)
        if 'model' in ck:
            state = ck['model']
        else:
            state = ck
        try:
            model.load_state_dict(state, strict=False)
            print('Loaded checkpoint into model (partial load allowed).')
        except Exception as e:
            print('Warning loading checkpoint:', e)

    # prepare inputs
    B = args.batch_size
    T = args.seq_len
    J = args.num_joints

    if args.use_dummy:
        joints = torch.randn(B, T, J, 3).to(device).float()
        img_feats = torch.randn(B, T, 2048).to(device).float()
        pose2d = torch.randn(B, T, J, 2).to(device).float()
    else:
        # Try to load a small sample from dataset or fallback to dummy
        joints = torch.randn(B, T, J, 3).to(device).float()
        img_feats = torch.randn(B, T, 2048).to(device).float()
        pose2d = torch.randn(B, T, J, 2).to(device).float()

    # layers to hook (dotted attribute paths on ARTS model)
    layers = [
        'pose_mesh_coevo.projoint',
        'pose_mesh_coevo.cfcer',
        'pose_mesh_coevo.mamba_fusion',
        'pose_mesh_coevo.out_proj',
        'pose_mesh_coevo.fusion_linear',
        'pose_mesh_coevo.pose_head',
        'pose_mesh_coevo.shape_head',
        'pose_mesh_coevo.temporal_local_mamba',
        'pose_mesh_coevo.residual'
    ]

    activations = {}
    handles = register_hooks(model, layers, activations)

    with torch.no_grad():
        # ARTS forward pass takes: pose2d, img_feat, is_train
        out = model(pose2d, img_feats, is_train=False)

    # cleanup hooks
    for h in handles:
        h.remove()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # save activations
    np.save(outdir / 'activations.npy', activations)

    # visualize each activation
    for name, arr in activations.items():
        try:
            plot_activation(name.replace('.', '_'), arr, outdir)
        except Exception as e:
            print('Plot failed for', name, e)

    # visualize model outputs if possible
    try:
        # model returns (pose3d, evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput)
        if isinstance(out, (list, tuple)) and len(out) >= 5:
            verts = out[4]  # final_mesh (B, V, 3)
            if isinstance(verts, torch.Tensor):
                v = verts.detach().cpu().numpy()
                # simple 2D scatter of first two coords
                for i in range(v.shape[0]):
                    plt.figure(figsize=(4,4))
                    plt.scatter(v[i,:,0], v[i,:,1], s=2)
                    plt.title(f'vertices_sample_{i}')
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig(outdir / f'vertices_{i}.png')
                    plt.close()
                np.save(outdir / 'vertices.npy', v)
    except Exception as e:
        print('Failed to visualize outputs:', e)

    print('Saved visualizations to', outdir)

if __name__ == '__main__':
    main()
