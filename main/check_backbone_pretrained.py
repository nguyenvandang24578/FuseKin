"""
check_backbone_pretrained.py - Backbone ResNet trong checkpoint co duoc pretrain khong? (FuseKin)
=================================================================================================

Kiem tra 2 viec, KHONG can GPU:

  [1] Backbone trong checkpoint (Teacher/Student) la trong so PRETRAIN hay NGAU NHIEN:
      - BatchNorm cua ResNet pretrain co running_mean / running_var khac nhau theo kenh.
        Neu MOI lop BN deu co running_mean = 0 va running_var = 1 (gia tri khoi tao cua PyTorch)
        -> backbone chua tung duoc pretrain va chua tung chay train mode -> dac trung anh la NGAU NHIEN.
      - So sanh truc tiep voi backbone cua SPIN (data_final/base_data/spin_model_checkpoint.pth.tar)
        de biet backbone co phai la cua SPIN khong.

  [2] (tuy chon, --cfg) Anh dau vao co duoc chuan hoa mean/std kieu ImageNet khong:
      ResNet pretrain (ImageNet / SPIN) can anh da tru mean, chia std. Neu anh chi nam trong [0, 1]
      thi can them buoc chuan hoa khi dung backbone pretrain.

Cach chay (tu thu muc goc repo):
    python main/check_backbone_pretrained.py \
        --ckpt ./experiment/exp_10-09_17_47/checkpoint/best.pth.tar \
               ./experiment/exp_10-08_00_34/checkpoint/best.pth.tar \
        [--spin ./data_final/base_data/spin_model_checkpoint.pth.tar] \
        [--cfg ./config/train_teacher.yml]      # them de kiem tra [2]

Luu y: --cfg import core.config, file nay tu tao mot thu muc experiment/exp_<thoi gian> rong
(hanh vi san co cua config.py). Co the xoa thu muc do sau khi chay.
"""
import os
import sys
import argparse
import pickle

sys.path.append('./lib')
sys.path.append('./')

import torch

parser = argparse.ArgumentParser(description='Kiem tra backbone co duoc pretrain khong')
parser.add_argument('--ckpt', type=str, nargs='+', required=True, help='mot hoac nhieu checkpoint Teacher/Student')
parser.add_argument('--spin', type=str, default='./data_final/base_data/spin_model_checkpoint.pth.tar',
                    help='checkpoint SPIN de so sanh (bo qua neu khong ton tai)')
parser.add_argument('--cfg', type=str, default='', help='config yaml; neu co thi kiem tra them chuan hoa anh [2]')
args = parser.parse_args()


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


class _PickleShim:
    Unpickler = _NumpyCompatUnpickler
    load = pickle.load
    Pickler = pickle.Pickler
    dump = pickle.dump


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def load_state(path):
    ckpt = torch.load(path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}


def backbone_of(state):
    """Lay cac tensor backbone, bo tien to 'backbone.' (de so voi SPIN dung ten conv1, layer1...)."""
    return {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}


def fmt(t, n=5):
    return '[' + ', '.join(f'{x:.4f}' for x in t.flatten()[:n].tolist()) + ']'


def analyze_bn(bb):
    """Dem so lop BN con o gia tri khoi tao (running_mean = 0, running_var = 1)."""
    names = sorted(k[:-len('.running_mean')] for k in bb if k.endswith('.running_mean'))
    untouched = []
    for n in names:
        rm, rv = bb[n + '.running_mean'].float(), bb[n + '.running_var'].float()
        if torch.all(rm == 0) and torch.all(rv == 1):
            untouched.append(n)
    return names, untouched


def compare_with(bb, ref):
    common = [k for k in bb if k in ref and bb[k].shape == ref[k].shape and bb[k].is_floating_point()]
    if not common:
        return 0, None, None
    max_diff = max((bb[k].float() - ref[k].float()).abs().max().item() for k in common)
    n_equal = sum(torch.allclose(bb[k].float(), ref[k].float(), atol=1e-6) for k in common)
    return len(common), n_equal, max_diff


def main():
    spin_bb = None
    if args.spin and os.path.isfile(args.spin):
        spin_state = load_state(args.spin)
        # checkpoint SPIN luu HMR voi ten conv1 / bn1 / layer1..4 (khong co tien to backbone.)
        spin_bb = {k: v for k, v in spin_state.items()
                   if k.split('.')[0] in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')}
        print(f'SPIN checkpoint: {args.spin} -> {len(spin_bb)} tensor backbone')
        if 'bn1.running_mean' in spin_bb:
            print(f'  SPIN bn1.running_mean[:5] = {fmt(spin_bb["bn1.running_mean"])}')
            print(f'  SPIN bn1.running_var [:5] = {fmt(spin_bb["bn1.running_var"])}')
    else:
        print(f'[BO QUA so sanh SPIN] khong thay file: {args.spin}')

    hr('[1] Backbone trong checkpoint co duoc pretrain khong?')
    for path in args.ckpt:
        print(f'\n  Checkpoint: {path}')
        if not os.path.isfile(path):
            print('    [LOI] khong thay file')
            continue
        state = load_state(path)
        bb = backbone_of(state)
        kind = ('student' if any('fusion.projector_student.' in k for k in state)
                else 'teacher' if any('fusion.joint_proj.' in k for k in state) else 'khong ro')
        print(f'    loai checkpoint: {kind} | so tensor backbone: {len(bb)}')
        if not bb:
            print('    [LOI] checkpoint khong co key backbone.*')
            continue

        if 'bn1.running_mean' in bb:
            print(f'    bn1.running_mean[:5] = {fmt(bb["bn1.running_mean"])}')
            print(f'    bn1.running_var [:5] = {fmt(bb["bn1.running_var"])}')
        if 'bn1.num_batches_tracked' in bb:
            print(f'    bn1.num_batches_tracked = {int(bb["bn1.num_batches_tracked"])}')
        if 'conv1.weight' in bb:
            w = bb['conv1.weight'].float()
            print(f'    conv1.weight: mean={w.mean():.5f} std={w.std():.5f} '
                  f'(khoi tao mac dinh cua PyTorch cho conv1 co std ~0.048)')

        names, untouched = analyze_bn(bb)
        print(f'    BN con o gia tri khoi tao (mean=0, var=1): {len(untouched)}/{len(names)} lop')

        if spin_bb is not None:
            n_common, n_equal, max_diff = compare_with(bb, spin_bb)
            if n_common:
                print(f'    So voi SPIN: {n_equal}/{n_common} tensor trung khop, chenh lech lon nhat = {max_diff:.4g}')

        print('    => KET LUAN: ', end='')
        if names and len(untouched) == len(names):
            print('backbone NGAU NHIEN (chua tung pretrain). Dac trung anh khong mang thong tin tu the.')
        elif spin_bb is not None and n_common and n_equal == n_common:
            print('backbone TRUNG voi SPIN (da nap pretrain SPIN).')
        elif names and len(untouched) == 0:
            print('BN co thong ke da hoc -> backbone CO trong so pretrain (khong trung SPIN; co the la ImageNet).')
        else:
            print('chi mot phan BN o gia tri khoi tao -> can xem them (nap thieu mot phan?).')

    if args.cfg:
        hr('[2] Anh dau vao co duoc chuan hoa mean/std kieu ImageNet khong?')
        from core.config import cfg, update_config
        update_config(args.cfg)
        from utils.jotr_dataset import get_test_dataset
        dataset = get_test_dataset(cfg.DATASET.test_list[0], None)
        imgs = torch.stack([dataset[i][0]['img'] for i in range(0, min(len(dataset), 64 * 50), 50)]).float()
        print(f'  So anh: {imgs.shape[0]} | shape: {tuple(imgs.shape[1:])}')
        print(f'  min={imgs.min():.3f}  max={imgs.max():.3f}')
        for c, name in enumerate(('R', 'G', 'B')):
            print(f'  kenh {c} ({name}): mean={imgs[:, c].mean():.3f}  std={imgs[:, c].std():.3f}')
        if imgs.min() >= 0 and imgs.max() <= 1.0 + 1e-6:
            print('  => Anh trong [0, 1], CHUA chuan hoa. Neu dung backbone pretrain (ImageNet/SPIN) can them')
            print('     (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] truoc khi dua vao backbone.')
            print('     (Thu tu kenh RGB/BGR cung can kiem tra: SPIN/ImageNet dung RGB.)')
        else:
            print('  => Anh co gia tri am hoac > 1 -> co ve da duoc chuan hoa (hoac thang 0-255). Xem lai trong dataset.')


if __name__ == '__main__':
    main()
