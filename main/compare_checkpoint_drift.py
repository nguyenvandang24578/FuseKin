"""
compare_checkpoint_drift.py — So sanh truc tiep state_dict Teacher vs Student (FuseKin)
===========================================================================================

Muc dich: tra loi cau hoi "phan downstream (sau fusion) cua Student da troi dat khoi
Teacher bao nhieu trong luc train?" — CHI so sanh tensor trong so, khong build model,
khong can forward/GPU. Chay duoc tren CPU, vai giay.

Dung de phan biet 2 gia thuyet khi thay 'feature khop (CKA cao) nhung output van lech
Teacher ~41mm':

  (a) Trong so downstream da TROI DAT that su trong luc train Student (cosine thap,
      rel_L2 cao) -> fix: them KD cho feat_hyper/feat_pose/root_pose_6d/pose_latent.
  (b) Trong so downstream GAN NHU KHONG DOI (cosine ~1, rel_L2 nho), nhung ham do RAT
      NHAY voi sai khac nho con sot o input feature -> fix huong khac: lam feature
      khop sat hon nua, hoac giam do nhay cua downstream (vd them smoothing/regularize).

Cach chay:
    cd <repo_root>
    python main/compare_checkpoint_drift.py \
        --teacher_ckpt <duong_dan_teacher>.pth.tar \
        --student_ckpt <duong_dan_student>.pth.tar \
        [--top_k 15]

Khong can --cfg, khong can CUDA — day chi la so sanh tensor thuan tuy.
"""
import argparse
import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='So sanh drift state_dict Teacher vs Student')
parser.add_argument('--teacher_ckpt', type=str, required=True)
parser.add_argument('--student_ckpt', type=str, required=True)
parser.add_argument('--top_k', type=int, default=15, help='so tensor troi dat nhieu nhat can in chi tiet')
args = parser.parse_args()


# --------------------------------------------------------------------------
# Cung logic lam sach key nhu load_model_weights() trong lib/core/base.py
# --------------------------------------------------------------------------
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


def load_clean_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break
    # bo prefix 'module.' (DataParallel)
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    # ckpt cu: 'teacher_model.' -> 'smpl_model.'
    old_p, new_p = 'teacher_model.', 'smpl_model.'
    state = {(new_p + k[len(old_p):] if k.startswith(old_p) else k): v
             for k, v in state.items()}
    return state


# --------------------------------------------------------------------------
# Cac submodule downstream (sau fusion) can so sanh
# --------------------------------------------------------------------------
DOWNSTREAM_PREFIXES = (
    'smpl_model.norm.',               # LayerNorm truoc spatial_hypers (tren joint_out)
    'smpl_model.node_pe.',
    'smpl_model.spatial_hypers.',
    'smpl_model.pose_embed.',
    'smpl_model.pose_pe.',
    'smpl_model.pose_context_attn.',
    'smpl_model.root_pose_head.',
    'smpl_model.body_pose_head.',
    'smpl_model.shape_embed.',
    'smpl_model.shape_token',
    'smpl_model.shape_head.',
    'smpl_model.cam_head.',
)


def submodule_of(key):
    """Lay ten submodule (segment thu 2 sau 'smpl_model.') de gom nhom khi in bao cao."""
    parts = key.split('.')
    return parts[1] if len(parts) > 1 else key


# --------------------------------------------------------------------------
# [MOI] Gating coefficients cua fusion.cfcer — "van" dieu khien bao nhieu
# thong tin anh duoc cho phep chay vao nhanh joint (va nguoc lai).
#
#   joint_out = coef1(joint_tok) + coef2(joint_att)   <- coef2 = van anh->joint
#   rgb_out   = coef3(rgb_tok)  + coef4(rgb_att)       <- coef4 = van joint->anh
#   joint_out = coef5(joint_out) + coef6(FFN(...))
#   rgb_out   = coef7(rgb_out)  + coef8(FFN(...))
#
# Khoi tao mac dinh = 1.0 cho moi coef. Neu coef2 cua Student thap hon han
# Teacher -> bang chung TRUC TIEP (khong qua suy luan MPJPE) rang Student da
# tu hoc cach KHOA VAN duong anh->joint trong luc train.
# --------------------------------------------------------------------------
FUSION_EXTRA_PREFIXES = (
    'smpl_model.fusion.img_proj.',
    'smpl_model.fusion.pos_emb_img',
    'smpl_model.fusion.norm_img_in.',
)


def print_fusion_gate_report(teacher_state, student_state, n_blocks=3):
    hr("[GATE] He so gating fusion.cfcer — van anh<->joint o tung block")
    print(f"{'block':<8}{'coef':<8}{'y nghia':<26}{'Teacher':>10}{'Student':>10}{'delta':>10}")
    rows = []
    for b in range(n_blocks):
        meanings = {
            'coef1': 'giu joint goc',
            'coef2': 'anh -> joint  (*)',
            'coef3': 'giu anh goc',
            'coef4': 'joint -> anh  (*)',
            'coef5': 'giu joint (FFN)',
            'coef6': 'FFN joint',
            'coef7': 'giu anh (FFN)',
            'coef8': 'FFN anh',
        }
        for c in range(1, 9):
            key = f'smpl_model.fusion.cfcer.blocks.{b}.coef{c}.bias'
            if key not in teacher_state or key not in student_state:
                continue
            vt = teacher_state[key].flatten()[0].item()
            vs = student_state[key].flatten()[0].item()
            tag = meanings[f'coef{c}']
            flag = '  <--' if c in (2, 4) else ''
            print(f"{b:<8}{c:<8}{tag:<26}{vt:>10.4f}{vs:>10.4f}{vs - vt:>+10.4f}{flag}")
            rows.append((b, c, vt, vs))

    coef2_vals = [(vt, vs) for b, c, vt, vs in rows if c == 2]
    if coef2_vals:
        t_avg = sum(v[0] for v in coef2_vals) / len(coef2_vals)
        s_avg = sum(v[1] for v in coef2_vals) / len(coef2_vals)
        print(f"\ncoef2 (van anh->joint) trung binh 3 block: Teacher={t_avg:.4f}  Student={s_avg:.4f}")
        if s_avg < 0.5 * t_avg or s_avg < 0.3:
            print("  [XAC NHAN] coef2 cua Student thap hon han Teacher (hoac gan 0 tuyet doi)")
            print("  -> Day la BANG CHUNG TRUC TIEP: Student da tu hoc cach KHOA VAN anh->joint.")
            print("     Khop hoan toan voi content-ablation test (randn khong doi MPJPE Student)")
            print("     va voi viec Student bat bien voi ca shuffle lan randn trong khi Teacher")
            print("     chi bat bien voi shuffle. Huong fix: them regularizer ep coef2 khong qua")
            print("     nho (vd L2 toi gia tri 1.0, hoac warmup dong bang coef2 vai epoch dau),")
            print("     hoac tang trong so loss buoc model phai dung anh de giam loss further.")
        else:
            print("  -> coef2 khong lech nhieu -> van gating KHONG phai nguyen nhan chinh,")
            print("     can xem lai phan [2]/[3] ben tren (drift o pose_context_attn/heads).")

    # ---- img_proj / pos_emb_img / norm_img_in ----
    extra_rows = []
    for prefix in FUSION_EXTRA_PREFIXES:
        for k in sorted(k for k in teacher_state if k.startswith(prefix)):
            if k not in student_state:
                continue
            wt, ws = teacher_state[k].float().flatten(), student_state[k].float().flatten()
            if wt.shape != ws.shape:
                continue
            diff = ws - wt
            rel = diff.norm().item() / wt.norm().clamp_min(1e-8).item()
            cos = F.cosine_similarity(wt.unsqueeze(0), ws.unsqueeze(0)).item()
            extra_rows.append((k, rel, cos))
    if extra_rows:
        print(f"\n{'key (img_proj / pos_emb_img / norm_img_in)':<55}{'rel_L2':>10}{'cosine':>10}")
        for k, rel, cos in extra_rows:
            print(f"{k:<55}{rel:>10.4f}{cos:>10.4f}")


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def main():
    print(f"Teacher ckpt: {args.teacher_ckpt}")
    print(f"Student ckpt: {args.student_ckpt}")

    teacher_state = load_clean_state_dict(args.teacher_ckpt)
    student_state = load_clean_state_dict(args.student_ckpt)

    rows = []                      # (key, l2, rel_l2, cosine, numel)
    shape_mismatch = []
    only_teacher, only_student = [], []

    for prefix in DOWNSTREAM_PREFIXES:
        t_keys = sorted(k for k in teacher_state if k.startswith(prefix))
        for k in t_keys:
            if k not in student_state:
                only_teacher.append(k)
                continue
            wt = teacher_state[k].float()
            ws = student_state[k].float()
            if wt.shape != ws.shape:
                shape_mismatch.append((k, tuple(wt.shape), tuple(ws.shape)))
                continue
            wt_flat, ws_flat = wt.flatten(), ws.flatten()
            diff = ws_flat - wt_flat
            l2 = diff.norm().item()
            rel_l2 = l2 / wt_flat.norm().clamp_min(1e-8).item()
            cos = F.cosine_similarity(wt_flat.unsqueeze(0), ws_flat.unsqueeze(0)).item()
            rows.append((k, l2, rel_l2, cos, wt_flat.numel()))

        s_keys = sorted(k for k in student_state if k.startswith(prefix))
        for k in s_keys:
            if k not in teacher_state:
                only_student.append(k)

    if not rows:
        print("\n[LOI] Khong tim thay tensor nao khop ten o cac prefix da khai bao. "
              "Kiem tra lai ten submodule trong teacher_student.py co dung voi "
              "DOWNSTREAM_PREFIXES trong script nay khong.")
        return

    hr("[1] TONG QUAN")
    print(f"So tensor so sanh duoc   : {len(rows)}")
    print(f"Shape mismatch           : {len(shape_mismatch)}")
    print(f"Chi co o Teacher         : {len(only_teacher)}")
    print(f"Chi co o Student         : {len(only_student)}")
    if shape_mismatch:
        print("\n  Shape mismatch (toi da 10):")
        for k, st, ss in shape_mismatch[:10]:
            print(f"    {k}: teacher{st} vs student{ss}")
    if only_teacher:
        print(f"\n  Chi co o Teacher (toi da 10): {only_teacher[:10]}")
    if only_student:
        print(f"\n  Chi co o Student (toi da 10): {only_student[:10]}")

    # ---- gom nhom theo submodule (weighted by numel) ----
    hr("[2] TOM TAT THEO SUBMODULE (trung binh co trong so theo so phan tu)")
    grouped = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # sum_l2w, sum_rel_w, sum_cos_w, total_numel
    for k, l2, rel_l2, cos, numel in rows:
        g = grouped[submodule_of(k)]
        g[0] += l2 * numel
        g[1] += rel_l2 * numel
        g[2] += cos * numel
        g[3] += numel

    summary = []
    for name, (sl2, srel, scos, n) in grouped.items():
        summary.append((name, sl2 / n, srel / n, scos / n, n))
    summary.sort(key=lambda x: -x[2])  # sap xep theo rel_L2 giam dan -> troi nhieu nhat len dau

    print(f"{'submodule':<22}{'rel_L2':>10}{'cosine':>10}{'L2':>12}{'numel':>12}")
    for name, l2, rel, cos, n in summary:
        flag = '  <-- TROI DAT NHIEU' if rel > 0.3 or cos < 0.8 else ''
        print(f"{name:<22}{rel:>10.4f}{cos:>10.4f}{l2:>12.4f}{n:>12d}{flag}")

    # ---- top_k tensor troi dat nhieu nhat (theo rel_L2) ----
    hr(f"[3] TOP {args.top_k} TENSOR TROI DAT NHIEU NHAT (theo rel_L2)")
    rows_sorted = sorted(rows, key=lambda r: -r[2])
    print(f"{'key':<55}{'rel_L2':>10}{'cosine':>10}")
    for k, l2, rel_l2, cos, numel in rows_sorted[:args.top_k]:
        print(f"{k:<55}{rel_l2:>10.4f}{cos:>10.4f}")

    # ---- ket luan goi y ----
    print_fusion_gate_report(teacher_state, student_state)

    hr("TONG KET")
    avg_rel = sum(r[2] * r[4] for r in rows) / sum(r[4] for r in rows)
    avg_cos = sum(r[3] * r[4] for r in rows) / sum(r[4] for r in rows)
    print(f"rel_L2 trung binh toan downstream : {avg_rel:.4f}")
    print(f"cosine trung binh toan downstream  : {avg_cos:.4f}")

    if avg_rel < 0.1 and avg_cos > 0.9:
        print("\n-> Downstream GAN NHU KHONG DOI so voi Teacher (gia thuyet 'b'):")
        print("   Student van dung gan dung ham so cua Teacher. 41mm lech output rat co")
        print("   the den tu viec ham nay NHAY voi sai khac nho con sot lai trong feature")
        print("   (cosine dist 0.006-0.064 van du de bi khuech dai qua spatial_hypers ->")
        print("   pose_context_attn -> VPoser). Huong fix: lam feature khop SAT HON NUA,")
        print("   chu khong phai them KD cho downstream (vi downstream co doi dau dau).")
    elif avg_rel > 0.3 or avg_cos < 0.7:
        print("\n-> Downstream DA TROI DAT RO RET khoi Teacher (gia thuyet 'a'):")
        print("   Cac lop sau fusion (spatial_hypers/pose_context_attn/heads) da hoc ra")
        print("   trong so KHAC han so voi luc khoi tao tu Teacher. Day la nguyen nhan")
        print("   chinh gay lech output, chu khong phai do input feature con sai khac nho.")
        print("   Huong fix: them KD truc tiep cho feat_hyper / feat_pose / root_pose_6d /")
        print("   pose_latent (dung issue #5 da neu trong README) de ghim ca downstream,")
        print("   khong chi 3 diem noi truoc fusion.")
    else:
        print("\n-> Ket qua nam giua 2 thai cuc — xem bang [2] de biet CHINH XAC submodule")
        print("   nao troi nhieu nhat (vd: neu chi 'pose_context_attn'/'body_pose_head' troi")
        print("   con 'cam_head'/'shape_head' dung yen, nghia la rieng nhanh pose bi anh")
        print("   huong, khop voi viec loi tap trung o khop xa (co tay/co chan) ban da thay).")


if __name__ == '__main__':
    main()
