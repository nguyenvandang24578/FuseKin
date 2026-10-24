"""Visual attention joint->image cua hai checkpoint, chi dung trong inspector."""
import csv
import json
import numpy as np


def _entropy(p):
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=-1) / np.log(p.shape[-1])


def _js(p, q):
    # JS chia log(2) nam trong [0,1], khong nham voi KL co huong.
    m = (p + q) * 0.5
    lp, lq, lm = [np.log(np.maximum(x, 1e-12)) for x in (p, q, m)]
    value = 0.5 * ((p * (lp - lm)).sum(-1) + (q * (lq - lm)).sum(-1)) / np.log(2.0)
    return np.clip(value, 0.0, 1.0)


def extract_joint_image_attention(output, block_index):
    layers = output.get('feat_layers', [])
    if not layers or not -len(layers) <= block_index < len(layers):
        raise ValueError(f'attn_block={block_index} ngoai {len(layers)} block fusion')
    index = block_index % len(layers)
    tensor = layers[index]['attn_joint_to_img']
    if tensor.ndim != 4 or tensor.shape[2:] != (17, 64) or tensor.shape[1] < 1:
        raise ValueError('Visual nay can attention (B,heads,17,64), grid anh 8x8')
    heads = tensor.detach().float().cpu().numpy().astype(np.float64)
    if not np.isfinite(heads).all() or (heads < 0).any() or (heads.sum(-1) <= 0).any():
        raise RuntimeError('Attention khong phai phan bo huu han/khong am')
    heads /= heads.sum(-1, keepdims=True)
    mean = heads.mean(axis=1)
    return {'attn': mean.astype(np.float32),
            'head_entropy': _entropy(heads).mean(1).astype(np.float32),
            'head_diversity_js': _js(heads, mean[:, None]).mean(1).astype(np.float32),
            'block_index': np.full(len(mean), index, dtype=np.int64),
            'head_count': np.full(len(mean), heads.shape[1], dtype=np.int64)}


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_attention_diagnostics(args, out_dir, raw, rows, payloads, gt_pose, selected,
                               joint_names, epochs, parity, dataset_n, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = raw['teacher']['attn'].astype(np.float64)
    s = raw['kd_current']['attn'].astype(np.float64)
    t /= t.sum(-1, keepdims=True)
    s /= s.sum(-1, keepdims=True)
    tv = 0.5 * np.abs(t - s).sum(-1)
    js = _js(t, s)
    kl = (t * (np.log(np.maximum(t, 1e-12)) - np.log(np.maximum(s, 1e-12)))).sum(-1)
    ent_t, ent_s = _entropy(t), _entropy(s)
    teacher_template_tv = 0.5 * np.abs(t - t.mean(0, keepdims=True)).sum(-1)
    input_error = np.linalg.norm(raw['kd_current']['input_pose'] - gt_pose, axis=-1) * 1000
    teacher_error = np.linalg.norm(raw['teacher']['pose'] - gt_pose, axis=-1) * 1000
    student_error = np.linalg.norm(raw['kd_current']['pose'] - gt_pose, axis=-1) * 1000
    input_mean = input_error[:, 1:].mean(-1)
    joint_rows = []
    row_index = {row['sample']: i for i, row in enumerate(rows)}
    for i, row in enumerate(rows):
        row.update({'input_mpjpe16_mm': float(input_mean[i]), 'attn_tv_mean': float(tv[i].mean()),
                    'attn_js_mean': float(js[i].mean()), 'attn_kl_t_to_s_nats_mean': float(kl[i].mean()),
                    'teacher_mean_head_entropy': float(ent_t[i].mean()),
                    'student_mean_head_entropy': float(ent_s[i].mean()),
                    'teacher_template_tv_mean': float(teacher_template_tv[i].mean())})
        # Mask detector chi de tach nhom; thong ke van giu du 17 khop GT mesh.
        for j, name in enumerate(joint_names):
            joint_rows.append({'sample': row['sample'], 'joint': j, 'joint_name': name,
                               'detector_valid': int(raw['kd_current']['detector_mask'][i, j] > 0),
                               'input_error_mm': float(input_error[i, j]),
                               'teacher_mesh_error_mm': float(teacher_error[i, j]),
                               'student_mesh_error_mm': float(student_error[i, j]),
                               'tv': float(tv[i, j]), 'js': float(js[i, j]), 'kl_t_to_s_nats': float(kl[i, j]),
                               'teacher_mean_head_entropy': float(ent_t[i, j]),
                               'student_mean_head_entropy': float(ent_s[i, j]),
                               'teacher_head_entropy_mean': float(raw['teacher']['head_entropy'][i, j]),
                               'student_head_entropy_mean': float(raw['kd_current']['head_entropy'][i, j]),
                               'teacher_head_diversity_js': float(raw['teacher']['head_diversity_js'][i, j]),
                               'student_head_diversity_js': float(raw['kd_current']['head_diversity_js'][i, j]),
                               'teacher_template_tv': float(teacher_template_tv[i, j])})
    _write_csv(out_dir / 'joints.csv', joint_rows)
    per_joint = []
    for j, name in enumerate(joint_names):
        per_joint.append({'joint': j, 'joint_name': name, 'n': len(rows),
                          'tv_mean': float(tv[:, j].mean()), 'js_mean': float(js[:, j].mean()),
                          'teacher_entropy_mean': float(ent_t[:, j].mean()),
                          'student_entropy_mean': float(ent_s[:, j].mean()),
                          'teacher_template_tv_mean': float(teacher_template_tv[:, j].mean()),
                          'input_error_mm_mean': float(input_error[:, j].mean())})
    _write_csv(out_dir / 'joint_summary.csv', per_joint)
    groups = []
    order = np.argsort(input_mean, kind='mergesort')
    for q, ids in enumerate(np.array_split(order, 4), 1):
        if not len(ids):
            continue
        for i in ids:
            rows[i]['input_error_quartile'] = q
        groups.append({'group': f'Q{q}_input_error', 'n': len(ids),
                       'input_mpjpe16_mm': float(input_mean[ids].mean()),
                       'teacher_mesh_mpjpe17_mm': float(teacher_error[ids].mean()),
                       'student_mesh_mpjpe17_mm': float(student_error[ids].mean()),
                       'tv_mean': float(tv[ids].mean()), 'js_mean': float(js[ids].mean()),
                       'teacher_entropy_mean': float(ent_t[ids].mean()),
                       'student_entropy_mean': float(ent_s[ids].mean())})
    _write_csv(out_dir / 'samples.csv', rows)
    _write_csv(out_dir / 'groups.csv', groups)
    archive = {'sample_ids': np.asarray(selected), 'joint_names': np.asarray(joint_names),
               'gt_pose_m': gt_pose, 'tv': tv, 'js': js, 'kl_t_to_s_nats': kl}
    for name, values in raw.items():
        for key, value in values.items():
            archive[f'{name}_{key}_m' if key in ('pose', 'input_pose') else f'{name}_{key}'] = value
    centers = (np.arange(8) + 0.5) / 8 * 2 - 1
    xx, yy = np.meshgrid(centers, centers)
    archive['image_cell_centers_xy'] = np.stack([xx, yy], -1).reshape(64, 2)
    np.savez_compressed(out_dir / 'attention.npz', **archive)
    heat_max = max(float(t.max()), float(s.max()), 1.0 / 64)
    delta_max = max(float(np.abs(s - t).max()), 1e-6)

    def matrix(ax, value, title, vmax, cmap='magma'):
        im = ax.imshow(value, aspect='auto', interpolation='nearest', vmin=0, vmax=vmax, cmap=cmap)
        ax.set_yticks(np.arange(17))
        ax.set_yticklabels(joint_names, fontsize=7)
        ax.set_xlabel('Image cell index: row * 8 + column')
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    matrix(axes[0, 0], t.mean(0), 'Teacher: mean attention across samples', heat_max)
    matrix(axes[0, 1], s.mean(0), 'Student: mean attention across samples', heat_max)
    matrix(axes[0, 2], np.abs(s - t).mean(0), 'Mean ABS per-sample Student - Teacher', delta_max)
    axes[1, 0].bar(np.arange(17), tv.mean(0))
    axes[1, 0].set_xticks(np.arange(17))
    axes[1, 0].set_xticklabels(joint_names, rotation=90, fontsize=7)
    axes[1, 0].set(ylabel='TV [0,1]', title='Mean TV per joint: 0 = same mean-head distribution')
    axes[1, 1].bar(np.arange(17) - 0.2, ent_t.mean(0), width=0.4, label='Teacher')
    axes[1, 1].bar(np.arange(17) + 0.2, ent_s.mean(0), width=0.4, label='Student')
    axes[1, 1].set_xticks(np.arange(17))
    axes[1, 1].set_xticklabels(joint_names, rotation=90, fontsize=7)
    axes[1, 1].set(ylabel='Normalized entropy', title='1 = uniform over 64 cells; not pose confidence')
    axes[1, 1].legend()
    axes[1, 2].scatter(input_mean, tv.mean(-1), s=12, alpha=0.5)
    axes[1, 2].set(xlabel='Input pose MPJPE16 (mm)', ylabel='Mean TV over 17 joints', title='Same samples, no grouping by model output error')
    fig.suptitle(f'Joint-to-image ATTENTION, block index {args.attn_block} | {args.set} n={len(rows)}\n'
                 'Mean over heads; identical spatial cells. This is NOT a 17x17 feature relation.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / 'attention_overview.png', dpi=150)
    plt.close(fig)
    visual_joints = list(range(17)) if args.attn_joints == ['all'] else [joint_names.index(name) for name in dict.fromkeys(args.attn_joints)]
    vis_dir = out_dir / 'vis_attention'
    vis_dir.mkdir()
    for idx, payload in sorted(payloads.items()):
        sample = row_index[idx]
        crop = np.clip(payload['image'].transpose(1, 2, 0), 0, 1)
        height, width = crop.shape[:2]
        points = (payload['p2'][:, :2] + 1) * 0.5 * np.array([width, height])
        fig, axes = plt.subplots(len(visual_joints), 4, figsize=(14, 3.3 * len(visual_joints)), squeeze=False)
        for r, joint in enumerate(visual_joints):
            valid = payload['mask'][joint] > 0
            name = joint_names[joint]
            for col in range(3):
                ax = axes[r, col]
                ax.imshow(crop, extent=(0, width, height, 0))
                if valid:
                    ax.scatter(*points[joint], s=60, facecolors='none', edgecolors='cyan', linewidths=1.5)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.axis('off')
            axes[r, 0].set_title(f'{name}: detector circle' + ('' if valid else ' MISSING') +
                                  f'\nInput error={input_error[sample, joint]:.1f} mm', fontsize=9)
            for col, attn, label, ent in ((1, t, 'Teacher', ent_t), (2, s, 'Student', ent_s)):
                im = axes[r, col].imshow(attn[sample, joint].reshape(8, 8), extent=(0, width, height, 0),
                                         interpolation='nearest', vmin=0, vmax=heat_max, cmap='magma', alpha=0.55)
                axes[r, col].set_title(f'{label}: H={ent[sample, joint]:.3f}, max={attn[sample, joint].max():.4f}', fontsize=9)
                plt.colorbar(im, ax=axes[r, col], fraction=0.04, pad=0.01)
            delta = (s[sample, joint] - t[sample, joint]).reshape(8, 8)
            im = axes[r, 3].imshow(delta, vmin=-delta_max, vmax=delta_max, cmap='coolwarm', interpolation='nearest')
            axes[r, 3].set_title(f'Student - Teacher\nTV={tv[sample, joint]:.3f}, JS={js[sample, joint]:.3f}', fontsize=9)
            axes[r, 3].set_xticks([])
            axes[r, 3].set_yticks([])
            plt.colorbar(im, ax=axes[r, 3], fraction=0.04, pad=0.01)
        fig.suptitle(f'#{idx} ann={payload["row"]["ann_id"]} | block={args.attn_block} | mean heads\n'
                     f'{payload["row"]["img_path"]}\nShared color scales across ALL samples. Cyan = detector, not GT. Attention need not peak on the joint.', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(vis_dir / f'{idx:06d}_attention.png', dpi=130)
        plt.close(fig)
    summary = {'mode': 'attention', 'args': vars(args), 'epochs': epochs, 'n': len(rows), 'dataset_n': dataset_n,
               'subset': len(rows) < dataset_n, 'selected_indices': selected, 'input_parity': parity,
               'grid_hw': [8, 8], 'head_aggregation': 'mean (no assumption of matching head indices)',
               'block_indices': {name: int(value['block_index'][0]) for name, value in raw.items()},
               'head_counts': {name: int(value['head_count'][0]) for name, value in raw.items()},
               'tv_mean': float(tv.mean()), 'tv_median': float(np.median(tv)), 'tv_p90': float(np.percentile(tv, 90)),
               'js_mean': float(js.mean()), 'js_p90': float(np.percentile(js, 90)),
               'kl_t_to_s_nats_mean': float(kl.mean()), 'tv_mean_no_pelvis': float(tv[:, 1:].mean()),
               'teacher_entropy_mean': float(ent_t.mean()), 'student_entropy_mean': float(ent_s.mean()),
               'teacher_head_diversity_js_mean': float(raw['teacher']['head_diversity_js'].mean()),
               'student_head_diversity_js_mean': float(raw['kd_current']['head_diversity_js'].mean()),
               'teacher_template_tv_mean': float(teacher_template_tv.mean()),
               'teacher_mesh_mpjpe17_mm': float(teacher_error.mean()), 'student_mesh_mpjpe17_mm': float(student_error.mean())}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    log(f'ATTENTION mean heads: TV mean={tv.mean():.6f}, median={np.median(tv):.6f}, p90={np.percentile(tv, 90):.6f} | JS mean={js.mean():.6f}')
    log(f'Entropy [0,1], 1=uniform: Teacher={ent_t.mean():.6f}, Student={ent_s.mean():.6f} | Teacher-vs-own-template TV={teacher_template_tv.mean():.6f}')
    for group in groups:
        log(f'{group["group"]} n={group["n"]} input={group["input_mpjpe16_mm"]:.2f}mm | TV={group["tv_mean"]:.6f} JS={group["js_mean"]:.6f}')
    log('TV/JS=0 la giong nhau; khong tu dat nguong cho nhan KD/khong KD. Khac biet chua chung minh Teacher dung hon.')
    log('Mean heads co the che khac biet tung head; xem head diversity trong joints.csv. Entropy khong phai confidence pose.')
    if len(rows) < 2:
        log('CHI 1 MAU: Teacher template TV=0 theo dinh nghia, khong du de doc do bien thien giua mau.')
    log('Teacher-vs-own-template TV can >=2 mau; gan0 voi nhieu mau la map it thay doi, khong rieng thong tin anh.')
    log('Chi visual/diagnostic: khong doi attention forward, khong ep peak tai detector, khong train/chon checkpoint.')
    log(f'Ket qua: {out_dir} | attention_overview.png, vis_attention/, attention.npz, joints.csv, groups.csv, summary.json')
