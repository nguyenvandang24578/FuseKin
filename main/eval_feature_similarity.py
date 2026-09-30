import os, sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append('./lib')
sys.path.append('./')

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import copy

import __init_path
from core.config import cfg, update_config
from core.base import prepare_network
from funcs_utils import load_checkpoint
from lib.core.base import load_model_weights

def main(args):
    print("Đang cấu hình môi trường cho Student...")
    update_config('config/train_student.yml')
    cfg.TRAIN.wandb = False
    
    print("Đang khởi tạo DataLoader và Load Student Model...")
    # Lấy DataLoader và mô hình chưa bọc DataParallel
    val_loaders, val_datasets, student_model, _, _, _, _, _ = prepare_network(args, load_dir=args.student_checkpoint, is_train=False)
    
    loader = val_loaders[0]

    # Load Teacher Model
    print("Đang khởi tạo Teacher Model...")
    teacher_ckpt = cfg.MODEL.get('TEACHER', args.teacher_checkpoint)
    if not teacher_ckpt:
        raise ValueError("Cần cung cấp đường dẫn checkpoint của Teacher qua tham số --teacher_checkpoint hoặc trong config cfg.MODEL.TEACHER")
    
    # Clone kiến trúc từ student sang teacher
    teacher_model = copy.deepcopy(student_model)
    teacher_model.mode = 'teacher'
    # Nạp tạ (weights) cho Teacher
    load_model_weights(teacher_model, teacher_ckpt)
    
    # Bọc DataParallel cho cả 2 model để chạy đa luồng GPU giống lúc train/test
    student_model = torch.nn.DataParallel(student_model).cuda()
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    
    student_model.eval()
    teacher_model.eval()

    # === THÊM HOOK ĐỂ LẤY FEATURE TỪNG LAYER ===
    student_layer_feats = {}
    teacher_layer_feats = {}

    def get_hook(model_name, layer_idx):
        def hook(module, input, output):
            # output của RGBJointCrossTransformerBlock là (rgb_out, joint_out)
            rgb_out, joint_out = output
            if model_name == 'student':
                student_layer_feats[layer_idx] = {'rgb': rgb_out, 'joint': joint_out}
            else:
                teacher_layer_feats[layer_idx] = {'rgb': rgb_out, 'joint': joint_out}
        return hook

    # Gắn hook vào từng block của mô hình
    # Cấu trúc: model -> module (DataParallel) -> smpl_model (mạng Teacher bên trong ARTS) -> cfcer -> blocks
    num_blocks = len(student_model.module.smpl_model.cfcer.blocks)
    for i in range(num_blocks):
        student_model.module.smpl_model.cfcer.blocks[i].register_forward_hook(get_hook('student', i))
        teacher_model.module.smpl_model.cfcer.blocks[i].register_forward_hook(get_hook('teacher', i))

    total_mse = 0.0
    total_cosine = 0.0
    
    # Biến lưu trữ tổng sai số cho từng layer
    layer_total_mse = {i: {'rgb': 0.0, 'joint': 0.0} for i in range(num_blocks)}
    layer_total_cos = {i: {'rgb': 0.0, 'joint': 0.0} for i in range(num_blocks)}
    
    sample_count = 0
    
    print("=========================================")
    print("BẮT ĐẦU TÍNH ĐỘ TƯƠNG ĐỒNG FEATURE (FEATURE SIMILARITY)...")
    print("=========================================")
    
    with torch.no_grad():
        for step, (inputs, targets, meta) in enumerate(tqdm(loader, desc="Evaluating")):
            if step == 0:
                print("\n[DEBUG] Các keys có sẵn trong 1 batch của DataLoader (3DPW):")
                print(f"  - inputs keys : {list(inputs.keys())}")
                print(f"  - targets keys: {list(targets.keys())}")
                print(f"  - meta keys   : {list(meta.keys())}\n")
                
            input_image = inputs['img'].cuda().float()
            
            # --- 1. Đầu vào cho Student (Ảnh + Pose 2D) ---
            input_pose2d = inputs['joints'].cuda().float()
            # Xử lý mask 2D y như bên vis_3dpw.py
            if 'joints_mask' in inputs:
                mask = inputs['joints_mask'].cuda().float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                input_pose2d = torch.cat([input_pose2d[..., :2], mask], dim=-1)
                
            # --- 2. Đầu vào cho Teacher (Ảnh + Pose 3D Ground Truth) ---
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda()
            
            # --- 3. Trích xuất đặc trưng (Forward Pass) ---
            # Forward Student (có dùng thông tin 2D)
            s_out = student_model(input_image, input_pose2d, is_train=False)
            s_feat = s_out['feat_global'] # Kích thước (B, C)
            
            # Forward Teacher (có dùng thông tin 3D sạch)
            t_out = teacher_model(input_image, gt_fit_joint_cam, is_train=False)
            t_feat = t_out['feat_global'] # Kích thước (B, C)
            
            # --- 4. Tính toán Layer-wise Similarity (Qua Hooks) ---
            for i in range(num_blocks):
                s_rgb, s_joint = student_layer_feats[i]['rgb'], student_layer_feats[i]['joint']
                t_rgb, t_joint = teacher_layer_feats[i]['rgb'], teacher_layer_feats[i]['joint']
                
                # Đánh giá RGB tokens
                # flatten(1) đưa (B, H*W, C) về (B, H*W*C) để so sánh toàn cục từng ảnh
                layer_total_mse[i]['rgb'] += (F.mse_loss(s_rgb, t_rgb, reduction='sum').item() / s_rgb.shape[-1])
                layer_total_cos[i]['rgb'] += F.cosine_similarity(s_rgb.flatten(1), t_rgb.flatten(1), dim=-1).sum().item()
                
                # Đánh giá Joint tokens
                layer_total_mse[i]['joint'] += (F.mse_loss(s_joint, t_joint, reduction='sum').item() / s_joint.shape[-1])
                layer_total_cos[i]['joint'] += F.cosine_similarity(s_joint.flatten(1), t_joint.flatten(1), dim=-1).sum().item()
            
            
            # --- 4. Tính toán các độ đo tương đồng ---
            # MSE (Tính tổng MSE của cả batch)
            mse = F.mse_loss(s_feat, t_feat, reduction='sum').item()
            # Vì reduction='sum' đang cộng dồn cả B x C, ta chia lại cho C để ra MSE trung bình mỗi sample
            mse = mse / s_feat.shape[1] 
            total_mse += mse
            
            # Cosine Similarity (Kích thước B, cộng dồn lại)
            cos_sim = F.cosine_similarity(s_feat, t_feat, dim=-1).sum().item()
            total_cosine += cos_sim
            
            sample_count += input_image.size(0)

    avg_mse = total_mse / sample_count
    avg_cosine = total_cosine / sample_count
    
    print("\n=========================================")
    print(f"KẾT QUẢ ĐÁNH GIÁ GLOBAL FEATURE TRÊN {sample_count} SAMPLES:")
    print(f"- Lỗi MSE trung bình: {avg_mse:.6f}")
    print(f"- Độ tương đồng Cosine: {avg_cosine:.4f}")
    
    print("\n=========================================")
    print(f"KẾT QUẢ ĐÁNH GIÁ LAYER-WISE FEATURE (Từng lớp của Cross-Transformer):")
    for i in range(num_blocks):
        print(f"--- Layer {i+1} ---")
        avg_mse_rgb = layer_total_mse[i]['rgb'] / sample_count
        avg_cos_rgb = layer_total_cos[i]['rgb'] / sample_count
        avg_mse_joint = layer_total_mse[i]['joint'] / sample_count
        avg_cos_joint = layer_total_cos[i]['joint'] / sample_count
        
        print(f"  [RGB Tokens]   - MSE: {avg_mse_rgb:.6f} | Cosine Sim: {avg_cos_rgb:.4f}")
        print(f"  [Joint Tokens] - MSE: {avg_mse_joint:.6f} | Cosine Sim: {avg_cos_joint:.4f}")
        
    print("=========================================")
    print("Kết luận: Nếu Cosine Similarity > 0.9 ở các layer cuối, chứng tỏ Student đã chưng cất thành công!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_checkpoint', type=str, required=True, help='Đường dẫn tới file checkpoint của Student (vd: output/student_model.pth)')
    parser.add_argument('--teacher_checkpoint', type=str, default='', help='Đường dẫn tới file checkpoint của Teacher (nếu trống sẽ lấy trong config)')
    args = parser.parse_args()
    main(args)
