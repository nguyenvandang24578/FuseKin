import os, sys
os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # cho Apple Accelerate/vecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import cv2
import torch
import numpy as np
import argparse
import __init_path
from core.config import cfg, update_config
from core.base import Teacher_Tester, Student_Tester

def save_obj(vertices, faces, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[+] Đã lưu file lưới 3D (.obj): {filename}")

import pyrender
import trimesh
import math

def render_perspective(img, vertices_3d, faces, focal, princpt):
    # vertices_3d: (6890, 3) in absolute camera coordinates (mesh_cam_render)
    mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces, process=False)
    
    # SMPL/OpenCV camera: +Z is forward, +Y is down.
    # PyRender camera: -Z is forward, +Y is up.
    # => Xoay mesh 180 độ quanh trục X
    rot = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 0.7, 0.6, 1.0)
    )
    mesh_pr = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.5, 0.5, 0.5))
    scene.add(mesh_pr, 'mesh')
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    scene.add(light, pose=np.eye(4))
    
    # Dùng đúng Perspective Camera của model
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1], znear=0.1, zfar=1000.0)
    scene.add(camera, pose=np.eye(4))
    
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0])
    rgb, depth = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    r.delete()
    
    valid_mask = (depth > 0)[:, :, np.newaxis]
    output_img = rgb[:, :, :3] * valid_mask + (1 - valid_mask) * img
    return output_img.astype(np.uint8)

def main(args):
    print(f"Đang khởi tạo Tester và load dữ liệu 3DPW ({args.mode} mode)...")
    if args.mode == 'student':
        update_config('config/train_student.yml')
        cfg.TRAIN.wandb = False
        tester = Student_Tester(args, load_dir=args.checkpoint)
    elif args.mode == 'arts':
        update_config('./config/train_init_mesh.yaml')
        cfg.TRAIN.wandb = False
        # Student_Tester và Teacher_Tester code giống hệt nhau, chỉ quan trọng config
        tester = Student_Tester(args, load_dir=args.checkpoint)
    else:
        update_config('config/train_teacher.yml')
        cfg.TRAIN.wandb = False
        tester = Teacher_Tester(args, load_dir=args.checkpoint)
        
    model = tester.model
    dataset = tester.val_datasets[0] # 3dpw
    loader = tester.val_loaders[0]
    
    faces = dataset.smpl.face

    model.eval()
    print("Đang chạy dự đoán trên 1 batch đầu tiên để vẽ Overlay với PyRender Perspective...")
    with torch.no_grad():
        # ==========================================================
        # ĐÁNH GIÁ TRUNG BÌNH TOÀN BỘ TẬP TEST (FULL DATASET)
        # ==========================================================
        print("Đang tính trung bình MPJPE trên TOÀN BỘ tập Test để xác thực con số 266mm (mất vài phút)...")
        from utils.transforms import rigid_align
        from tqdm import tqdm
        h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        
        total_mpjpe = 0
        total_pa_mpjpe = 0
        sample_count = 0
        
        with torch.no_grad():
            for inputs, targets, meta in tqdm(loader, desc="Evaluating Test Set"):
                model_inputs = {
                    key: value.cuda() if torch.is_tensor(value) else value
                    for key, value in inputs.items()
                }
                
                if cfg.MODEL.name == 'arts' or cfg.MODEL.name == 'teacher':
                    target_mesh_tensor = targets['smpl_mesh_cam'].cuda().float()
                    h36m_regressor = torch.as_tensor(
                        dataset.h36m_joint_regressor,
                        device='cuda',
                        dtype=target_mesh_tensor.dtype,
                    )
                    teacher_gt_joints = torch.matmul(
                        h36m_regressor.unsqueeze(0).expand(target_mesh_tensor.shape[0], -1, -1),
                        target_mesh_tensor,
                    )
                    input_pose = teacher_gt_joints - teacher_gt_joints[:, 0:1, :]
                else:
                    input_pose = model_inputs['joints']
                    
                outputs = model(model_inputs['img'], input_pose, is_train=False, use_gt_3d = True)
                
                pred_mesh = outputs['smpl_mesh_cam'].detach().cpu().numpy()
                target_mesh = targets['smpl_mesh_cam'].detach().cpu().numpy()
                
                for idx in range(pred_mesh.shape[0]):
                    
                    gt_h36m_3d = np.dot(dataset.h36m_joint_regressor, target_mesh[idx])
                    pred_h36m_3d = np.dot(dataset.h36m_joint_regressor, pred_mesh[idx])
                    
                    gt_root_relative = gt_h36m_3d - gt_h36m_3d[0:1, :]
                    pred_root_relative = pred_h36m_3d - pred_h36m_3d[0:1, :]
                    
                    gt_eval_3d = gt_root_relative[h36m_eval_joint, :]
                    pred_eval_3d = pred_root_relative[h36m_eval_joint, :]
                    
                    mpjpe = np.sqrt(np.sum((pred_eval_3d - gt_eval_3d) ** 2, axis=1)).mean() * 1000
                    pred_aligned_3d = rigid_align(pred_eval_3d, gt_eval_3d)
                    pa_mpjpe = np.sqrt(np.sum((pred_aligned_3d - gt_eval_3d) ** 2, axis=1)).mean() * 1000
                    
                    total_mpjpe += mpjpe
                    total_pa_mpjpe += pa_mpjpe
                    sample_count += 1

        print("=========================================")
        print(f"KẾT QUẢ TRUNG BÌNH TRÊN {sample_count} SAMPLES TẦM NHÌN CHUNG:")
        print(f"- Average MPJPE:    {total_mpjpe / sample_count:.2f} mm")
        print(f"- Average PA-MPJPE: {total_pa_mpjpe / sample_count:.2f} mm")
        print("=========================================")
        
        # Lặp ngẫu nhiên 100 samples để render hình cho đa dạng
        import random
        total_batches = len(loader)
        # Chọn ngẫu nhiên 100 batch index
        random_indices = set(random.sample(range(total_batches), min(100, total_batches)))
        
        count = 0
        os.makedirs('output_100', exist_ok=True)
        for batch_idx, (inputs, targets, meta) in enumerate(loader):
            if batch_idx not in random_indices:
                continue
                
            print(f"Đang render sample {count+1}/100 (từ batch thứ {batch_idx})...")
            model_inputs = {
                key: value.cuda() if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            if args.mode == 'arts' or args.mode == 'teacher':
                target_mesh_tensor = targets['smpl_mesh_cam'].cuda().float()
                h36m_regressor = torch.as_tensor(
                    dataset.h36m_joint_regressor,
                    device='cuda',
                    dtype=target_mesh_tensor.dtype,
                )
                teacher_gt_joints = torch.matmul(
                    h36m_regressor.unsqueeze(0).expand(target_mesh_tensor.shape[0], -1, -1),
                    target_mesh_tensor,
                )
                input_pose = teacher_gt_joints - teacher_gt_joints[:, 0:1, :]
            else:
                input_pose = model_inputs['joints'].float()
                if 'joints_mask' in model_inputs:
                    mask = model_inputs['joints_mask'].float()
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(-1)
                    # Nối mask (chiều 3) vào joints (x, y) để pose_lifter biết joint nào bị che
                    input_pose = torch.cat([input_pose[..., :2], mask], dim=-1)
                
            outputs = model(model_inputs['img'], input_pose, is_train=False, use_gt_3d = True)
            
            pred_mesh = outputs['smpl_mesh_cam'].detach().cpu().numpy()
            
            # Lấy ảnh gốc
            input_img_tensor = inputs['img'][0]
            img_np = input_img_tensor.numpy().transpose(1, 2, 0)
            orig_img = (img_np * 255).astype(np.uint8)
    
            idx = 0
            
            # ==========================================================
            # TỰ TÍNH TOÁN TRUE TRANSLATION CHO ẢNH CROP 256x256
            # ==========================================================
            data_idx = meta['idx'][0].item()
            raw_data = dataset.datalist[data_idx]
            
            # Lấy thông số camera gốc
            real_focal = raw_data['cam_param']['focal']
            real_princpt = raw_data['cam_param']['princpt']
            
            # Lấy tọa độ điểm gốc (Root - Pelvis) tuyệt đối của GT trong ảnh gốc
            _, smpl_joint_cam_abs = dataset.get_smpl_coord(raw_data['smpl_param'])
            orig_root = smpl_joint_cam_abs[dataset.root_joint_idx] # [X_orig, Y_orig, Z_orig]
            
            # Chiếu Root GT 3D xuống tọa độ pixel 2D trên ảnh GỐC
            x_orig = orig_root[0] / orig_root[2] * real_focal[0] + real_princpt[0]
            y_orig = orig_root[1] / orig_root[2] * real_focal[1] + real_princpt[1]
            
            # Dùng ma trận affine (img2bb_trans) để chuyển tọa độ 2D gốc sang hệ pixel 2D của ảnh CROP 256x256
            img2bb_trans = meta['img2bb_trans'][0].cpu().numpy()
            pt_crop = np.dot(img2bb_trans, np.array([x_orig, y_orig, 1.0]))
            x_crop, y_crop = pt_crop[0], pt_crop[1]
            
            # Tính lại Z_crop (tz) để giữ nguyên tỷ lệ (Scale) tương đối khi tiêu cự thay đổi thành f=1500
            s_x = img2bb_trans[0, 0] # Tỷ lệ thu phóng khi cắt và resize bbox về 256x256
            tz_true = orig_root[2] * cfg.DATASET.focal[0] / (real_focal[0] * s_x)
            
            # Dùng x_crop, y_crop, tz_true và camera ảo (1500, 128) để tính ngược ra X_crop, Y_crop 3D
            tx_true = (x_crop - cfg.DATASET.princpt[0]) * tz_true / cfg.DATASET.focal[0]
            ty_true = (y_crop - cfg.DATASET.princpt[1]) * tz_true / cfg.DATASET.focal[1]
            
            true_crop_trans = np.array([tx_true, ty_true, tz_true])
            
            # Ghép True Translation vào lưới GT (Nhớ trừ đi orig_root để đưa về Root-Relative trước!)
            target_mesh_root = targets['smpl_mesh_cam'].detach().cpu().numpy()
            target_verts = target_mesh_root[idx] - orig_root + true_crop_trans
            
            # Tính pred_verts tương tự GT: chuyển mesh về root-relative rồi cộng true translation
            pred_h36m = np.dot(dataset.h36m_joint_regressor, pred_mesh[idx])
            pred_root = pred_h36m[0]
            pred_verts = pred_mesh[idx] - pred_root + true_crop_trans
            # ==========================================================
            
            # ----- RENDER OVERLAY PRED -----
            rendered_pred = render_perspective(
                orig_img, 
                pred_verts, 
                faces, 
                cfg.DATASET.focal, 
                cfg.DATASET.princpt
            )
            
            # ----- RENDER OVERLAY GT -----
            rendered_gt = render_perspective(
                orig_img, 
                target_verts, 
                faces, 
                cfg.DATASET.focal, 
                cfg.DATASET.princpt
            )
            
            # Convert RGB sang BGR để vẽ chữ và lưu bằng OpenCV
            rendered_pred_bgr = cv2.cvtColor(rendered_pred, cv2.COLOR_RGB2BGR)
            rendered_gt_bgr = cv2.cvtColor(rendered_gt, cv2.COLOR_RGB2BGR)
            
            # Thêm Text ghi chú
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rendered_pred_bgr, "Prediction (Bad Repo)", (10, 20), font, 0.5, (0, 0, 255), 2)
            cv2.putText(rendered_gt_bgr, "Ground Truth", (10, 20), font, 0.5, (0, 255, 0), 2)
            
            # Ghép 2 ảnh theo chiều ngang
            combined_img = np.concatenate((rendered_pred_bgr, rendered_gt_bgr), axis=1)
            
            cv2.imwrite(f'output_100/{count:03d}_overlay_compare.jpg', combined_img)
            
            save_obj(pred_mesh[idx], faces, 'output/mesh_bad_repo_pred.obj')
            save_obj(target_mesh_root[idx], faces, 'output/mesh_3dpw_gt.obj')
            
            # ==========================================================
            # PHẦN 2: CHIẾU GT JOINT 2D VÀ OPENPOSE LÊN ẢNH GỐC ĐỂ KIỂM CHỨNG
            # ==========================================================
            data_idx = meta['idx'][0].item()
            raw_data = dataset.datalist[data_idx]
            
            # 1. Load ảnh gốc (Original Image)
            orig_full_img_path = raw_data['img_path']
            orig_full_img = cv2.imread(orig_full_img_path)
            
            # 2. Lấy Absolute GT 3D Joints từ smpl_param
            _, smpl_joint_cam_abs = dataset.get_smpl_coord(raw_data['smpl_param'])
            
            # 3. Project GT 3D Joints lên ảnh gốc bằng Real cam_param
            real_focal = raw_data['cam_param']['focal']
            real_princpt = raw_data['cam_param']['princpt']
            
            gt_2d_x = smpl_joint_cam_abs[:, 0] / smpl_joint_cam_abs[:, 2] * real_focal[0] + real_princpt[0]
            gt_2d_y = smpl_joint_cam_abs[:, 1] / smpl_joint_cam_abs[:, 2] * real_focal[1] + real_princpt[1]
            
            # 4. Lấy OpenPose
            openpose_2d = raw_data['openpose'] # (N, 3) (x, y, conf)
            
            # 5. Vẽ Bounding Box mục tiêu (Màu xanh lá)
            bbox = raw_data['tight_bbox'] # x, y, w, h
            cv2.rectangle(orig_full_img, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
            cv2.putText(orig_full_img, "Target BBox", (int(bbox[0]), int(bbox[1]-10)), font, 0.7, (0, 255, 0), 2)
            
            # 6. Vẽ OpenPose (Màu Xanh Dương - Blue)
            for pt in openpose_2d:
                if pt[2] > 0.05: # Confidence
                    cv2.circle(orig_full_img, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)
            cv2.putText(orig_full_img, "OpenPose (Blue)", (20, 40), font, 0.8, (255, 0, 0), 2)
            
            # 7. Vẽ GT 2D Projected (Màu Đỏ - Red)
            for x, y in zip(gt_2d_x, gt_2d_y):
                cv2.circle(orig_full_img, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.putText(orig_full_img, "GT Projected (Red)", (20, 80), font, 0.8, (0, 0, 255), 2)
            
            cv2.imwrite(f'output_100/{count:03d}_gt_projection_points.jpg', orig_full_img)
            
            # ==========================================================
            # PHẦN 3: RENDER GT MESH LÊN ẢNH GỐC (DÙNG TIÊU CỰ THẬT)
            # ==========================================================
            print("Đang render GT Mesh lên ảnh gốc...")
            
            # Load lại ảnh gốc (sạch, chưa vẽ râu ria)
            orig_clean_img = cv2.imread(orig_full_img_path)
            
            # Lấy Mesh thay vì Joint
            smpl_mesh_cam_abs, _ = dataset.get_smpl_coord(raw_data['smpl_param'])
            
            # Bắt buộc ảnh orig_clean_img phải là RGB trước khi render (cv2 đọc là BGR)
            orig_clean_img_rgb = cv2.cvtColor(orig_clean_img, cv2.COLOR_BGR2RGB)
            
            # Render lưới GT tuyệt đối lên ảnh gốc
            rendered_gt_orig = render_perspective(
                orig_clean_img_rgb, 
                smpl_mesh_cam_abs, 
                faces, 
                real_focal, 
                real_princpt
            )
            
            # Save
            rendered_gt_orig_bgr = cv2.cvtColor(rendered_gt_orig, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'output_100/{count:03d}_gt_mesh_on_original.jpg', rendered_gt_orig_bgr)
            
            # ==========================================================
            # PHẦN 4: VẼ 17 KHỚP H36M (SKELETON) CỦA PRED VÀ GT ĐỂ THẤY LỖI MPJPE
            # ==========================================================
            print("Đang vẽ Skeleton so sánh 17 khớp H36M...")
            
            # Lấy ma trận Regressor H36M từ dataset
            h36m_regressor = dataset.h36m_joint_regressor
            
            # Sinh 17 khớp 3D bằng cách nhân Regressor với Mesh 3D (Đã có True Translation)
            gt_h36m_3d = np.dot(h36m_regressor, target_verts)
            pred_h36m_3d = np.dot(h36m_regressor, pred_verts)
            
            # Chiếu 17 khớp 3D xuống ảnh 2D Crop (dùng camera ảo 1500, 128)
            def project_3d_to_2d(joints_3d, f, c):
                x = joints_3d[:, 0] / joints_3d[:, 2] * f[0] + c[0]
                y = joints_3d[:, 1] / joints_3d[:, 2] * f[1] + c[1]
                return np.stack([x, y], axis=1)
                
            gt_h36m_2d = project_3d_to_2d(gt_h36m_3d, cfg.DATASET.focal, cfg.DATASET.princpt)
            pred_h36m_2d = project_3d_to_2d(pred_h36m_3d, cfg.DATASET.focal, cfg.DATASET.princpt)
            
            # Định nghĩa các đoạn xương (Bones) nối 17 khớp H36M
            h36m_skeleton = [
                (0, 1), (1, 2), (2, 3),        # Chân phải
                (0, 4), (4, 5), (5, 6),        # Chân trái
                (0, 7), (7, 8), (8, 9), (9, 10), # Cột sống lên đầu
                (8, 14), (14, 15), (15, 16),   # Tay phải
                (8, 11), (11, 12), (12, 13)    # Tay trái
            ]
            
            # Tạo ảnh canvas (Lấy ảnh crop gốc 256x256 chuyển sang BGR)
            skeleton_img = cv2.cvtColor(orig_img.copy(), cv2.COLOR_RGB2BGR)
            
            # Hàm vẽ Skeleton
            def draw_skeleton(img, joints_2d, color):
                for pt in joints_2d:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)
                for bone in h36m_skeleton:
                    pt1 = (int(joints_2d[bone[0]][0]), int(joints_2d[bone[0]][1]))
                    pt2 = (int(joints_2d[bone[1]][0]), int(joints_2d[bone[1]][1]))
                    cv2.line(img, pt1, pt2, color, 2)
            
            # Vẽ GT bằng màu Xanh lá (Green), Pred bằng màu Đỏ (Red)
            draw_skeleton(skeleton_img, gt_h36m_2d, (0, 255, 0)) # Green
            draw_skeleton(skeleton_img, pred_h36m_2d, (0, 0, 255)) # Red
            
            cv2.putText(skeleton_img, "GT (Green) vs Pred (Red)", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.imwrite(f'output_100/{count:03d}_skeleton_compare.jpg', skeleton_img)
            
            # ==========================================================
            # PHẦN 5: VẼ 3D KHÔNG GIAN (SKELETON) - TOP VIEW & SIDE VIEW
            # ==========================================================
            print("Đang vẽ không gian 3D (Top View & Side View) bằng Matplotlib...")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 7))
            
            def plot_3d_skeleton(ax, joints_3d, color, label):
                # Vẽ các điểm khớp
                ax.scatter(joints_3d[:, 0], joints_3d[:, 2], -joints_3d[:, 1], c=color, s=20, label=label)
                # Vẽ xương
                for bone in h36m_skeleton:
                    pt1 = joints_3d[bone[0]]
                    pt2 = joints_3d[bone[1]]
                    # Hệ trục Matplotlib 3D: x = X, y = Z (depth), z = -Y (up)
                    ax.plot([pt1[0], pt2[0]], [pt1[2], pt2[2]], [-pt1[1], -pt2[1]], c=color, linewidth=2)
            
            # 1. Góc nhìn từ trên xuống (Top View) - Thấy rõ sai số Depth (Z)
            ax1 = fig.add_subplot(121, projection='3d')
            plot_3d_skeleton(ax1, gt_h36m_3d, 'g', 'Ground Truth')
            plot_3d_skeleton(ax1, pred_h36m_3d, 'r', 'Prediction')
            ax1.set_title('Top View (Nhin Tu Tren Xuong)')
            ax1.set_xlabel('X (Ngang)')
            ax1.set_ylabel('Z (Chieu Sau)')
            ax1.set_zlabel('Y (Doc)')
            ax1.view_init(elev=90, azim=-90)
            ax1.legend()
            
            # 2. Góc nhìn ngang (Side View)
            ax2 = fig.add_subplot(122, projection='3d')
            plot_3d_skeleton(ax2, gt_h36m_3d, 'g', 'Ground Truth')
            plot_3d_skeleton(ax2, pred_h36m_3d, 'r', 'Prediction')
            ax2.set_title('Side View (Nhin Ngang)')
            ax2.set_xlabel('X (Ngang)')
            ax2.set_ylabel('Z (Chieu Sau)')
            ax2.set_zlabel('Y (Doc)')
            ax2.view_init(elev=0, azim=0)
            
            plt.tight_layout()
            plt.savefig(f'output_100/{count:03d}_skeleton_3d_space.jpg', dpi=150)
            plt.close()
            
            # ==========================================================
            # PHẦN 6: TÍNH TOÁN ĐIỂM SỐ MPJPE TRỰC TIẾP TRÊN KHUNG HÌNH NÀY
            # ==========================================================
            print("Đang tính điểm MPJPE trực tiếp trên sample này...")
            from utils.transforms import rigid_align
            
            # 14 khớp dùng để evaluate trong H36M
            h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
            
            # Đưa về Root-relative
            gt_root_relative = gt_h36m_3d - gt_h36m_3d[0:1, :]
            pred_root_relative = pred_h36m_3d - pred_h36m_3d[0:1, :]
            
            # Lấy 14 khớp evaluate
            gt_eval_3d = gt_root_relative[h36m_eval_joint, :]
            pred_eval_3d = pred_root_relative[h36m_eval_joint, :]
            
            # Tính MPJPE (Euclidean distance trung bình) -> đổi ra milimet
            mpjpe = np.sqrt(np.sum((pred_eval_3d - gt_eval_3d) ** 2, axis=1)).mean() * 1000
            
            # Tính PA-MPJPE (Procrustes Alignment)
            pred_aligned_3d = rigid_align(pred_eval_3d, gt_eval_3d)
            pa_mpjpe = np.sqrt(np.sum((pred_aligned_3d - gt_eval_3d) ** 2, axis=1)).mean() * 1000
            
            print("=========================================")
            print("ĐIỂM SỐ CỦA RIÊNG KHUNG HÌNH NÀY:")
            print(f"- MPJPE:      {mpjpe:.2f} mm")
            print(f"- PA-MPJPE:   {pa_mpjpe:.2f} mm")
            
            print("=========================================")
            
            count += 1
            if count >= 100:
                break
            
        print("Đã vẽ xong 100 samples vào thư mục output_100/!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file .pth.tar')
    parser.add_argument('--mode', type=str, default='teacher', choices=['teacher', 'student', 'arts'], help='Chọn mô hình test: teacher (nhận GT 3D) hoặc student (nhận output từ pose_lifter)')
    parser.add_argument('--resume_training', action='store_true')
    args = parser.parse_args()
    main(args)
