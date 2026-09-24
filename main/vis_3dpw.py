import os
import sys
import cv2
import torch
import numpy as np
import argparse
import __init_path
from core.config import cfg
from core.base import Tester

def save_obj(vertices, faces, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"[+] Đã lưu file lưới 3D (.obj): {filename}")

# Hàm convert cam copy y nguyên từ run_demo.py
def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    x, y, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:, 3]
    cx, cy, h = x + w / 2., y + h / 2., h
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

def main(args):
    print("Đang khởi tạo Tester và load dữ liệu 3DPW...")
    cfg.merge_from_file('config/train_init_mesh.yaml')
    cfg.TRAIN.wandb = False
    
    tester = Tester(args, load_dir=args.checkpoint)
    model = tester.model
    dataset = tester.val_datasets[0] # 3dpw
    loader = tester.val_loaders[0]
    
    faces = dataset.smpl.face

    model.eval()
    print("Đang chạy dự đoán trên 1 batch đầu tiên để vẽ Overlay theo chuẩn run_demo.py...")
    with torch.no_grad():
        for inputs, targets, meta in loader:
            model_inputs = {
                key: value.cuda() if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            outputs = model(model_inputs['img'], model_inputs['joints'], is_train=False)
            
            pred_mesh = outputs['smpl_mesh_cam'].detach().cpu().numpy()
            pred_cam = outputs['cam_param'].detach().cpu().numpy() # cam param (B, 3)
            
            # Lấy ảnh gốc
            input_img_tensor = inputs['img'][0]
            img_np = input_img_tensor.numpy().transpose(1, 2, 0)
            orig_img = (img_np * 255).astype(np.uint8)

            os.makedirs('output', exist_ok=True)
            
            idx = 0
            
            # ----- 1. RENDER OVERLAY THEO CÁCH CỦA RUN_DEMO.PY -----
            # Import Renderer y như run_demo.py
            sys.path.append('main') # Để tìm module renderer nếu nó ở trong main/
            try:
                from renderer import Renderer
            except ImportError:
                print("\n[LỖI]: Không tìm thấy file 'renderer.py' trong Repo của bạn! Chạy bằng code của run_demo.py sẽ bị crash ở đây.")
                print("Code demo cũ cần module renderer (thường chứa class PyRender), nhưng Repo của bạn đang thiếu file này.")
                sys.exit(1)

            orig_height, orig_width = orig_img.shape[:2]
            
            # Giả lập Bbox tương đương kích thước Crop (vì ảnh đã được crop sẵn về 256x256)
            # bbox theo chuẩn x, y, w, h
            bbox = np.array([[0, 0, orig_width, orig_height]])
            
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam[[idx]],
                bbox=bbox,
                img_width=orig_width,
                img_height=orig_height
            )

            color = (1.0, 0.6059142480254321, 0.5)
            
            # Setup renderer for visualization
            renderer = Renderer(faces, resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
            rendered_img = renderer.render(
                orig_img,
                pred_mesh[idx],
                cam=orig_cam[0],
                color=color,
                mesh_filename='output/mesh_bad_repo_pred.obj',
                rotate=False
            )
            
            # Vì renderer trả về RGB, ta cần convert sang BGR để OpenCV save ảnh đúng màu
            rendered_img_bgr = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('output/overlay_pred_rundemo.jpg', rendered_img_bgr)
            
            print("=========================================")
            print("Tất cả xong! Bạn hãy vào thư mục output/ để xem file overlay_pred_rundemo.jpg (vẽ bằng Renderer chuẩn)!")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file .pth.tar')
    parser.add_argument('--resume_training', action='store_true')
    args = parser.parse_args()
    main(args)
