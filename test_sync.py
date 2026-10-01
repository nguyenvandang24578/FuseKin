import sys
import os
import argparse
import torch

# Chèn đường dẫn module để python hiểu được code trong thư mục lib
sys.path.insert(0, os.path.abspath('lib'))
sys.path.insert(0, os.path.abspath('data_final'))

from core.config import cfg, update_config
from core.base import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    # Đường dẫn đến file config của bạn
    parser.add_argument('--cfg', type=str, default='config/train_student.yml')
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Nạp cấu hình
    update_config(args.cfg)
    
    # Ép batch_size nhỏ xuống để test cho nhanh
    cfg.TRAIN.batch_size = 10 
    
    # Test với toàn bộ 5 dataset mà bạn sẽ train chung
    dataset_names = ['Human36M', 'MuCo', 'MSCOCO', 'CrowdPose', 'PW3D']
    
    print(f"[*] Đang test bộ 5 dataset: {dataset_names}")
    print("[*] Gọi hàm get_dataloader...")
    
    try:
        dataset_list, batch_generator = get_dataloader(args, dataset_names, is_train=True)
    except Exception as e:
        print("[-] Lỗi khi tạo Dataloader:")
        import traceback
        traceback.print_exc()
        return

    print("\n[+] Đã khởi tạo thành công Dataloader (MultipleDatasets)!")
    print(f"[+] Tổng số batch: {len(batch_generator)}")
    print("[*] Đang thử fetch batch đầu tiên để kiểm tra đồng bộ dữ liệu (Collate)...")
    
    try:
        # Lấy 1 batch ra để kiểm tra
        inputs, targets, meta = next(iter(batch_generator))
        print("\n=======================================================")
        print("[+] Fetch dữ liệu THÀNH CÔNG! Các Dataset ĐÃ ĐỒNG BỘ!")
        print("=======================================================")
        
        print("\n--- CẤU TRÚC: INPUTS ---")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f" - {k}: shape {list(v.shape)}, dtype: {v.dtype}")
            else:
                print(f" - {k}: type {type(v)}")
                
        print("\n--- CẤU TRÚC: TARGETS ---")
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                print(f" - {k}: shape {list(v.shape)}, dtype: {v.dtype}")
            else:
                print(f" - {k}: type {type(v)}")
                
        print("\n--- CẤU TRÚC: META INFO ---")
        for k, v in meta.items():
            if isinstance(v, torch.Tensor):
                print(f" - {k}: shape {list(v.shape)}, dtype: {v.dtype}")
            else:
                print(f" - {k}: type {type(v)}")
                
    except Exception as e:
        print("\n[-] Lỗi xảy ra trong quá trình fetch dữ liệu (Do không đồng bộ Keys/Shapes):")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
