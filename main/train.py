import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # cho Apple Accelerate/vecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
from core.config import cfg, update_config

import warnings
warnings.filterwarnings("ignore")

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser(description='Train Pose2Mesh')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resume_training', action='store_true', help='Resume Training')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--exp_name', type=str, default='',
                    help='ten thu muc thi nghiem: experiment/<exp_name> (doc trong core/config.py). '
                         'Bo trong -> <ten file cfg>_<MM-DD_HH-MM-SS>')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
print('Seed = ', args.seed)

os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

import torch
# torch.backends.cudnn.enabled = False
import __init_path
import shutil
import random
import numpy as np
from funcs_utils import save_checkpoint, check_data_pararell, count_parameters
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

files_to_backup = [
    './lib/models/Multimodel.py',
    './lib/models/ARTS.py',
    './lib/models/teacher.py',
    './lib/models/hypergcn.py',
    './lib/models/common.py',
    './lib/core/base.py',
    './lib/core/config.py',
]
for file_path in files_to_backup:
    if os.path.exists(file_path):
        shutil.copyfile(src=file_path, dst=os.path.join(cfg.checkpoint_dir, os.path.basename(file_path)))

from core.base import Trainer, Tester, Teacher_Trainer, Teacher_Tester, Student_Trainer, Student_Tester

# Resume: lay last.pth.tar trong thu muc checkpoint cua chinh run nay (experiment/<exp_name>/checkpoint)
resume_path = ''
if args.resume_training:
    resume_path = os.path.join(cfg.checkpoint_dir, 'last.pth.tar')
    if not os.path.isfile(resume_path):
        sys.exit(f'[resume] Khong tim thay {resume_path}. Kiem tra lai --exp_name (phai trung ten thu muc run cu).')
    print(f'===> Resume tu: {resume_path}')

if cfg.MODEL.name == 'ARTS':
    trainer = Trainer(args, load_dir='./experiment/exp_04-26_09_16/checkpoint/best.pth.tar')
    tester = Tester(args)  # if not args.debug else None
elif cfg.MODEL.name == 'teacher':
    trainer = Teacher_Trainer(args, load_dir=resume_path)
    tester = Teacher_Tester(args)
elif cfg.MODEL.name == 'student':
    trainer = Student_Trainer(args, load_dir=resume_path)
    tester = Student_Tester(args)

print("===> Start training...")

for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
    trainer.train(epoch)
    trainer.lr_scheduler.step()

    tester.test(epoch, current_model=trainer.model)

    if len(trainer.error_history['joint']) > 0:
        is_best = tester.joint_error < min(trainer.error_history['joint']) or tester.surface_error < min(trainer.error_history['surface'])
    else:
        is_best = True

    trainer.error_history['surface'].append(tester.surface_error)
    trainer.error_history['joint'].append(tester.joint_error)

    ckpt = {
        'epoch': epoch,
        'model_state_dict': check_data_pararell(trainer.model.state_dict()),  # 
        'optim_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
        'train_log': trainer.loss_history,
        'test_log': trainer.error_history
    }
    if hasattr(trainer, 'awl'):
        ckpt['awl_state_dict'] = trainer.awl.state_dict()
    if getattr(trainer, 'feat_projector', None) is not None:   # None khi khong dung KD feat (scratch / legacy)
        ckpt['projector_state_dict'] = trainer.feat_projector.state_dict()
    save_checkpoint(ckpt, epoch, is_best)

print('Training Finished! All logs were saved in ', cfg.checkpoint_dir)

# ---- Test 1 LAN sau khi train: nap best.pth.tar (chon theo val) roi danh gia tren cfg.TEST.final_test_list ----
final_list = list(cfg.TEST.get('final_test_list', []) or [])
if final_list and cfg.MODEL.name in ('teacher', 'student'):
    best_path = os.path.join(cfg.checkpoint_dir, 'best.pth.tar')
    best_ck = torch.load(best_path, map_location='cpu', weights_only=False)
    net = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    missing, unexpected = net.load_state_dict(best_ck['model_state_dict'], strict=False)
    print(f"\n===> TEST CUOI tren {final_list} | {best_path} (epoch {best_ck.get('epoch')}) "
          f"| thieu {len(missing)} / thua {len(unexpected)} key")
    del best_ck
    cfg.DATASET.test_list = final_list
    if cfg.TEST.get('select_dataset', '') not in final_list:
        cfg.TEST.select_dataset = ''          # tranh assert trong _select_errors (select dat cho val)
    final_tester = Teacher_Tester(args) if cfg.MODEL.name == 'teacher' else Student_Tester(args)
    if hasattr(final_tester, 'eval_nohint'):
        final_tester.eval_nohint = True       # T-hint: test cuoi in ca CO va KHONG goi y
    with torch.no_grad():
        final_tester.test(epoch=0, current_model=trainer.model)
