# FuseKin — bối cảnh kỹ thuật hiện tại (FuseKin-v3)

Cập nhật **2026-11-03**. Nội dung kỹ thuật đối chiếu với code/YAML local; đây không phải xác nhận trạng thái môi trường hoặc file trên server. Kết quả và tiến độ nằm trong [experiment.md](experiment.md), lệnh thường dùng trong [README.md](README.md).

## 1. Phạm vi

- Bài toán **image-based**, một ảnh, `seqlen: 1`.
- Hai phần train riêng: lifter 2D→3D (DiffLifter); Teacher/Student v3 kết hợp ảnh + pose để dự đoán mesh SMPL.
- Phiên bản chốt: **FuseKin-v3**. Giao thức test chốt (2026-11-03): OpenPose cho 3DPW/OC/PC, Higher-HRNet cho Crowd, giống JOTR/DPMesh.
- Cấu hình chính: lifter `finetune_difflifter_d512_e15_jotr_pf05_3dpw.yaml` → Teacher `train_teacher_v3_jotr_pf05.yml` → Student `train_student_v3_kd_jotr_pf05lift3dpw.yml`.

## 2. Kiến trúc Student/Teacher v3

### Phần nền (cố định, không còn cờ)

| Thành phần | Cách xử lý |
|---|---|
| Ảnh | Crop 256×256 → ResNet-50 pretrained SPIN → đặc trưng layer4 (thêm F16/F8 khi cần cho [B]/[D]) |
| Pose | 2D → DiffLifter đóng băng (S=1, K=1) → H36M-17 3D, mét, root-relative. Teacher nhận GT 3D thay cho lifter |
| Token ảnh/khớp | Chiếu về embedding 512; PE2D Fourier dùng chung cho vị trí 2D của khớp và tâm ô ảnh |
| Fusion | 3 block: self-attention khớp, cross-attention hai chiều ảnh↔khớp, FFN |
| Ngữ cảnh xương | HyperGCN tạo `ctx` cho 17 khớp |
| Pose SMPL | 24 query từ mean pose + đặc trưng xương cha/con → cross-attention `ctx` → delta rotation 6D theo khớp; hai khớp tay cuối đặt 0 |
| Shape/camera | Shape token đọc `ctx` + token ảnh; camera từ đặc trưng toàn cục |

### Các module v3 (cờ `cfg.MODEL.v3_*`, mặc định tắt trong `config.py`, bật hết trong các YAML v3)

- **[A]** `v3_bb_finetune`: backbone được train, nhóm lr riêng `TRAIN.lr * v3_bb_lr_scale` (0.1); `lr_warmup` tôn trọng khóa `lr_scale`; BN giữ thống kê (backbone luôn eval). `v3_bb_heatmap`: `ResNetBackbone.enable_heatmap_fusion`, `x += Conv3x3([x, heatmap 2D × conf])` sau maxpool, conv khởi tạo 0; heatmap từ `ARTS.make_heatmaps`.
- **[B]** `v3_joint_feat`: `sample_joint_feats` lấy F16 + F8 tại khớp 2D (mask 0 → 0) → `jf_proj` (khởi tạo 0) cộng vào token khớp trước fusion. **[B']** `v3_hop_bias`: bias attention theo hop (0..8, mỗi head) trong self-attn giữa khớp.
- **[C]** `v3_depth_head`: `JointDepthHead` phân loại độ sâu tương đối pelvis (64 bin, [-1, 1] m); kỳ vọng / entropy / (z lifter − kỳ vọng) → `depth_proj` (khởi tạo 0) cộng vào token khớp; loss CE mềm `base._depth_loss`, trọng số `v3_depth_loss_w`.
- **[D]** `v3_refine_iters` T (YAML: 3): `RefitRefiner` + `StreamGRU` 26 luồng (24 góc 6D + shape + camera thô). Feedback = F16/F8 cửa sổ 3×3 tại H36M-17 chiếu từ mesh + (2D − chiếu)×conf + tham số + `cam_bbox`; ước lượng trước được detach, điểm chiếu clamp [-2, 2]; dropout theo khớp. Output chính = bước cuối; `refine_history` → `base._hard_losses` cộng loss các bước trước với trọng số gamma^(số bước còn lại), chuẩn hóa.
- **[E]** `v3_joint_uncert`: `smpl_model.joint_log_sigma`. AWL nhận L1 có trọng số khớp `exp(-log_sigma)` (detach, chuẩn hóa mean 1, luôn ≥ 0); `log_sigma` học riêng bằng NLL trên lỗi detach (`joint_sigma_nll`), cộng ngoài AWL. `v3_pose2d_noise_prob`: `base._noisy_2d` (jitter + mất khớp) trước lifter/heatmap/PE của Student.

Teacher dùng chung kiến trúc (cờ áp cho cả hai). Teacher dùng 2D ở heatmap [A] và PE2D, nên **phải train cùng phân bố 2D với Student** (vì vậy có `teacher_v3_jotr_pf05` riêng cho dữ liệu PoseFix).

### Teacher và KD

- Teacher: GT 3D + che khớp 30% (`teacher_joint_corrupt_prob`, chỉ khi train) + `img_probe`.
- Student KD: `init_student_from_teacher=True` (khi bật [A] thì copy cả backbone + `heat_fuse` từ Teacher); KD MSE qua `Linear(LN(Student))` tới `LN(Teacher)` trên `feat_joint` (B,17,512) và `feat_global` sau fusion; `kd_w_proj=0`, `kd_w_joint=kd_w_global=1`, `kd_hard_w=1`, `kd_lambda=1`. Scratch (`student_use_kd=False`): loss = `kd_hard_w * hard + privileged` (YAML scratch dùng `kd_hard_w 0.5`).
- `TEST.save_best_per_set=True`: mỗi epoch lưu thêm `checkpoint/best_<tap>.pth.tar` theo MPJPE từng tập trong `test_list`.

## 3. DiffLifter

Code chính: `lib/models/diff_lifter.py`; train độc lập bằng `main/finetune_difflifter.py`.

| Mục | Giá trị trong lifter v3 `d512_e15_jotr_pf05_3dpw` |
|---|---|
| Điều kiện | `(B,17,6)` = `[x, y, conf, cam_bbox(3)]`; conf là mask nhị phân (`lift_soft_conf=False`) |
| Input denoiser | Nối `[x,y,conf]` với noisy 3D, chiếu thành token; cộng joint embedding, timestep embedding và camera embedding |
| Backbone | SpatialMixSTE, 8 block, D=512, 8 head; spatial-only; GCN tắt (`gcn_attn=False`) |
| Diffusion | Cosine T=1000; Gaussian noise; dự đoán `x0`; GT mét × 3 trước khi thêm nhiễu |
| Output/loss | Output trừ pelvis, đơn vị mét; loss MPJPE là L2 norm, không bình phương |
| Sampling | DDIM, eta=1; S bước, K giả thuyết; Student dùng S=1/K=1, seed 0 |
| Train | 5 bộ (H36M, MuCo, MSCOCO, CrowdPose, 3dpw-train), 20% mỗi bộ, 512,000 mẫu/epoch, 15 epoch, EMA 0.999; best theo MPJPE trên `3dpw-val` |
| Augmentation 2D | joint/limb drop 0.05, jitter 0.02, occ_aug 0.15 |

Checkpoint: `difflift_cfg` lưu kiến trúc; `model_state` là EMA; `raw_model_state` phục vụ resume. Đổi kiến trúc lifter phải đồng bộ constructor, `finetune_difflifter.save_ckpt/load_ckpt`, `diag_difflifter.build_model` và `ARTS._build_diff_lifter`. Trọng số `diff_lifter.*` lưu trong `best.pth.tar` của Student được nạp lại khi eval, nên eval Student không phụ thuộc file lifter hiện tại.

## 4. Dữ liệu và hệ tọa độ

**H36M-17**, đúng thứ tự:

```text
Pelvis, R_Hip, R_Knee, R_Ankle, L_Hip, L_Knee, L_Ankle,
Torso, Neck, Nose, Head_top, L_Shoulder, L_Elbow, L_Wrist,
R_Shoulder, R_Elbow, R_Wrist
```

Cha lấy từ `H36M_PARENTS` trong `diff_lifter.py`.

- 2D chuẩn hóa theo `output_hm_shape`: `x/W*2-1`, `y/H*2-1`; không trừ pelvis 2D; mask=0 → tọa độ 0.
- `cam_bbox = [(cx-px)/f, (cy-py)/f, max(w,h)/f]` (`lib/utils/cam_bbox.py`), xử lý flip/rotation augmentation.
- 3D lifter: mét, root-relative tại pelvis H36M. 3DPW test dùng H36M regressor từ GT mesh; H36M test dùng mocap `orig_joint_cam`.

**Nguồn 2D khi train (kiểu JOTR):** `h36m/muco/coco/crowdpose_2d_source: 'gt_posefix'` = 2D GT + nhiễu `utils/posefix.synthesize_pose` (`posefix_scale` 0.5 cho lifter/Teacher/Student hiện tại; `posefix_fast` giống từng bit, nhanh ~10%). `jotr_strict=True`: tên khớp gốc như JOTR (Neck/Torso/Nose của MuCo mask 0), CrowdPose lọc `fit_err < 5`. `update_bbox=True`: scale bbox ngẫu nhiên mỗi lần lấy mẫu. `fusekin_unified=True`: mẫu train thống nhất 4 bộ (`lib/utils/fusekin_sample.py`). Nhãn 3D MSCOCO/CrowdPose là pseudo-SMPL NeuralAnnot, không phải mocap.

**Nguồn 2D khi test:** `pw3d_2d_source: 'openpose'` → OpenPose (3DPW/OC/PC), Higher-HRNet (Crowd, fallback OpenPose). `PW3D` dùng `self.is_crowd` riêng từng dataset.

**Halpe-26 (RTMPose)** vẫn hỗ trợ cho chẩn đoán/so sánh (`pw3d_2d_source: 'halpe26'`, `*_2d_source: 'halpe26'`, `halpe26_to_human36m`). File: `data_final/<bộ>/*_halpe26.*`, tool `tools/extract_halpe26_*.py`. Biến thể 3DPW qua `DATASET.pw3d_halpe_suffix`: `_sam` (SAM che ngoài mask), `_sam_opfix` (thay pose nhầm người bằng OpenPose), `_oracle` (chọn nguồn theo GT, chỉ đo trần). Crowd dùng `3DPW_validation_halpe26*.json` + ann_id validation, không dùng Halpe test.

**Oracle GT 2D:** `DATASET.input_2d_gt=True` thay 2D bằng GT ở mọi bộ (chỉ đo cận trên).

## 5. Config, checkpoint và môi trường

- Key phải khai báo trong `lib/core/config.py` trước khi YAML ghi đè; key không tồn tại gây `ValueError`. Tính năng mới mặc định **tắt**.
- `main/train.py --exp_name NAME` tạo `experiment/NAME`; thư mục đã tồn tại và không rỗng bị từ chối nếu không resume.
- Đánh giá lifter bằng **`best_epoch.bin`**; mesh bằng **`checkpoint/best.pth.tar`** (hoặc `best_<tap>.pth.tar` khi báo theo từng tập như JOTR/DPMesh).
- Script đánh giá dùng chung `lib/utils/ckpt_tools.build_and_load`; `Teacher_Tester = Student_Tester = Tester`.
- Môi trường `arts` cho model; `pose2d` cho RTMPose/ONNX (rtmlib 0.0.16, onnxruntime-gpu 1.19.2).

## 6. Giao thức và chọn checkpoint

| Thành phần | Train | Chọn best |
|---|---|---|
| Lifter `jotr_pf05_3dpw` | 4 bộ + **3dpw-train** | MPJPE `3dpw-val` (chứa 2 sequence Crowd) |
| Teacher `teacher_v3_jotr_pf05` | 4 bộ (H36M, MuCo, MSCOCO, CrowdPose) | 3DPW test |
| Student `student_v3_kd_jotr_pf05lift3dpw` | 4 bộ, không 3dpw-train | 3DPW test (`best.pth.tar`) + `best_<tap>` |
| Mesh best | `main/train.py` xét cải thiện MPJPE **hoặc** MPVPE trên tập được chọn | |

Giới hạn cần ghi khi báo kết quả:
- Lifter đã thấy 3DPW-train → toàn pipeline **chưa** là giao thức JOTR. OC/PC toàn bộ chứa sequence train/val → bị rò rỉ qua lifter; báo OC/PC không rò rỉ với `pw3d_ocpc_test_only=True` (`config/eval_lifter_jotr_pf05_3dpw_ocpctest.yaml`).
- `3dpw-crowd` = 2 sequence của 3DPW validation (courtyard_hug_00, courtyard_dancing_00) → chọn checkpoint theo `3dpw-val` có rò rỉ Crowd.
- Chọn best theo test chỉ là kết quả phát triển; bảng cuối cần tập validation tách riêng.

| Thước đo | Dùng cho |
|---|---|
| H36M-17 root-relative, MPJPE mm | So lifter với lifter; oracle pose input |
| `dataset.evaluate`, 14 khớp, MPJPE/PA-MPJPE/MPVPE mm | So mesh FuseKin |

Không so trực tiếp hai loại MPJPE. Oracle dùng GT chỉ đo tiềm năng.

## 7. Quy ước loss mesh (sửa 2026-10-25)

- Loss mesh: pred trừ pelvis H36M (từ mesh tuyệt đối), cùng gốc với GT. Trước đó trừ pelvis SMPL, lệch ~38 mm.
- Loss chiếu 2D: H36M-17 từ mesh (giống GT 2D) thay vì SMPL-30 theo tên (lệch TB 68 mm). Model trả thêm `smpl_mesh_cam_render_det`.
- `dataset.evaluate` không bị ảnh hưởng. Checkpoint train trước ngày này dùng loss cũ.

## 8. Code cũ còn lại (mặc định tắt, không dùng cho v3)

Các nhánh từ giai đoạn trước v3 vẫn còn trong source nhưng không bật trong YAML v3: gate theo spread (`difflift_spread_gate*`), KD quan hệ (`kd_joint_relation*`), Selector (`cfg.SELECTOR`, `main/train_selector.py`, `main/eval_selector.py`), denoiser GCN (`DIFFLIFT.gcn_attn`). Các script `oracle_best_hypothesis.py`, `analyze_teacher.py`, `inspect_student_spread.py` gọi trực tiếp `smpl_model` theo kiến trúc cũ, chưa hỗ trợ v3. File/code đã gỡ nằm trong `_to_delete/` và lịch sử git.

## 9. Refiner 2D (2026-11-03, đã code, chưa chạy)

Mục tiêu: dùng ảnh sửa 2D detector **trước** khi lift (nhầm người, khớp bị che, khớp bỏ sót). Train độc lập, chưa ghép vào Student.

- Model `lib/models/refiner2d.py` (`Refiner2D`): ResNet-50 SPIN (BN backbone giữ thống kê, lr × `bb_lr_scale`) + heatmap 2D detector ghép ở stride 4 (`heat_fuse`, khởi tạo 0, giống [A]) → 3 deconv (8×8 → 64×64) → 17 heatmap → soft-argmax ra `xy` [-1,1]; head độ bất định `log_b` (đơn vị ô lưới 64) đọc đặc trưng kỳ vọng tại khớp, cờ khớp đầu vào, |dự đoán − đầu vào|, đỉnh xác suất (đều detach).
- Loss (`refiner_losses`, trọng số cố định, không AWL): lỗi Euclid soft-argmax (`w_coord`) + CE heatmap với Gauss mục tiêu chuẩn hóa (`w_heat`) + Laplace NLL `e/b + log b` với e detach (`w_nll`).
- Dữ liệu: 4 bộ train, `fusekin_unified`, 2D `gt_posefix` (PoseFix 1.0 như JOTR). `DATASET.return_clean_2d=True` → `targets['joint_img_clean']` = GT 2D **trước** PoseFix (khung 64, cùng phép đổi tên khớp với đầu vào; không gọi thêm random nên đầu vào giữ nguyên từng bit). Mask = khớp GT trong khung. Khi train bỏ ngẫu nhiên khớp đầu vào (`input_drop_prob`) để học điền khớp thiếu.
- Đánh giá (`lib/utils/refiner2d_tools.py`, `main/eval_refiner2d.py`): 3DPW/OC/PC/Crowd với OpenPose/HHRNet; 12 khớp tay chân; khớp xa = lỗi > 0.2 × kích thước người (giống `compare_2d_sources`). Biến thể `op` / `refine` (giữ mask detector) / `refine_fill` (điền khớp tay chân bị bỏ sót khi b ≤ `--fill_max_b`). Tùy chọn lifter đóng băng → MPJPE17.
- Chọn checkpoint: báo theo **epoch cuối** (lịch cố định); không chọn theo 3dpw-val (chứa Crowd) hay test.
- Config: `cfg.DATASET.return_clean_2d`, `cfg.REFINE2D.*` (mặc định tắt / không ảnh hưởng run cũ). YAML `config/train_refiner2d_jotr.yaml`. Bản sao file dataset/config trước khi sửa: `_to_delete/bak_refiner2d/`.

## 10. ImgLifter (2026-11-04, đã code, chưa chạy) — thay DiffLifter cũ

- `lib/models/img_lifter.py`: diffusion 2D → 3D **có ảnh**. Đầu vào từ Refiner 2D đóng băng (`refiner2d_jotr_mesh`): `xy` (định nghĩa mesh), `log_b`, cờ detector có khớp, `cam_bbox`, `f8` (2048×8×8 → 64 token ảnh), `feat_j` (đặc trưng kỳ vọng tại khớp; `Refiner2D.forward(..., return_feats=True)`).
- Điều kiện khớp c_j = MLP([x, y, log_b, det, cam_bbox]) + Linear(feat_j): cộng ở đầu vào; adaLN-Zero (từ c_j + t) ở mọi block; cross-attn khớp → 64 token ảnh ở mọi block. 8 block, D=512, 8 head. Diffusion như DiffLifter cũ (cosine T=1000, dự đoán x0, scale 3, DDIM, loss MPJPE L2).
- Train (`main/train_img_lifter.py`): 4 bộ, không 3DPW; 2D vào Refiner = GT + PoseFix 1.0; bỏ ngẫu nhiên khớp vào (`input_drop_prob`); nhiễu thêm vào xy theo b (`train_xy_noise`, b chặn 8 ô); mục tiêu `orig_joint_cam` (H36M-17 từ mesh, mét, root-relative); EMA 0.999.
- Suy luận: K giả thuyết × S bước DDIM; gộp `mean` hoặc `jpma` (fit weak-perspective trọng số 1/b², chọn giả thuyết theo từng khớp khi b ≤ `jpma_max_b`). Eval (`main/eval_img_lifter.py`, `lib/utils/img_lifter_tools.py`): MPJPE17/PA, oracle min-K, `--ablate_img`, so với lifter cũ trên cùng mẫu.
- Config `cfg.IMGLIFT.*`, YAML `config/train_img_lifter_jotr.yaml`. Chưa ghép vào Student.
