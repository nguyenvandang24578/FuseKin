# FuseKin — kết quả và tiến độ thí nghiệm (FuseKin-v3)

Cập nhật **2026-11-03**. Bản này đã rút gọn, chỉ giữ dòng FuseKin-v3 và kết luận còn dùng được. Bản đầy đủ trước khi rút gọn (mục 1–51, gồm toàn bộ thí nghiệm trước v3) nằm ở `_to_delete/docs_bak_20261103/experiment.md` và trong lịch sử git.

Kiến trúc và quy ước: [01_PROJECT_CONTEXT.md](01_PROJECT_CONTEXT.md). Lệnh chạy: [README.md](README.md).

**Quy ước ghi:** mỗi kết quả ghi tên run/YAML, checkpoint + epoch, dữ liệu train, tập chọn best, tập đánh giá và thước đo. Mesh = `dataset.evaluate` 14 khớp, MPJPE / PA-MPJPE / MPVPE (mm). Lifter = H36M-17 root-relative, MPJPE17 / PA (mm). Không so trực tiếp hai loại. Số liệu dưới đây do người dùng chạy trên server và báo lại.

## 1. Trạng thái hiện tại

| Hạng mục | Run / YAML | Trạng thái |
|---|---|---|
| Giao thức chốt | 2D test OpenPose (3DPW/OC/PC), Higher-HRNet (Crowd) — như JOTR/DPMesh | Người dùng chốt 2026-11-03 |
| Lifter | `finetune_difflifter_d512_e15_jotr_pf05_3dpw` (`best_epoch.bin`) | Có kết quả (mục 6). Có train 3DPW-train → **không** còn giao thức JOTR |
| Teacher v3 | `train_teacher_v3_jotr_pf05.yml` → `teacher_v3_jotr_pf05` | Có kết quả: 3DPW 27.52 / 19.28 / 37.72 |
| Student v3 KD | `train_student_v3_kd_jotr_pf05lift3dpw.yml` → `student_v3_kd_jotr_pf05lift3dpw_t2` | Có kết quả: 3DPW 78.62, OC 73.12, PC 88.82, Crowd 83.02 |
| Nút thắt | Pose đầu vào từ lifter (lỗi khớp đầu vào khi train ~93 mm) | Student ≈ lifter, ảnh chưa sửa được 2D sai (mục 7) |
| Hướng tiếp | **Refiner 2D**: ảnh sửa 2D detector trước khi lift; bỏ KD khỏi pipeline chính; lifter train lại sau khi Refiner có hiệu quả | Đã code, chưa chạy (mục 9) |

## 2. Mốc so sánh

| Phương pháp | 3DPW MPJPE / PA / MPVPE | OC | PC | Crowd |
|---|---|---:|---:|---:|
| JOTR | 76.4 / 48.7 / 92.6 | | | |
| DPMesh | 73.6 / 47.4 / 90.7 | 70.9 | 82.2 | 79.9 |
| InstanceHMR | 73.2 / 44.3 / 80.3 | | | |

OC/PC của JOTR/DPMesh tính trên **toàn bộ** sequence (giao thức JOTR); OC n = 20,243.

## 3. Kiến trúc FuseKin-v3 và sự cố khi train

Thiết kế tổng hợp từ 3DCrowdNet, DPMesh, JOTR, ReFit, PandaPose, SAM 3D Body, ScoreHypo, BlockGCN (chi tiết: `01_PROJECT_CONTEXT.md` mục 2):
[A] fine-tune backbone + heatmap 2D; [B] đặc trưng ảnh tại khớp, [B'] hop bias; [C] phân bố độ sâu từng khớp; [D] tinh chỉnh lặp kiểu ReFit; [E] loss độ bất định + nhiễu 2D.

- **Vỡ train ở epoch 6–7 (đã sửa):** NLL của [E] (`|e|/σ + log σ`) âm khi σ < 1 m, đưa vào AWL làm trọng số phình tới ~7.5e5. Sửa trong `lib/core/base.py`: AWL nhận L1 có trọng số khớp `exp(-log_sigma)` (detach, chuẩn hóa mean 1); `log_sigma` học riêng bằng NLL trên lỗi detach (`joint_sigma_nll`), cộng ngoài AWL. Phải train lại từ đầu.
- **[D]:** feedback dùng ước lượng trước đã detach + clamp điểm chiếu [-2, 2].
- **Sửa loss gốc (áp dụng cho mọi checkpoint sau 2026-10-25):** loss mesh trừ pelvis H36M (trước trừ pelvis SMPL, lệch ~38 mm); loss chiếu 2D dùng H36M-17 từ mesh (trước SMPL-30 theo tên, lệch ~68 mm). Eval không bị ảnh hưởng.

## 4. Kết quả v3 với 2D Halpe (giai đoạn trước khi chốt OpenPose)

Train 4 bộ (H36M, MuCo, MSCOCO, CrowdPose), không 3DPW-train; lifter `finetune_difflifter_r25_cam_gtmask_d512_e40` (latest ep40); best chọn theo **3DPW test** (kết quả phát triển).

| Run | 3DPW MPJPE / PA / MPVPE | OC | PC | Crowd |
|---|---|---:|---:|---:|
| `student_v3_scratch_jotr` (best ep15) | 75.53 / 48.31 / 90.22 | 63.24 | 93.04 | 97.54 |
| `teacher_v3_jotr` (GT 3D, oracle) | 25.70 / 16.99 / 36.41 | | | |
| `student_v3_kd_jotr` (22 epoch) | **74.62 / 47.64 / 89.80** | 62.49* | 90.48* | 95.40* |

\* best từng tập đến ep13 (`best_<tap>.pth.tar`). KD so với scratch −0.91 mm, nhỏ hơn dao động giữa epoch (±1 mm, 1 seed). So DPMesh: 3DPW MPVPE thắng, MPJPE thua 1.0 mm; OC thắng, PC/Crowd thua xa.

## 5. Chẩn đoán 2D: lỗi nằm ở đâu

Công cụ: `main/diag_2d_crowd.py` (checkpoint v3 scratch, 2D Halpe).

| Tập | Mesh MPJPE halpe → fixfar → dropfar | % mẫu có khớp xa tự tin |
|---|---|---:|
| 3DPW | 75.53 → 74.63 | 3.2% |
| OC | 63.24 → 62.97 | 2.7% |
| PC | 93.04 → **80.75** (dropfar 89.47) | 9.0% |
| Crowd | 97.54 → **83.55** (dropfar 93.64) | 10.2% |

- Khoảng cách PC/Crowd gần như hoàn toàn do ~10% mẫu **nhầm người** (MPJPE ~260 mm); mẫu sạch ngang 3DPW. Chỉ bỏ khớp sai lấy lại rất ít, cần tọa độ đúng.
- Lỗi sinh từ DiffLifter khi 2D nhầm người; mesh chỉ chép lại ([D] không khuếch đại). Crop test không phải vấn đề (IoU 0.84–0.88).
- **Nhiễu "nhầm người" khi train lifter (wp_aug):** không sửa được mẫu xấu (PC/Crowd xấu 221 → 222 mm). Từ 2D đơn thuần không phân biệt được một đoạn chi bị dời với tư thế thật → đã xóa cờ.
- **Oracle GT 2D** (lifter + Student v3 scratch train/test GT 2D): 3DPW 58.96 / – / 71.88; Crowd 56.74 / 36.56 / 71.31 → 2D là nút thắt chung, không chỉ ở crowd (cận trên lạc quan).
- **Nguồn 2D trên 3DPW-Crowd** (`main/compare_2d_sources.py`): Halpe (RTMPose trên bbox GT rộng) bắt nhầm người bên cạnh; OpenPose / Higher-HRNet bám đúng người hơn. Halpe che mask SAM (`_sam`): mẫu có khớp xa 20.9% → 18.0% (sửa 113 / hỏng 58); OpenPose 2.9% (bỏ 13.9%), HHRNet 5.1%.
- Chưa có kết quả: `_sam_opfix` (thay pose Halpe nhầm người bằng OpenPose), oracle chọn nguồn 2D theo GT (`config/test_student_v3_kd_jotr_oracle.yml`, chỉ đo mức trần).

## 6. Lifter giao thức JOTR (2D GT + PoseFix khi train, OpenPose/HHRNet khi test)

- Dữ liệu 2D đồng bộ với JOTR: `*_2d_source: 'gt_posefix'`, `jotr_strict`, `update_bbox`, mask nhị phân. `tools/check_jotr_equivalence.py`: khớp 100% cả 8 bộ (200 mẫu/bộ). Khác JOTR: CrowdPose chỉ giữ người có NeuralAnnot (35,379 vs 40,504).
- PoseFix tốn ~74% thời gian sinh dữ liệu; `posefix_fast` giống từng bit, chỉ nhanh ~10%.

| Lifter (MPJPE17 / PA) | 3DPW | H36M | OC | PC | Crowd |
|---|---|---|---|---|---|
| Halpe `gtmask_d512_e40` best ep11 | 73.11 / 49.24 | 55.45 / 42.86 | 80.44 / 52.86 | 95.94 / 60.28 | 99.94 / 55.88 |
| `d512_e40_jotr` (PoseFix 1.0) ep15 | 85.56 / 53.72 | 57.41 / 44.90 | 93.20 / 62.93 | 100.40 / 73.91 | 88.79 / 61.14 |
| `d512_e15_jotr_pf05` (PoseFix 0.5) ep7 | 81.43 / 52.59 | | | | |
| `d512_e15_jotr_pf05` ep5 | 82.47 / 53.84 | 64.16 / 50.17 | 93.13 / 63.52 | 99.95 / 74.16 | 84.64 / 58.90 |

- PoseFix + OpenPose: chỉ Crowd tốt hơn; PA kém ở mọi tập. Sau ep7 lệch phân bố train/test (3DPW xấu dần). Chưa tách hai yếu tố (nhiễu PoseFix vs nguồn 2D OpenPose).
- **`d512_e15_jotr_pf05_3dpw`** (thêm 3dpw-train, 2D OpenPose, 20% mỗi bộ), đánh giá **không rò rỉ** với `eval_lifter_jotr_pf05_3dpw_ocpctest.yaml` (OC/PC chỉ sequence test), MPJPE17: 3DPW **77.47** | OC-test 79.43 (n 4157) | PC-test 40.98 (n 281, quá ít) | Crowd 69.69. OC/PC toàn bộ bị rò rỉ ~25 mm qua 3DPW-train; Crowd rò rỉ ở bước chọn checkpoint (3dpw-val chứa 2 sequence Crowd).

## 7. Student v3 KD giao thức OpenPose

- `train_student_v3_kd_jotr_pf05lift3dpw.yml`: Student train 4 bộ với 2D GT + PoseFix 0.5, tỉ lệ 0.25 mỗi bộ, lifter `jotr_pf05_3dpw`; test OpenPose/HHRNet; OC/PC toàn bộ. Có rò rỉ qua lifter → không phải giao thức JOTR.
- Teacher phải nhận cùng phân bố 2D với Student (2D đi vào heatmap [A] và PE 2D): `teacher_v3_jotr_pf05` 3DPW 27.52 / 19.28 / 37.72 (Halpe 25.70 / 16.99 / 36.41). Chạy với Teacher Halpe cũ (`student_v3_kd_jotr_pf05lift3dpw`) xấu dần ep1 → ep3, chỉ để tham khảo.

| Run `student_v3_kd_jotr_pf05lift3dpw_t2` (đến ep15) | MPJPE |
|---|---:|
| 3DPW | 78.62 (ep11; PA tốt nhất 50.68 ep14) |
| OC | 73.12 (ep13) |
| PC | 88.82 (ep4) |
| Crowd | 83.02 (ep14) |

So v3 KD Halpe: 3DPW +4.0, OC +10.6, PC −1.7, Crowd −12.4. Chưa vượt DPMesh ở tập nào.

**Chẩn đoán KD** (`main/diag_kd_features.py`, 3000 mẫu 3DPW / 1923 Crowd): Teacher MPJPE14 24.96 / 30.52; Student 85.52 / 82.88. Cosine Student–Teacher: token đầu vào 0.26, token khớp sau fusion 0.88, global 0.91. → Tầng sâu đã giống Teacher; khoảng cách 25 vs 83–86 mm nằm ở **đầu vào** (pose lifter vs GT 3D), KD không lấp được. Mẫu lệch Teacher nhiều là mẫu lỗi lớn (Spearman joint +0.85 / +0.79; tương quan, chưa phải nhân quả).

`main/diag_student_vs_lifter.py` (lỗi tay chân lifter vs Student theo nhóm 2D tốt / xa / thiếu, ảnh xám, oracle fixfar): đã code, chưa có kết quả.

## 8. Kết luận giữ lại từ giai đoạn trước v3

- Nhóm giả thuyết DiffLifter có chứa pose tốt (oracle K=20 giảm mạnh lỗi), nhưng mọi cách chọn tự động đã thử (selector bằng ảnh, gate theo spread, augmentation bằng giả thuyết) đều **không** chuyển thành lợi ích mesh rõ ràng → không tiếp tục.
- Spread (std giữa giả thuyết) tương quan yếu với lỗi từng khớp (Spearman 0.09–0.31); không dùng như độ tin cậy.
- Denoiser GCN + attention song song không tốt hơn attention thuần (3DPW +1.5 mm).
- Teacher GT 3D gần như không dùng ảnh để sửa pose hỏng (ảnh TB chỉ +0.9–2.2 mm); Student thì dùng ảnh (+8–10 mm).
- Prior SimCC/GMM: tạm hoãn; code probe giữ lại.

## 9. Refiner 2D (2026-11-03, người dùng xác nhận, đã code, chưa chạy)

- Lý do: oracle GT 2D giảm ~16 mm trên 3DPW; sửa ~3% khớp sai thô giảm 12–14 mm trên PC/Crowd; Student ≈ lifter → ảnh cần sửa 2D **trước** lifter. Chỉ dùng chuẩn OpenPose cho benchmark (người dùng chốt), train bằng GT + PoseFix như JOTR, không trộn Halpe.
- Thiết kế và code: `01_PROJECT_CONTEXT.md` mục 9. YAML `config/train_refiner2d_jotr.yaml` (PoseFix 1.0, 20 epoch × 64,000 mẫu, batch 64, lr head 1e-3 / backbone 1e-4, giảm ×0.1 ở epoch 15, 18).
- Lệnh:
  - `python main/train_refiner2d.py --cfg config/train_refiner2d_jotr.yaml --exp_name refiner2d_jotr --gpu 0`
  - `python main/eval_refiner2d.py --cfg config/train_refiner2d_jotr.yaml --ckpt experiment/refiner2d_jotr/checkpoint/last.pth.tar --lifter_ckpt experiment/finetune_difflifter_d512_e15_jotr_pf05_3dpw/best_epoch.bin --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --gpu 0`
- **Điều kiện đi tiếp:** % khớp xa trên PC/Crowd thấp hơn `op`, **và** lifter MPJPE17 với `refine` hoặc `refine_fill` thấp hơn `op` trên 3DPW và Crowd (mốc lifter `jotr_pf05_3dpw`: 77.47 / 69.69, cùng lifter với `op` trong cùng lần chạy là mốc chính). Không đạt → dừng hướng này.
- Sau đó: ghép 2D đã sửa vào Student (heatmap [A], PE2D, [D], lifter), bỏ KD; rồi train lại lifter sạch (không 3DPW-train) theo phân bố lỗi của Refiner.
- **Kết quả run `refiner2d_jotr` (target annotation, người dùng 2026-11-03):** train tốt dần (lỗi 2D ra 18.5 → 10.1 → 7.6 px ở ep1/3/7, vào ~20 px) nhưng test so với GT mesh **xấu dần**: 3DPW lỗi % refine 6.29 (ep3) → 6.69 (ep7), OpenPose 6.19. Lifter `jotr_pf05_3dpw` (3000 mẫu, ep3) MPJPE17 op / refine / refine_limb / op_fill: 3DPW 90.01 / 95.47 / 96.23 / 89.49; PC 76.96 / 86.15 / 86.38 / 78.85; Crowd 69.60 / 79.38 / 78.48 / 70.78. Giả thuyết lỗi do Pelvis/Neck/Nose **sai** (refine_limb ≈ refine). OpenPose chỉ ~0.5% khớp xa; điền khớp thiếu không giúp lifter. → Dừng run này.
- **Run mới `refiner2d_jotr_mesh`** (chỉ đổi `REFINE2D.target: 'mesh'` = 2D H36M-17 tính từ mesh GT, cùng định nghĩa GT 3D benchmark): `python main/train_refiner2d.py --cfg config/train_refiner2d_jotr_mesh.yaml --exp_name refiner2d_jotr_mesh --gpu 0`. Tiêu chí: `loi %` refine thấp hơn rõ op (6.19 trên 3DPW). Lifter hiện tại học 2D định nghĩa annotation/OpenPose → muốn đo MPJPE công bằng cần train lifter với 2D định nghĩa mesh (bước sau). Kết quả: chưa có.

## 10. ImgLifter thay DiffLifter cũ (2026-11-04, người dùng xác nhận, đã code, chưa chạy)

- Kết quả Refiner target mesh (`refiner2d_jotr_mesh`, ep20, 3000 mẫu đầu, lỗi 2D % kích thước người op → refine): 3DPW 6.19 → 4.67 (khớp xa 0.46 → 0.11; sửa 129 / hỏng 10), PC 5.35 → 4.16, Crowd 5.78 → 4.45 (khớp xa 0.66 → 1.51; sửa 41 / hỏng 223). corr(log b, log lỗi) 0.29–0.46. → Đạt điều kiện, giữ Refiner.
- ImgLifter (thiết kế: `01_PROJECT_CONTEXT.md` mục 10) gộp: ảnh trong denoiser, 2D Refiner + b, adaLN mọi block, 3D định nghĩa mesh, K giả thuyết + JPMA, không 3DPW-train. Người dùng chọn thay một lần (không thử từng phần).
- Lệnh: `python main/train_img_lifter.py --cfg config/train_img_lifter_jotr.yaml --exp_name img_lifter_jotr --gpu 0`; eval: `python main/eval_img_lifter.py --cfg config/train_img_lifter_jotr.yaml --ckpt experiment/img_lifter_jotr/checkpoint/last.pth.tar --old_lifter_ckpt experiment/finetune_difflifter_d512_e15_jotr_pf05_3dpw/best_epoch.bin --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --gpu 0`.
- Mốc: lifter cũ `jotr_pf05_3dpw` (có 3DPW-train) 3DPW 77.47 / Crowd 69.69 MPJPE17 toàn tập; eval có cột `old_lifter_op` trên cùng mẫu. Phép tách: `--ablate_img`, mean vs jpma, oracle min-K.
- Kết quả: chưa có.

## 11. Refiner DÀY: 17 khớp + 48 đỉnh mesh (2026-11-04, người dùng xác nhận, đã code, chưa chạy)

- Lý do: HiPART (arXiv 2503.23331) cho thấy 2D dày tính từ mesh giúp lifter rõ (48 điểm −5.4 mm, 96 điểm −4.7 mm trên H36M); họ làm dày không dùng ảnh, Refiner của mình dự đoán điểm dày từ ẢNH.
- Nhãn: `lib/utils/dense2d.py` chiếu 48 đỉnh SMPL cố định (FPS trên mesh mẫu, `tools/select_dense_vertices.py` → `config/dense48_fps.json`) từ mesh GT vào crop; tính ngay khi đọc dữ liệu (`fusekin_sample`, 3DPW test khi `pw3d_test_gt2d`). Lật ảnh: mesh dựng từ pose đã lật nên đỉnh giữ đúng vị trí giải phẫu.
- Refiner: `num_dense` kênh thêm sau 17 khớp; `init_ckpt` chép từ `refiner2d_jotr_mesh` (17 kênh đầu của lớp final). ImgLifter hiện tại chỉ lấy 17 khớp đầu.
- Trình tự: (1) `python tools/select_dense_vertices.py --n 48 --out config/dense48_fps.json` (xem hình .png); (2) `python tools/vis_dense2d.py --cfg config/train_refiner2d_jotr_mesh_dense48.yaml --n 8` (người dùng kiểm tra hình); (3) `python main/train_refiner2d.py --cfg config/train_refiner2d_jotr_mesh_dense48.yaml --exp_name refiner2d_dense48 --gpu 0`.
- Kiểm tra sau train: 17 khớp không tệ hơn bản mesh (3DPW 4.67%), và lỗi điểm dày. Kết quả: chưa có.

## 12. ImgLifter DÀY + trộn 3dpw-train (2026-11-04, người dùng xác nhận, đã code, chưa chạy)
- Refiner dày đóng băng `refiner2d_dense48_scratch` (last). ImgLifter: 17 token khớp (khử nhiễu → 3D) + 48 token ngữ cảnh (chỉ điều kiện 2D/ảnh, det = 0, không đầu ra 3D). Key `IMGLIFT.num_ctx` (0 = tắt, ckpt cũ vẫn nạp được).
- Train 5 bộ, tỷ lệ 0.2 mỗi bộ: Human36M, MuCo, MSCOCO, CrowdPose (GT + PoseFix) + 3dpw-train (OpenPose). Key `DATASET.drop_raw_smpl_stats` để collate được.
- Đổi 2 yếu tố so với `img_lifter_jotr` (dày + 3dpw-train) → không tách được đóng góp. 3dpw-train phá giao thức JOTR với OC/PC → báo thêm `--ocpc_mode test`.
- Config `config/train_img_lifter_dense48_3dpw.yaml`, exp `img_lifter_dense48_3dpw`. Kết quả: chưa có.

## 13. ImgLifter dày + lớp Markov + loss xương/góc (2026-11-05, người dùng xác nhận gộp, đã code, chưa chạy)
- Ý tưởng Markov Optimization Layer (DiffusionPose, AAAI-26): mỗi block truyền tin giữa 17 token khớp theo 16 xương H36M, msg = ψ(h_i)·φ(h_i,h_j), gộp logsumexp, khởi tạo 0. Điểm mới: cổng sigmoid(a(log_b − c)) theo độ bất định Refiner → khớp mơ hồ nhận nhiều từ khớp kề.
- Loss trên x0: `w_bone` × L1 độ dài xương + `w_angle` × (Δcos góc 2 xương nối tiếp)². Key `IMGLIFT.markov_layer / markov_gate_b / w_bone / w_angle`, mặc định tắt.
- Eval thêm MPJPE tách khớp detector sót / có.
- Config `config/train_img_lifter_dense48_3dpw_markov.yaml`, so với §12. Đổi 2 thành phần cùng lúc; ablation chạy sau. Kết quả: chưa có.

## 14. ImgLifter Markov v2 (2026-11-05, người dùng duyệt sửa gộp 3 điểm, đã code/chờ chạy)

- Mốc so sánh: bản Markov §13, cùng 5 bộ train, Refiner dày, lịch 25 epoch và trọng số `w_bone=0.5 / w_angle=0.1`. Chưa có kết quả thực nghiệm được cung cấp cho hai bản.
- Sửa tích thông điệp theo cạnh có hướng, loại cạnh quay lại (8 vòng/block); length loss đổi từ L1 sang công thức (4) của DiffusionPose; cổng bất định dùng hệ số dương. Chi tiết và lựa chọn `L_std`/số vòng nằm ở `01_PROJECT_CONTEXT.md` §11. Giữ nguyên 13 cặp góc, MPJPE L2, đơn vị/root của pose.
- Đây là run gộp theo phạm vi người dùng đã duyệt, **không tách được đóng góp ba thay đổi**. Length loss mới khác thang đo L1 cũ; không coi trọng số giữ nguyên là đóng góp gradient giữ nguyên. Log thêm loss đã nhân trọng số và total để theo dõi. Chưa chạy code, train/test hoặc kiểm cú pháp bằng Python tại máy local.
- Config mới `config/train_img_lifter_dense48_3dpw_markov_v2.yaml`, kết quả mới `experiment/img_lifter_dense48_3dpw_markov_v2/`; không sửa YAML/run cũ và không resume checkpoint Markov cũ bằng YAML v2. Các key mới mặc định tắt, builder và checkpoint có lưu chế độ; resume từ metadata khác bị từ chối.
- Train: `python main/train_img_lifter.py --cfg config/train_img_lifter_dense48_3dpw_markov_v2.yaml --exp_name img_lifter_dense48_3dpw_markov_v2 --gpu 0`.
- Eval sau khi hoàn tất lịch train: `python main/eval_img_lifter.py --cfg config/train_img_lifter_dense48_3dpw_markov_v2.yaml --ckpt experiment/img_lifter_dense48_3dpw_markov_v2/checkpoint/last.pth.tar --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --ocpc_mode test --gpu 0`. Dùng epoch cuối theo giao thức ImgLifter; không tự chọn theo log test. Có 3DPW-train nên không báo là giao thức JOTR; OC/PC toàn bộ chỉ để tham khảo.
- Bước tiếp: người dùng chạy trên server, gửi log loss/train và kết quả eval MPJPE17 cùng K/S/seed. Chưa có bằng chứng để kết luận cải thiện.
