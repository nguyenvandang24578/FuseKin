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
- Config `config/train_img_lifter_dense48_3dpw.yaml`, exp `img_lifter_dense48_3dpw`.
- Kết quả (người dùng gửi 2026-11-05, `last` = epoch 25, K=10 S=4, mean; OC/PC chỉ seq test), so với Markov v1 §13:

| Tập | Không Markov | Markov v1 | Δ | Khớp sót (không / có Markov) | Khớp có (không / có Markov) |
|---|---:|---:|---:|---|---|
| 3DPW | 66.41 | 65.80 | −0.61 | 81.88 / 81.50 | 63.30 / 62.64 |
| OC (test) | 71.14 | 71.27 | +0.13 | 75.95 / 76.43 | 70.34 / 70.41 |
| PC (test, n=281) | 49.24 | 54.82 | +5.58 | 18.50 / 30.97 | 53.34 / 58.00 |
| Crowd | 65.38 | 64.63 | −0.75 | 94.40 / 95.16 | 59.24 / 58.17 |

  Markov v1 (+ loss xương/góc) chỉ hơn < 1 mm trên 3DPW/Crowd, không hơn trên OC, kém trên PC (281 mẫu, nhiễu lớn). Phần giảm nằm ở khớp CÓ, không ở khớp SÓT → chưa ủng hộ giả thuyết "Markov giúp khớp bị che"; chưa tách được Markov vs loss.

## 13. ImgLifter dày + lớp Markov + loss xương/góc (2026-11-05, người dùng xác nhận gộp, đã code, chưa chạy)
- Ý tưởng Markov Optimization Layer (DiffusionPose, AAAI-26): mỗi block truyền tin giữa 17 token khớp theo 16 xương H36M, msg = ψ(h_i)·φ(h_i,h_j), gộp logsumexp, khởi tạo 0. Điểm mới: cổng sigmoid(a(log_b − c)) theo độ bất định Refiner → khớp mơ hồ nhận nhiều từ khớp kề.
- Loss trên x0: `w_bone` × L1 độ dài xương + `w_angle` × (Δcos góc 2 xương nối tiếp)². Key `IMGLIFT.markov_layer / markov_gate_b / w_bone / w_angle`, mặc định tắt.
- Eval thêm MPJPE tách khớp detector sót / có.
- Config `config/train_img_lifter_dense48_3dpw_markov.yaml`, so với §12. Đổi 2 thành phần cùng lúc; ablation chạy sau.
- Kết quả (người dùng gửi 2026-11-05, `last` = epoch 25, K=10 S=4, gộp mean, MPJPE17 / PA, mm; OC/PC chỉ seq test):

| Tập | n | MPJPE17 | PA | khớp sót / có | Lifter cũ |
|---|---:|---:|---:|---|---:|
| 3DPW | 35,515 | 65.80 | 43.78 | 81.5 / 62.6 (sót 16.7%) | 77.47 |
| OC (test) | 4,157 | 71.27 | 45.83 | 76.4 / 70.4 | |
| PC (test) | 281 | 54.82 | 56.65 | 31.0 / 58.0 | |
| Crowd | 1,923 | 64.63 | 47.62 | 95.2 / 58.2 (sót 17.5%) | 69.69 |

  jpma kém mean ở mọi tập; oracle min-K thấp hơn mean ~6–8 mm. PC test chỉ 281 mẫu → không ổn định. Chưa có bản §12 (không Markov) để tách đóng góp Markov.

## 14. ImgLifter Markov v2 (2026-11-05, người dùng duyệt sửa gộp 3 điểm, đã code/chờ chạy)

- Mốc so sánh: bản Markov §13, cùng 5 bộ train, Refiner dày, lịch 25 epoch và trọng số `w_bone=0.5 / w_angle=0.1`. Chưa có kết quả thực nghiệm được cung cấp cho hai bản.
- Sửa tích thông điệp theo cạnh có hướng, loại cạnh quay lại (8 vòng/block); length loss đổi từ L1 sang công thức (4) của DiffusionPose; cổng bất định dùng hệ số dương. Chi tiết và lựa chọn `L_std`/số vòng nằm ở `01_PROJECT_CONTEXT.md` §11. Giữ nguyên 13 cặp góc, MPJPE L2, đơn vị/root của pose.
- Đây là run gộp theo phạm vi người dùng đã duyệt, **không tách được đóng góp ba thay đổi**. Length loss mới khác thang đo L1 cũ; không coi trọng số giữ nguyên là đóng góp gradient giữ nguyên. Log thêm loss đã nhân trọng số và total để theo dõi. Chưa chạy code, train/test hoặc kiểm cú pháp bằng Python tại máy local.
- Config mới `config/train_img_lifter_dense48_3dpw_markov_v2.yaml`, kết quả mới `experiment/img_lifter_dense48_3dpw_markov_v2/`; không sửa YAML/run cũ và không resume checkpoint Markov cũ bằng YAML v2. Các key mới mặc định tắt, builder và checkpoint có lưu chế độ; resume từ metadata khác bị từ chối.
- Train: `python main/train_img_lifter.py --cfg config/train_img_lifter_dense48_3dpw_markov_v2.yaml --exp_name img_lifter_dense48_3dpw_markov_v2 --gpu 0`.
- Eval sau khi hoàn tất lịch train: `python main/eval_img_lifter.py --cfg config/train_img_lifter_dense48_3dpw_markov_v2.yaml --ckpt experiment/img_lifter_dense48_3dpw_markov_v2/checkpoint/last.pth.tar --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --ocpc_mode test --gpu 0`. Dùng epoch cuối theo giao thức ImgLifter; không tự chọn theo log test. Có 3DPW-train nên không báo là giao thức JOTR; OC/PC toàn bộ chỉ để tham khảo.
- Bước tiếp: người dùng chạy trên server, gửi log loss/train và kết quả eval MPJPE17 cùng K/S/seed. Chưa có bằng chứng để kết luận cải thiện.

## 15. ImgLifter Markov v3 (2026-11-05, người dùng xác nhận, đã code, chưa chạy)
- Người dùng báo: run v2 có sai số train lớn. Giả thuyết (chưa kiểm chứng): tích 8 vòng tin nhắn có dấu, không chuẩn hóa → triệt tiêu/bùng nổ.
- Sửa: `IMGLIFT.markov_msg_norm` (mặc định tắt) chuẩn hóa RMS tin nhắn sau mỗi vòng (dấu "∝" của công thức (1)). Bỏ `pw3d_ocpc_test_only` khỏi YAML để eval nhanh PC so được với §12/§13.
- Config `config/train_img_lifter_dense48_3dpw_markov_v3.yaml`, run mới (không resume v2). Kết quả: chưa có.

## 16. Student v4: Refiner dày + ImgLifter Markov v1 (2026-11-05, người dùng xác nhận, đã code, chưa chạy)
- Giữ kiến trúc Student v3 [A]–[E]. Đổi: (1) pose 3D đầu vào = Refiner dày + ImgLifter Markov v1 đóng băng (K=5, S=2, mean) thay DiffLifter cũ; (2) 2D cho Student ([A], PE, [B], [D]) = 17 khớp 2D Refiner, mask 1; (3) bỏ KD/Teacher; (4) PoseFix 1.0. Student train 4 bộ, không 3dpw-train (lifter có).
- Key: `MODEL.imglift_checkpoint` ('' = DiffLifter cũ), `imglift_refiner_checkpoint`, `imglift_K`, `imglift_S`, `student_2d_from_refiner`.
- Config `config/train_student_v4_imglift_markov.yml`, exp `student_v4_imglift_markov`. Đổi nhiều yếu tố cùng lúc. Báo epoch cuối (`last.pth.tar`), OC/PC chỉ seq test.
- Kết quả (người dùng gửi 2026-11-06, `last.pth.tar` epoch 22, `test_student_3dpw_sets.py --ocpc_mode test`). LIFT = pose ImgLifter đầu vào Student đo CÙNG thước đo 14 khớp mesh:

| Tập | n | MPJPE | PA | MPVPE | LIFT | LIFT-PA | Student − LIFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3DPW | 35,515 | 74.33 | 48.77 | 90.62 | 72.59 | 47.36 | +1.7 |
| OC (test) | 4,157 | 81.64 | 51.31 | 95.37 | 78.79 | 49.08 | +2.9 |
| PC (test) | 281 | 70.76 | 71.16 | 110.65 | 62.10 | 64.48 | +8.7 |
| Crowd | 1,923 | 78.15 | 53.93 | 100.48 | 70.90 | 51.47 | +7.3 |

  Student KÉM pose lifter đầu vào ở mọi tập (rõ nhất Crowd/PC) → Student chưa thêm giá trị, còn làm rơi thông tin. So DPMesh: 3DPW 74.33 vs 73.6 (thua 0.7), Crowd 78.15 vs 79.9 (thắng 1.8).

## 17. Student v5: query nhận biết hình học xương (2026-11-07, người dùng xác nhận, đã code, chưa chạy)
- Chẩn đoán v4 (§16, `last`): lift → token (sau fusion, TRƯỚC HyperGCN) → mesh_it0 → mesh cuối = 3DPW 72.59 → 73.98 → 74.04 → 74.33; Crowd 70.90 → 70.64 → 78.10 → 78.15; OC 78.79 → 79.25 → 81.63 → 81.64. Mất ở HyperGCN hoặc query SMPL (chưa tách được); [D] đóng góp ~0.
- v5 = v4 + `MODEL.v5_bone_query`: `ctx_joint_head` giải mã 3D từ ctx (sau HyperGCN, loss smooth-L1 với GT, trọng số `v5_ctx_loss_w`) → vector xương cha→con (hướng + độ dài) → MLP (khởi tạo 0) cộng vào 24 query SMPL; bỏ [D] (`v3_refine_iters 0`). Chẩn đoán thêm cột `ctx`.
- Config `config/train_student_v5_bonequery.yml`, exp `student_v5_bonequery`. Kết quả: chưa có.
- Biến thể `config/train_student_v5_bonequery_jotr.yml` (exp `student_v5_bonequery_jotr`): chỉ đổi `TRAIN.mix_mode: 'jotr'` (server mạnh); lịch epoch tạm `end_epoch 10`, `lr_step [6, 9]`, chỉnh theo số iter/epoch thực tế. Bản `ratio 0.25` chạy trên server yếu.
- Kết quả v5 ratio 0.25 (người dùng gửi 2026-11-08, `last.pth.tar` epoch 22, `--ocpc_mode test`, mesh 14 khớp; Δ so v4 §16):

| Tập | n | MPJPE | PA | MPVPE | LIFT | Student − LIFT | Δ MPJPE vs v4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3DPW | 35,515 | 73.15 | 48.82 | 89.77 | 72.59 | +0.6 | −1.18 |
| OC (test) | 4,157 | 81.74 | 51.67 | 96.62 | 78.79 | +2.9 | +0.10 |
| PC (test) | 281 | 68.58 | 68.60 | 108.55 | 62.10 | +6.5 | −2.18 |
| Crowd | 1,923 | 80.39 | 56.50 | 103.87 | 70.90 | +9.5 | +2.24 |

  Chẩn đoán tầng (lift → token → ctx → mesh): 3DPW 72.59 → 72.44 → 73.18 → 73.15; OC 78.79 → 78.34 → 79.60 → 81.74; PC 62.10 → 66.19 → 65.80 → 68.58; Crowd: người dùng chưa gửi dòng chẩn đoán.
  Nhận xét: fusion hết mất trên 3DPW/OC (v4 mất 1.4 mm ở 3DPW); HyperGCN (token → ctx) tăng lỗi +0.7 (3DPW), +1.3 (OC); query → mesh ≈ 0 ở 3DPW nhưng +2.1 (OC), +2.8 (PC). Crowd tệ hơn v4 2.2 mm. So DPMesh: 3DPW 73.15 vs 73.6 (thắng 0.45), Crowd 80.39 vs 79.9 (thua 0.5).
- Kiểm tra trọng số (người dùng chạy `tools/check_hypergcn_scale.py`, 2026-11-08, `last` v4 & v5):
  - HyperGCN `layer_scale_1` (khởi tạo 1e-5): v4 ≈ 0.04–0.05, v5 ≈ 0.0125 → HyperGCN CÓ được dùng (không phải đồng nhất), v5 dùng ít hơn v4 ~3–4 lần.
  - BatchNorm1d (kênh = 17 khớp) v4: `running_var` tới 5.7e11 (block 0), 1.3e9 (block 1), `running_mean` tới −1e4 → thống kê BN bùng nổ; v5 bình thường (var 3–13). Giả thuyết (chưa kiểm chứng): `p_adj`/`b_adj` học tự do + `norm()` chia (tổng cột + 0.001) → tổng cột gần 0 làm nổ; BN test dùng running stats → lệch train/test ở một số khớp.
  - Nhánh khởi tạo 0 (`bone_proj`, `bone_geo_proj`, `jf_proj`, `depth_proj`): mean|w| ≈ 0.006–0.009 (tham chiếu lớp thường 0.024) → đều được học, độ lớn trọng số không đủ kết luận đóng góp → cần ablation.
- Kiểm tra lần 2 (script mở rộng: BN theo khớp + tổng cột `p_adj`/`b_adj`, người dùng chạy 2026-11-08) — **xác nhận nguyên nhân BN v4 nổ**:
  - v4: khớp BN nổ trùng đúng cột ma trận siêu cạnh có tổng ≈ 0 / âm (`norm()` nhân 1/(tổng+0.001), kỳ dị tại tổng = −0.001). Block 0: p_adj RWrist −0.0022 (×826), LWrist −0.0050, b_adj Neck −0.0073 → BN nổ RWrist (var 5.7e11), LWrist 7.4e4, Neck 1.9e3. Block 1: p_adj Head −0.0009 (×8959) → BN Head 1.3e9; RWrist 9.2e7. Block 2: b_adj LKnee +0.0016 (×390) → BN LKnee 6.5e4. Khởi tạo tổng cột ≥ 0.69.
  - v5: tổng cột ≥ 0.07 (hệ số ≤ 14), BN bình thường; trôi ma trận ~0.02 (v4 ~0.05). Nhưng b_adj Neck block 2 đã trôi 0.87 → 0.07 → vẫn có nguy cơ nếu train lâu hơn.
  - ~50% phần tử ma trận thành âm ở cả v4/v5 (khởi tạo không âm) → siêu đồ thị học tự do không còn giữ cấu trúc giải phẫu.
  - Hệ quả: so sánh v4 ↔ v5 bị nhiễu vì HyperGCN v4 hỏng số học ở cổ tay/đầu/cổ/gối. Kiến trúc HyperGCN giống hệt HyperDiff (người dùng muốn giữ); vấn đề nằm ở việc ma trận được học tự do không ràng buộc.

## 18. Đo tương quan độ bất định – lỗi từng khớp (2026-11-08, người dùng xác nhận, đã code, chưa chạy)
- Mục đích: kiểm tra tiền đề ý tưởng "hypergraph dẫn dắt bởi độ bất định" (đề xuất contribution, giả thuyết) trước khi code Student mới.
- Script mới `main/corr_uncert_lifter.py` (chỉ đánh giá): Refiner dày + ImgLifter Markov v1, K=5 S=2 như Student. Tín hiệu: `log_b` Refiner, spread K giả thuyết (mm), kết hợp (rank TB), detector sót. Lỗi: 3D, 3D PA, 2D Refiner. Báo Spearman gộp / trong mẫu / trong loại khớp, AUROC khớp xấu (top 20% e3d_pa), lỗi theo quintile, bảng theo khớp; lưu `.npz`.
- Ngưỡng đọc (đặt trước): Spearman trong mẫu ≥ ~0.4 và AUROC ≥ ~0.7 → tín hiệu đủ tốt.
- Kết quả (người dùng gửi 2026-11-08, K=5 S=2, OC/PC chỉ seq test). Tín hiệu kết hợp (comb) / spread / log_b / detector sót:

| Tập | n | [B] comb vs e3d | [B] comb vs e3d_PA | [C] comb vs e3d_PA | AUROC comb | AUROC spread | AUROC log_b | AUROC sót | e3d_PA quintile thấp→cao (comb) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3DPW | 35,515 | 0.55 | 0.32 | 0.30 | 0.73 | 0.71 | 0.68 | 0.51 | 28.3 → 72.8 |
| OC (test) | 4,157 | 0.52 | 0.34 | 0.40 | 0.77 | 0.74 | 0.72 | 0.48 | 24.6 → 74.6 |
| PC (test) | 281 | 0.68 | 0.69 | 0.04 | 0.83 | 0.87 | 0.66 | 0.43 | 23.2 → 88.5 |
| Crowd | 1,923 | 0.50 | 0.31 | 0.62 | 0.83 | 0.82 | 0.78 | 0.55 | 21.5 → 93.7 |

  - AUROC đạt ngưỡng ở mọi tập; lỗi nhóm bất định cao gấp 2.5–4 lần nhóm thấp. [B] vs e3d đạt (0.50–0.68) nhưng [B] vs e3d_PA chưa đạt (0.31–0.34, trừ PC n nhỏ).
  - Detector sót gần như vô dụng (AUROC ≈ 0.5; OC sót 46.7 vs có 45.4 mm) → Refiner đã lấp khớp sót tốt; "khớp bị che" ≠ "khớp sót detector".
  - log_b gần như không xếp hạng được lỗi 2D trong mẫu ([B] 0.10–0.16) → giải thích khả dĩ vì sao cổng b của Markov không giúp khớp che (§13). spread tốt hơn log_b.
  - Lưu ý gây nhiễu: spread và e3d root-relative cùng tăng theo khoảng cách tới root (Pelvis spread = 0) → [B] vs e3d bị thổi phồng. Cần [B] sau khi chuẩn hóa theo loại khớp (tính lại từ .npz).
- Bước tiếp (2026-11-08, đã code, chưa chạy): `tools/analyze_corr_npz.py` (chỉ CPU, đọc .npz) — đổi tín hiệu/lỗi thành phân vị TRONG từng loại khớp (bỏ Pelvis) để loại nhiễu loại khớp; báo [B*] trong mẫu, [D*] AUROC khớp xấu bất thường, precision@3 (ngẫu nhiên 0.188). Ngưỡng đặt trước: [B*] ≥ ~0.3, [D*] ≥ ~0.7.
- Kết quả phân tích chuẩn hóa theo loại khớp (người dùng chạy `tools/analyze_corr_npz.py`, 2026-11-08, 16 khớp bỏ Pelvis):

| Tập | [B*] spread vs e3d / e3d_PA | [B*] comb vs e3d_PA | [D*] AUROC comb vs e3d_PA | P@3 comb vs e3d_PA (ngẫu nhiên 0.188) |
|---|---|---:|---:|---:|
| 3DPW | 0.19 / 0.10 | 0.12 | 0.66 | 0.27 |
| OC | 0.14 / 0.05 | 0.06 | 0.68 | 0.25 |
| PC (n=281) | 0.01 / 0.04 | 0.04 | 0.49 | 0.22 |
| Crowd | 0.14 / 0.07 | 0.07 | 0.81 | 0.23 |

  - **Kết luận (theo ngưỡng đặt trước): KHÔNG ĐẠT.** [B*] 0.05–0.19 ≪ 0.3 → sau khi bỏ ảnh hưởng loại khớp, tín hiệu bất định gần như KHÔNG chỉ ra được khớp nào trong cùng một người tệ bất thường. [D*] vẫn 0.66–0.81 (trừ PC) → tín hiệu chủ yếu phản ánh độ khó CẤP MẪU (ảnh khó → mọi khớp cùng bất định, cùng sai), không phải cấp khớp.
  - → Tiền đề của "hypergraph truyền tin từ khớp tin cậy sang khớp không tin cậy trong cùng người" không được ủng hộ với tín hiệu hiện có. log_b còn kém hơn spread.
  - Lưu ý chưa loại trừ: spread ước lượng từ K=5 nên nhiễu ở cấp khớp; chưa thử K lớn.

## 19. Cận trên IK: pose lifter → IK giải tích (HybrIK) → mesh (2026-11-08, người dùng xác nhận, đã code, chưa chạy)
- Mục đích: trước khi làm Student v6 (mesh neo vào hình học qua IK), đo xem chỉ IK từ pose lifter (không học gì) đạt bao nhiêu so với LIFT và Student v5.
- File mới (không sửa file cũ, không key config): `lib/models/hybrik_lbs.py` (chép nguyên văn ARTS lbs.py), `lib/models/hybrik_smpl.py` (`HybrIKSMPL`, dựng buffer từ lớp smplpytorch sẵn có, 29 điểm = 24 khớp SMPL + 5 đỉnh đầu chi, chỉ nhánh test), `tools/fit_h36m_to_smpl29.py` (khớp M 29×17 bằng bình phương tối thiểu trên mesh GT train Human36M+MuCo, không 3DPW), `main/ik_upper_bound.py` (biến thể LIFT / A_gt29 / B_M_gt17 / C_M_lift; β trung bình, twist 0; đo 14 khớp + MPVPE như dataset.evaluate).
- Đọc: C ≈ LIFT → IK giữ thông tin, làm v6; A/B lớn → lỗi do β/twist hoặc ánh xạ 17→29. Kết quả: chưa có.
- Bước 1 (người dùng chạy 2026-11-08): M khớp trên 40k mẫu (Human36M + MuCo 20k mỗi bộ, val 10%), lỗi TB 33.0 mm. Điểm có tương ứng H36M: gối 4 mm; pelvis 8.5; collar/neck/head/head_top 9–12; vai 12–13; hông 17; cổ chân 17–22; khuỷu 18–23; cổ tay 16–26. Điểm H36M không có: bàn tay 35–43, bàn chân 85–88, ngón giữa 78–81, ngón chân cái 129–132 mm. Giả thuyết: lệch hông/khuỷu/cổ tay là offset theo hệ trục cục bộ của chi (ánh xạ tuyến tính toàn cục không biểu diễn được); bàn tay/bàn chân không có thông tin hướng trong H36M-17 → xoay cổ tay/cổ chân sai → ảnh hưởng chủ yếu MPVPE.
- Bước 2 (người dùng chạy 2026-11-08, `ik_upper_bound.py`, K=5 S=2, β trung bình, twist 0, KHÔNG train). MPJPE / PA / MPVPE (14 khớp):

| Tập | LIFT | A_gt29 | B_M_gt17 | **C_M_lift (không học)** | Student v5 (`last`) |
|---|---|---|---|---|---|
| 3DPW | 72.53 / 47.35 | 24.52 / 12.64 / 29.02 | 25.26 / 15.88 / 36.32 | **73.18 / 49.23 / 90.59** | 73.15 / 48.82 / 89.77 |
| OC (test) | 78.67 / 48.98 | 31.99 / 14.84 / 36.02 | 31.69 / 18.55 / 43.75 | **82.02 / 51.60 / 96.85** | 81.74 / 51.67 / 96.62 |
| PC (test) | 62.19 / 64.60 | 19.04 / 16.74 / 28.40 | 23.49 / 20.62 / 41.10 | **64.19 / 67.97 / 103.61** | 68.58 / 68.60 / 108.55 |
| Crowd | 70.89 / 51.56 | 31.42 / 12.51 / 36.12 | 31.03 / 15.13 / 44.06 | **77.56 / 54.08 / 101.08** | 80.39 / 56.50 / 103.87 |

  - **Kết luận chính:** lifter + M + IK, KHÔNG học gì, đã ngang Student v5 ở 3DPW/OC (±0.3) và TỐT HƠN ở PC (−4.4) và Crowd (−2.8 MPJPE, −2.4 PA, −2.8 MPVPE). → Toàn bộ phần học của Student v5 (backbone, fusion, HyperGCN, query) hiện không thêm giá trị so với baseline không học.
  - Ánh xạ M gần như không làm hỏng MPJPE (A ≈ B, B kém hơn ~3 PA, ~7–8 MPVPE). Sàn A (~25–32 mm) chủ yếu do β trung bình + twist 0 (+ khác định nghĩa khớp) → phần có thể học (β, twist, bàn tay/chân).
  - C vẫn kém LIFT: 3DPW +0.7, OC +3.4, PC +2.0, Crowd +6.7 → IK làm mất thông tin nhiều nhất ở Crowd. Giả thuyết (chưa kiểm chứng): hướng pelvis (SVD từ hông/spine nhiễu) và β trung bình lệch nhiều ở Crowd.
  - Lưu ý so sánh với SOTA: lifter có dùng 3dpw-train.

## 20. Student v6: mesh neo vào hình học qua IK (2026-11-08, người dùng xác nhận, đã code, chưa chạy)
- Lý do: §19 — lifter + M + IK không học đã ngang/hơn Student v5 → cho Student xuất phát từ baseline này.
- Luồng (thay bước query → 6D): pose17 = lift + Δ17(ctx) (→ `ctx_3d`, loss `v5_ctx_loss_w`) → J29 = M·pose17 + Δ29(token query; 5 điểm đầu chi lấy token khớp cha) → twist = (1,0) + Linear(token query 1..23) → `HybrIKSMPL.ik` (β = shape head) → rotmat → axis-angle (`rotmat_to_aa_safe`, tránh NaN) → `get_coord` cũ. Mọi head khởi tạo 0 → lúc đầu = baseline C (§19). 5 loss cũ giữ nguyên; thêm `v6_j29_loss_w` (smooth-L1 J29 vs GT29 từ mesh GT, gốc Pelvis H36M, mask is_valid_fit).
- Key mới (mặc định tắt): `MODEL.v6_ik`, `v6_m_path`, `v6_delta17/29`, `v6_twist`, `v6_j29_loss_w`, `hgcn_freeze_adj` (requires_grad=False cho p_adj/b_adj, không sửa hypergraph.py). `bone_proj` (v4) vẫn cộng vào query; `per_joint_head` không dùng khi v6.
- File sửa (backup `_to_delete/bak_student_v6/`): `config.py`, `teacher_student.py`, `ARTS.py`, `base.py` (`_v6_j29_loss`, log `j29` mm), `hybrik_smpl.py` (`ik()`, `rotmat_to_aa_safe`, buffer không lưu ckpt). `v6_M` lưu trong ckpt.
- Config: `config/train_student_v6_ik.yml` (exp `student_v6_ik`) = v5 + v6_ik + hgcn_freeze_adj, tắt v5_bone_query; `config/train_student_v6_ik_lite.yml` (exp `student_v6_ik_lite`) = v6_ik + tắt [B] [B'] [C] [E]. Cùng ratio 0.25, 22 epoch như v5.
- Mốc phải vượt: baseline C (3DPW 73.18, OC 82.02, PC 64.19, Crowd 77.56). Kết quả: chưa có.
- Log train v6_ik (người dùng gửi 2026-11-08 tối, đang ở epoch 17/22; eval mỗi epoch CHỈ để theo dõi, không chọn ckpt):
  - 3DPW ep1–16: 74.4–77.9 (ep16 75.64 / 49.66 / 92.10); OC 83.8–87.5 (ep16 85.06); Crowd 77.6–80.9 (ep16 77.61); PC 61.0–73.2 (nhiễu, n=281). → Kém baseline IK không học (3DPW 73.18, OC 82.02) và kém v5 ở 3DPW/OC; Crowd ≈ baseline (77.56).
  - Loss train: `ctx 3D` (pose17 = lift + Δ17) đứng yên 0.0186 → 0.0182 suốt 16 epoch; lỗi lifter train 65 mm không đổi; `privileged_3d` (giải mã trực tiếp từ token) giảm về 0.019 ≈ ctx → token chỉ tái tạo được pose lifter, không sửa được. J29 lỗi 93.6 → 89.3 mm (giảm rất chậm).
  - Giả thuyết (chưa kiểm chứng): (H1) Δ17/Δ29 gần như không học → mesh ≈ IK(M·lift, β dự đoán, twist học) ; phần kém hơn baseline (~2.5 mm 3DPW) đến từ β dự đoán (độ dài xương) và/hoặc twist, vì baseline dùng β trung bình + twist 0. (H2) Đặc trưng ảnh của Student không chứa thêm thông tin để sửa pose lifter (lifter đã dùng ảnh) — cả trên tập train.
- Log train v6_ik_lite (người dùng gửi 2026-11-08 tối, tới epoch 11/22; theo dõi): 3DPW ep1–11 74.2–77.7 (ep11 75.65 / 49.92 / 92.76), OC 83.4–87.2 (ep11 84.14), Crowd 77.9–82.5 (ep11 78.30), PC 60.9–67.3. Cùng ep11 bản đầy đủ: 3DPW 76.11, OC 85.62, Crowd 80.86 → lite không kém hơn (thậm chí nhỉnh hơn), nhanh hơn (13 vs 20 phút/epoch). Cùng hiện tượng: ctx 3D đứng yên 0.0182–0.0185, J29 ~90–92 mm, privileged ≈ ctx → củng cố H2 (Student không sửa được pose lifter, kể cả trên train). Chưa phải kết quả cuối.

- **Kết quả cuối v6_ik** (người dùng gửi 2026-11-09, eval epoch 22 = `last` trong log train, OC/PC chỉ seq test): 3DPW **75.53 / 49.60 / 92.00**, OC 84.64 / 51.84 / 98.87, PC 66.11 / 66.32 / 104.03, Crowd 78.50 / 55.43 / 101.45. Train ep22: ctx 3D 0.0183, privileged 0.0185 (≈ lifter 65.4 mm), J29 88.6 mm.
  - So baseline C không học (73.18 / 82.02 / 64.19 / 77.56): **kém hơn điểm xuất phát** +2.4 (3DPW), +2.6 (OC), +1.9 (PC), +0.9 (Crowd). So v5: kém 3DPW (+2.4), OC (+2.9); tốt hơn PC (−2.5), Crowd (−1.9).
  - Chẩn đoán (kết hợp §21): Δ17/Δ29 không học (Student không có thông tin ngoài lifter); β lấy từ shape head (học theo pseudo-GT) làm sai độ dài xương → IK dựng khung sai → kém cả β trung bình. Nguồn shape đúng là độ dài xương của lifter (fit §21: 69.9 → 62.3 ≈ LIFT trên val).

## 21. Fitting 2D dày lúc test: tách pose / shape (2026-11-08, người dùng xác nhận, đã code, chưa chạy)
- Lý do: survey (HybrIK/NIKI: pose qua khớp + IK; Virtual Markers / CameraHMR / NLF: shape cần điểm BỀ MẶT, khớp chỉ cho cơ thể "trung bình"; ScoreHMR: fitting 2D cần prior mạnh). Ta đã có 48 đỉnh mesh 2D + b từ Refiner nhưng chưa dùng cho shape; Student không sửa được pose lifter (§20).
- Script mới `main/fit_dense2d.py` (không train, không sửa code cũ): khởi tạo = baseline C (M·lift → IK, β 0, twist 0) → giai đoạn 1 tối ưu t + hướng toàn cục, giai đoạn 2 tối ưu θ (khớp 0..21) + β + t. Loss: Huber 2D (px crop 256, trọng số 1/b² chuẩn hóa) của 17 khớp [+ 48 đỉnh], prior 3D |khớp mesh − lift| (mm), β², (θ−θ0)². Chiếu bằng camera ảo của mô hình (focal 5000, crop 256) sau khi xoay hệ camera thật → camera ảo bằng view_rot từ cam_bbox (tắt bằng `--no_view_rot`). Biến thể: `init_IK` (= C), `j17`, `dense`.
- Giao thức: chọn trọng số trên **3dpw-val** (lifter/Student không dùng), sau đó mới chạy test. Kết quả: chưa có.
- Lần chạy 1 (người dùng, 2026-11-08, 3dpw-val 2000 mẫu, mặc định w2d 1 huber 5px w3d 0.1 wbeta 1 wpose 50, view_rot): init_IK 69.93 / 51.24 / 88.80; j17 **65.76 / 49.24 / 85.18** (−4.2 / −2.0 / −3.6), lỗi 2D 2.49 px; dense 65.92 / 49.37 / 84.84, lỗi 2D 3.11 px. **Lỗi code:** |β| = 0.00 vì smplpytorch dùng β nội bộ khi norm(β) = 0 → β không có gradient, shape KHÔNG được tối ưu → so sánh j17/dense chưa đo được đóng góp cho shape. Đã sửa: khởi tạo β = 1e-4. Cần chạy lại.
- Lần chạy 2 (sau khi sửa β, người dùng 2026-11-08, 3dpw-val 2000, cùng trọng số mặc định): init_IK 69.93 / 51.24 / 88.80; **j17 62.64 / 48.53 / 81.75** (−7.3 / −2.7 / −7.1), |β| 0.71; dense 62.70 / 48.56 / **81.28**, |β| 0.73. → Fitting β + θ + t lúc test giảm mạnh cả MPJPE và MPVPE; 48 đỉnh dày chỉ thêm −0.5 MPVPE. Giả thuyết (chưa kiểm chứng): phần lớn lợi ích đến từ β khớp độ dài xương của pose lifter (prior 3D) chứ không từ 2D → cần ablation `--w2d 0`.
- Ablation A (người dùng 2026-11-08, 3dpw-val 2000, `--w2d 0 --variants j17`: chỉ prior 3D + β + θ): **62.31 / 48.29 / 81.31**, |β| 0.81 — bằng/nhỉnh hơn bản có 2D (62.64 / 48.53 / 81.75). → **Toàn bộ lợi ích đến từ việc khớp SMPL (β, θ) vào pose 3D của lifter**, 2D Refiner không thêm gì. IK baseline chỉ dùng HƯỚNG xương + β trung bình nên bỏ phí độ dài xương của lifter. Chưa biết LIFT trên val (cần so để biết fit có vượt chính lifter không).
- LIFT trên val (người dùng chạy `ik_upper_bound.py --sets 3dpw-val --max_samples 2000`): LIFT **62.02 / 47.92**; C_M_lift 69.95 / 51.25 / 88.74 (IK mất 7.9 mm trên val, so với chỉ 0.65 trên test 3DPW, 6.7 trên Crowd). → Fit (A) 62.31 ≈ LIFT: fitting chỉ **giữ lại** thông tin lifter mà IK (β TB, chỉ hướng xương) làm mất, KHÔNG vượt lifter. Kết luận: theo hướng này, mesh MPJPE bị chặn trên bởi LIFT; muốn vượt SOTA phải cải thiện lifter (pose dưới che khuất).
- **CẢNH BÁO giao thức:** 3DPW-Crowd được lấy từ split VALIDATION của 3DPW (`3dpw-crowd` dùng `3DPW_latest_validation.json`, khác nguồn 2D: HHRNet) → `3dpw-val` CHỒNG LẤN với Crowd test. Không được chọn trọng số / checkpoint trên `3dpw-val` nếu báo Crowd. Dùng trọng số mặc định (chưa tinh chỉnh) hoặc lọc bỏ các sequence của Crowd khỏi val.

## 22. Student v7: mesh neo theo pose lifter, Student chỉ đoán bề mặt (2026-11-09, người dùng duyệt, đã code, chưa chạy)
- Lý do (lý thuyết + §20–21): (1) cùng nguồn thông tin → trung bình posterior của lifter đã tối ưu cho pose, Student không sửa được (Δ17 đứng yên); (2) mesh = swing + độ dài xương (có trong pose3d) + twist + shape mềm/đầu chi (KHÔNG có trong khớp 3D → cần ảnh); (3) residual khởi tạo 0 → không thể kém điểm neo. Fit §21: β từ xương lifter đưa mesh ≈ LIFT.
- Luồng: pose3d + ảnh → fusion → HyperGCN (khóa ma trận) → ctx + token siêu cạnh (`part_tok` 10 đoạn chi, `body_tok` 5 chi/thân, đọc bằng ma trận thuộc cùng nhóm khớp hypergraph.py) → giải mã: pose thân = pose3d (KHÔNG sửa), J29 = M·pose3d + Δ 7 điểm đầu chi {10,11,22,23,24,27,28} (query + body_tok); twist khớp 1..23 từ (query + part_tok của đoạn chi chứa xương); β = β_xương (Gauss-Newton 3 bước trên 16 xương H36M T-pose, không tham số) + Δβ (shape token + TB body_tok); camera như cũ → IK → SMPL. Mọi head khởi tạo 0.
- Loss: mesh, khớp, chiếu 2D, smpl_shape; J29 chỉ 7 điểm đầu chi (`v7_j29_loss_w`); `v7_dbeta_w`·TB(Δβ²). BỎ loss tham số pose (`v7_no_pose_param_loss`, pop trước AWL), privileged (`v7_no_privileged`), ctx_3d.
- Key mới (`config.py`, mặc định tắt): `v7_anchor`, `v7_part_readout`, `v7_twist`, `v7_delta_limb`, `v7_dbeta`, `v7_delta17` (ablation), `v7_no_hgcn` (ablation), `v7_bone_gn_iters`, `v7_bone_beta_reg` (1e-4), `v7_dbeta_w`, `v7_j29_loss_w`, `v7_no_pose_param_loss`, `v7_no_privileged`. Dùng chung `v6_m_path`, `hgcn_freeze_adj`.
- File sửa (backup `_to_delete/bak_student_v7/`): `config.py`, `teacher_student.py` (hằng số V7_*, nhánh `_v7_outputs`, cờ bỏ HyperGCN), `ARTS.py`, `hybrik_smpl.py` (`beta_from_bones`), `base.py` (J29 theo chỉ số, Δβ, bỏ privileged / pose param).
- Config `config/train_student_v7_anchor.yml` (exp `student_v7_anchor`) = lite v6 + v7. Ablation dự kiến: `v7_no_hgcn`, `v7_part_readout False`, `v7_delta17 True`. Kết quả: chưa có.
- **Kết quả v7_anchor** (người dùng gửi 2026-11-10, eval epoch 22 = `last` trong log train, OC/PC chỉ seq test), MPJPE / PA / MPVPE:

| Tập | v7 (`last`) | LIFT | Baseline IK (C) | v5 | v6_ik | DPMesh |
|---|---|---|---|---|---|---|
| 3DPW | **73.46 / 48.17 / 90.61** | 72.53 / 47.36 | 73.18 / 49.23 / 90.59 | 73.15 / 48.82 / 89.77 | 75.53 / 49.60 / 92.00 | 73.6 / 47.4 / 90.7 |
| OC (test) | **80.43 / 50.57 / 95.14** | 78.67 / 48.98 | 82.02 / 51.60 / 96.85 | 81.74 / 51.67 / 96.62 | 84.64 / 51.84 / 98.87 | (70.9, toàn bộ seq) |
| PC (test) | **61.85 / 64.71 / 98.91** | 62.19 / 64.60 | 64.19 / 67.97 / 103.61 | 68.58 / 68.60 / 108.55 | 66.11 / 66.32 / 104.03 | (82.2, toàn bộ seq) |
| Crowd | **71.72 / 52.49 / 94.87** | 70.89 / 51.47 | 77.56 / 54.08 / 101.08 | 80.39 / 56.50 / 103.87 | 78.50 / 55.43 / 101.45 | 79.9 / 51.1 / – |

  - So v5: 3DPW ≈ (+0.3 MPJPE, −0.7 PA, +0.8 MPVPE); OC −1.3 / −1.1 / −1.5; PC −6.7 / −3.9 / −9.6; Crowd **−8.7 / −4.0 / −9.0**. So baseline IK: Crowd −5.8 / −1.6 / −6.2, PC −2.3 / −3.3 / −4.7, OC −1.6 / −1.0 / −1.7, 3DPW ≈. So LIFT: MPJPE +0.9 (3DPW), +1.8 (OC), −0.3 (PC), +0.8 (Crowd). → Bản Student tốt nhất tới nay; lợi ích lớn nhất ở che khuất / đông người. So DPMesh: 3DPW MPJPE ≈ (−0.1), MPVPE ≈, PA kém 0.8; Crowd MPJPE −8.2 nhưng PA kém 1.4. Lưu ý: lifter dùng 3dpw-train.
  - Log train: |β0| = 2.00 ổn định (β từ xương, lifter đóng băng); TB Δβ² tăng 0.19 → 0.38 (|Δβ| ~ 1.9, cùng cỡ β0); J29 7 điểm đầu chi 148 → 138 mm (giảm chậm). Giả thuyết (chưa kiểm chứng): β0 quá lớn / nhiễu do λ = 1e-4 yếu (16 xương, 10 tham số), Δβ đang bù ngược → có thể là lý do MPVPE 3DPW không giảm so với baseline IK và MPJPE còn cách LIFT ~1 mm.
- Test chính thức v7 `last` (người dùng chạy `test_student_3dpw_sets.py --ocpc_mode test`, 2026-11-10): **trùng khớp** eval epoch 22 — 3DPW 73.46 / 48.17 / 90.61 (LIFT 72.59 / 47.36), OC 80.43 / 50.57 / 95.14 (LIFT 78.79 / 49.08), PC 61.85 / 64.71 / 98.91 (LIFT 62.10 / 64.48), Crowd 71.72 / 52.49 / 94.87 (LIFT 70.90 / 51.47). Mesh − LIFT: +0.87 / +1.64 / −0.25 / +0.82 MPJPE; PA +0.81 / +1.49 / +0.23 / +1.02. Cột token ~670–740 mm là bình thường (privileged không train ở v7).
- **B (2026-11-10, người dùng duyệt, chưa chạy):** `config/train_student_v7_anchor_breg.yml` (exp `student_v7_anchor_breg`) = v7 chỉ đổi `v7_bone_beta_reg` 1e-4 → 1e-3 (cùng bậc J^T J; chọn theo lý thuyết). Kỳ vọng (giả thuyết): |β0| ~1, Δβ² không tăng, mesh sát LIFT hơn, MPVPE 3DPW giảm. Kết quả: chưa có.
  - **Log breg (người dùng gửi 2026-11-10 23:11, đang train, mới tới epoch 17/22 — CHƯA phải `last`):** log train |β0| = 0.68 (v7: 2.00); TB Δβ² 0.09 → 0.06–0.08, không tăng (v7: 0.19 → 0.38); J29 7 điểm đầu chi 148 → 138 mm (giống v7). Eval epoch 17 (MPJPE / PA / MPVPE): 3DPW 73.38 / 48.22 / 90.09, OC 80.46 / 50.60 / 94.98, PC 61.77 / 65.26 / 99.05, Crowd 72.28 / 52.51 / 95.29. So với v7 `last`: chênh ≤ 0.6 mm ở mọi tập (3DPW MPVPE −0.5, Crowd MPJPE +0.6). Nhận xét sơ bộ: vế "|β0| giảm, Δβ không bù ngược" của giả thuyết B ĐÚNG, nhưng vế "mesh sát LIFT hơn / MPVPE giảm" CHƯA thấy → độ lớn β0 có lẽ không phải nguyên nhân khoảng cách ~1 mm tới LIFT. Chờ `last` (epoch 22) để kết luận.
  - **Kết quả breg `last`** (eval epoch 22, người dùng gửi 2026-11-11) và **test chính thức** (`test_student_3dpw_sets.py --ocpc_mode test`, 2026-11-11) **trùng khớp**, MPJPE / PA / MPVPE: 3DPW 73.32 / 48.16 / 90.11 (LIFT 72.59 / 47.36), OC 80.54 / 50.60 / 95.45 (LIFT 78.79 / 49.08), PC 61.56 / 64.96 / 100.09 (LIFT 62.10 / 64.48, n = 281), Crowd 72.30 / 52.53 / 95.18 (LIFT 70.90 / 51.47). Log: |β0| 0.69, TB Δβ² 0.085, J29 đầu chi 137.8 mm.
  - So v7 `last`: MPJPE −0.14 / +0.11 / −0.29 / +0.58, MPVPE −0.50 / +0.31 / +1.18 / +0.31 → ngang nhau (trong dao động giữa epoch; PC MPVPE ep21 97.85 vs ep22 100.09). Mesh − LIFT (test): MPJPE +0.73 / +1.75 / −0.54 / +1.40; PA +0.80 / +1.52 / +0.48 / +1.06 — gần như y hệt v7.
  - Kết luận: giả thuyết B đúng một nửa — β0 nhỏ lại, Δβ hết bù ngược, nhưng metric KHÔNG cải thiện → độ lớn β0 không phải nguyên nhân khoảng cách tới LIFT. Bản chính chọn KHÔNG theo test (đề xuất breg vì β có ý nghĩa vật lý hơn; chờ người dùng quyết).
  - **breg `last` trên TOÀN BỘ OC / PC** (`--ocpc_mode both`, người dùng gửi 2026-11-11; RÒ RỈ vì lifter train 3dpw-train → KHÔNG đưa vào bảng paper), MPJPE / PA / MPVPE (LIFT / LIFT-PA): OC toàn bộ (n 20243) 52.15 / 37.04 / 67.36 (48.44 / 34.19) vs chỉ seq test (n 4157) 80.54 / 50.60 / 95.45 (78.79 / 49.08); PC toàn bộ (n 2218) 74.23 / 58.53 / 99.51 (72.36 / 57.49) vs chỉ seq test (n 281) 61.56 / 64.96 / 100.09 (62.10 / 64.48).
    - OC: rò rỉ rất nặng (LIFT 78.8 → 48.4) → số toàn bộ vô nghĩa. PC: LIFT toàn bộ (72.4) còn KÉM phần test (62.1) → phần test của PC là tập con dễ / quá nhỏ; mức rò rỉ trên PC chưa tách được (cần tách theo split train/val/test).
    - Mesh − LIFT: OC toàn bộ +3.71 (seq test +1.75), PC toàn bộ +1.87 (seq test −0.54). Giả thuyết: lifter càng chính xác (nhớ mẫu train) thì Student (không train 3dpw) càng làm xấu thêm → Student/IK đang cộng một sai số nền vài mm, không bám được lifter khi lifter rất tốt.

## 23. Ablation D: lifter + β_xương + IK, KHÔNG học (2026-11-11, người dùng duyệt, đã chạy)
- Mục đích: tách lợi ích của Student so với baseline IK (C, β TB, twist 0) thành phần do β_xương (Gauss-Newton, không học) và phần do các thành phần HỌC (HyperGCN, twist, Δβ, Δ đầu chi).
- Code: `main/ik_upper_bound.py` thêm biến thể `D_M_lift_bb` (β = `beta_from_bones(lift, 3, 1e-3)`, twist 0) + `--bone_gn_iters`, `--bone_beta_reg` (backup `_to_delete/bak_ik_upper_bound/`). Lifter Markov v1, K5 S2, OC/PC chỉ seq test.
- Kết quả (người dùng gửi 2026-11-11), MPJPE / PA / MPVPE:

| Tập | LIFT | C (β TB) | D (β_xương) | Student breg `last` |
|---|---|---|---|---|
| 3DPW | 72.53 / 47.35 | 73.18 / 49.23 / 90.59 | 74.53 / 49.17 / 91.74 | 73.32 / 48.16 / 90.11 |
| OC (test) | 78.67 / 48.98 | 82.02 / 51.60 / 96.85 | 83.21 / 51.77 / 98.03 | 80.54 / 50.60 / 95.45 |
| PC (test, n 281) | 62.19 / 64.60 | 64.19 / 67.97 / 103.61 | 64.76 / 67.32 / 104.10 | 61.56 / 64.96 / 100.09 |
| Crowd | 70.89 / 51.56 | 77.56 / 54.08 / 101.08 | 73.17 / 53.99 / 97.09 | 72.30 / 52.53 / 95.18 |

- Phân rã C → Student (MPJPE / PA / MPVPE):
  - β_xương (C → D): 3DPW +1.35 / −0.06 / +1.15; OC +1.19 / +0.17 / +1.18; PC +0.57 / −0.65 / +0.49; Crowd **−4.39 / −0.09 / −3.99**.
  - Phần học (D → Student): 3DPW **−1.21 / −1.01 / −1.63**; OC **−2.67 / −1.17 / −2.58**; PC **−3.20 / −2.36 / −4.01**; Crowd **−0.87 / −1.46 / −1.91**.
- Kết luận (dữ kiện): phần học của Student đóng góp NHẤT QUÁN trên cả 4 tập (MPJPE −0.9…−3.2, PA −1.0…−2.4, MPVPE −1.6…−4.0) → contribution đứng vững. NHƯNG lợi ích lớn ở Crowd so với C chủ yếu do β_xương (−4.4 trong −5.3 MPJPE), không phải phần học → paper phải báo trung thực. β_xương một mình làm XẤU 3DPW / OC / PC (+0.6…+1.4) nhưng tốt cho Crowd.
- Giả thuyết (chưa kiểm chứng): một phần lợi ích "phần học" có thể chỉ là Δβ sửa lại β_xương xấu (vì D kém C ở 3 tập) → ablation tắt Δβ (`v7_dbeta False`) / tắt twist / tắt Δ đầu chi sẽ tách tiếp.

## 24. Dọn code: v7 (breg) là phiên bản Student CHÍNH THỨC DUY NHẤT (2026-11-11, người dùng duyệt phạm vi: chỉ code kiến trúc + YAML)
- Từ đây chỉ còn 1 Student = v7 anchor với `v7_bone_beta_reg = 1e-3` (breg). Config chính thức: `config/train_student.yml` (nội dung = `train_student_v7_anchor_breg.yml`, chỉ đổi header). Ablation = chép file này, đổi 1 key `v7_*`.
- `lib/models/teacher_student.py` (1060 → 613 dòng): chỉ còn `Student` (gộp FusionPose2Mesh). Bỏ Teacher, joint_encoder 'gt', spread gate, hop bias, [B] [C] [D] [E], img probe, v5 bone_query, v6 IK, nhánh góc 6D (`per_joint_head`), `shape_head` (v7 không dùng). Tên module / tham số giữ nguyên → ckpt v7 cũ nạp được; key `per_joint_head.*`, `shape_head.*` thêm vào `LEGACY_KEY_PARTS` (bỏ qua khi nạp).
- `lib/models/ARTS.py` (312 → 207 dòng): chỉ đường Refiner dày + ImgLifter → Student. Bỏ Teacher, DiffLifter trong ARTS, spread gate, F16. GIỮ `make_heatmaps`, `build_diff_lifter`, `diff_lift` (Refiner / script eval / tools còn import) và `privileged_joint_head` (base.py còn đọc `privileged_3d`, loss đã tắt) — dọn khi tới base.py.
- YAML: giữ 3 file (`train_student.yml`, `train_refiner2d_jotr_mesh_dense48_scratch.yaml`, `train_img_lifter_dense48_3dpw_markov.yaml`); 40 YAML cũ chuyển vào `_to_delete/bak_cleanup_v7_20261111/config/`. Backup 2 file model: `_to_delete/bak_cleanup_v7_20261111/lib_models/`.
- NGOÀI phạm vi (chưa dọn): `base.py` (Teacher_Trainer / KD / loss cũ — Teacher_Trainer giờ báo lỗi rõ ràng khi gọi), `config.py`, `main/`, `tools/`. Các script cũ dùng Teacher / YAML cũ (analyze_teacher, inspect_student_spread, oracle_best_hypothesis, diag_*, ...) sẽ không chạy được nữa.
- Lưu ý: bỏ module thừa làm thay đổi thứ tự khởi tạo ngẫu nhiên → train lại cùng YAML sẽ KHÔNG trùng bit với run breg cũ (kết quả kỳ vọng tương đương).
- Kiểm tra cần chạy (người dùng): (1) test ckpt breg `last` bằng `config/train_student.yml` phải ra đúng 73.32 / 48.16 / 90.11 … và log nạp chỉ "thừa" key `per_joint_head` / `shape_head`, "thiếu" 0; (2) train thử 1 epoch không lỗi (kể cả lúc lưu ckpt). Kết quả: (1) ĐẠT (người dùng gửi 2026-11-11 14:05) — code mới + `config/train_student.yml` test ckpt breg `last` ra TRÙNG KHỚP từng số: 3DPW 73.32 / 48.16 / 90.11, OC 80.54 / 50.60 / 95.45, PC 61.56 / 64.96 / 100.09, Crowd 72.30 / 52.53 / 95.18, LIFT không đổi. (2) ĐẠT (người dùng xác nhận 2026-11-11 14:57: train thử chạy + lưu ckpt không lỗi). → Dọn code HOÀN TẤT.

## 25. Ablation HyperGCN (2026-11-11, người dùng yêu cầu, đã tạo YAML, chưa chạy)
- `config/train_student_abl_no_hgcn.yml` (exp `student_abl_no_hgcn`) = `train_student.yml`, CHỈ đổi `v7_no_hgcn` False → True: ctx = norm(joint_out) + node_pe, KHÔNG qua HyperGCN; `part_tok` / `body_tok` vẫn đọc từ ctx bằng ma trận thuộc (giữ readout) → chỉ đo đóng góp của truyền tin trên siêu đồ thị. Không sửa code.
- So với: Student breg `last` (3DPW 73.32 / 48.16 / 90.11, OC 80.54 / 50.60 / 95.45, PC 61.56 / 64.96 / 100.09, Crowd 72.30 / 52.53 / 95.18). Lifter cũ (Markov v1, có 3dpw-train) → so tương đối hợp lệ; số tuyệt đối phải chạy lại khi đổi lifter.
- Đọc kết quả: chênh lệch giữa các epoch ~±0.5 mm (PC MPVPE ±2) → chỉ coi HyperGCN có đóng góp nếu kém rõ hơn mức này trên nhiều tập. Kết quả: chưa có.
