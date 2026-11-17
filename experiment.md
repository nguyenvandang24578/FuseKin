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
- **Log no_hgcn (người dùng gửi 2026-11-11 23:17, mới tới epoch 13/22 — CHƯA phải `last`)**, MPJPE / PA / MPVPE, so cùng epoch với log breg:
  - ep13 no_hgcn: 3DPW 73.52 / 48.30 / 89.86, OC 80.47 / 50.63 / 94.78, PC 61.86 / 65.45 / 99.47, Crowd 72.25 / 52.33 / 95.11.
  - ep13 breg: 3DPW 73.72 / 48.30 / 91.15, OC 80.75 / 50.79 / 96.02, PC 62.34 / 65.24 / 101.05, Crowd 72.09 / 52.62 / 95.70.
  - TB ep11–13 (no_hgcn vs breg): 3DPW MPJPE 73.58 vs 73.56, MPVPE 89.95 vs 90.65; OC MPJPE 80.73 vs 80.68; Crowd MPJPE 72.30 vs 72.38. Log train: J29 đầu chi 138.5 mm, |β0| 0.68, Δβ² ~0.07 (giống breg).
  - Nhận xét sơ bộ: bỏ HyperGCN KHÔNG làm kém (≈, MPVPE 3DPW thậm chí tốt hơn ~0.7, trong dao động). Giả thuyết: (1) self-attn giữa các khớp trong 3 block fusion đã truyền tin giữa khớp → HyperGCN thừa; (2) cấu trúc xương đã nằm sẵn trong pose3d lifter. Chờ `last` (epoch 22) để kết luận.
- **Kiểm tra trọng số HyperGCN** (`tools/check_hypergcn_scale.py`, người dùng chạy 2026-11-12, ckpt breg `last` + v7 `last`, chỉ đọc trọng số):
  - layer_scale_1 mean|.| 2.6e-2 / 3.0e-2 / 3.4e-2 (block 0/1/2, breg; v7: 2.3e-2 / 2.4e-2 / 2.6e-2), max ~0.1–0.17; layer_scale_2 ~1e-2. Khởi tạo 1e-5 → tăng ~1000–3000 lần → HyperGCN ĐƯỢC HỌC, KHÔNG phải ánh xạ đồng nhất.
  - alpha part / body tăng (1.06–1.16), alpha joint ~0.98–1.0 → mixer ưu tiên nhánh siêu cạnh hơn nhánh khớp. BN khỏe (var 0.8–7.5, không nổ); p_adj / b_adj không trôi (khóa). bone_proj (khởi tạo 0) mean 5e-3, max 0.06 → query có dùng ctx. Hai run cho cùng bức tranh.
  - Kết luận: giả thuyết "HyperGCN không được train (≈ identity)" BỊ BÁC. HyperGCN có thay đổi ctx (ước lượng thô ~vài–10% mỗi block, giả thuyết — chưa đo đầu ra mixer) nhưng ablation (§25, ep13) cho thấy bỏ đi không kém → nghiêng về giả thuyết THỪA CHỨC NĂNG (self-attn khớp trong fusion + readout siêu cạnh làm được việc tương đương) / LỆCH NHIỆM VỤ (cấu trúc xương đã nằm trong pose3d lifter, phần còn lại là bề mặt cần ảnh).

## 26. HyperGCN đổi chỗ: SMPL-24 trên token query (2026-11-12, người dùng duyệt, đã code, chưa chạy)
- Lý do (§25): HyperGCN 17 khớp H36M được học (layer scale ~3e-2) nhưng bỏ đi không kém → thừa chức năng (self-attn khớp trong fusion + readout siêu cạnh) / lệch nhiệm vụ (cấu trúc xương đã nằm trong pose3d lifter). Đặt HyperGCN vào chỗ chứa đúng đại lượng Student đoán: 24 token query SMPL (twist, Δ đầu chi) → buộc dự đoán trong cùng chuỗi chi nhất quán.
- Luồng: Fusion → ctx (KHÔNG qua HyperGCN 17 khớp, `v7_no_hgcn`) → query SMPL cross-attn ctx → **HyperGCN SMPL-24 (3 lớp)** → head twist / Δ đầu chi (+ part_tok / body_tok như cũ) → J29 → IK → mesh.
- Siêu cạnh SMPL-24 bám sát 10 part / 5 body H36M của HyperDiff (cùng thứ tự), thêm spine1-3, collar, foot, hand vào đúng chuỗi chi: part [0,3,6,9] [9,12,15] [0,2,5] [5,8,11] [0,1,4] [4,7,10] [9,14,17,19] [9,13,16,18] [19,21,23] [18,20,22]; body [0,1,2,3,6,9,12,13,14,15,16,17] [17,19,21,23] [16,18,20,22] [2,5,8,11] [1,4,7,10]; đồ thị khớp = cây SMPL. Phủ đủ 24 khớp.
- Code (backup `_to_delete/bak_smpl_hgcn_20261112/`): file mới `lib/models/hypergraph_smpl.py` (`HyperGCNSMPL24` kế thừa `HyperGCN`, chỉ ghi đè `_init_part_adj` / `_init_body_adj`; block chép `HGraphBlock`, layer scale 1e-5) — KHÔNG sửa `hypergraph.py` / `HGraph.py`; `config.py` thêm `smpl_hgcn` (False), `smpl_hgcn_layers` (3); `teacher_student.py` tạo `smpl_hypers` + 1 dòng sau `pose_context_attn`; `hgcn_freeze_adj` khóa cả ma trận SMPL-24. Tắt → y hệt cũ.
- YAML `config/train_student_smpl_hgcn.yml` (exp `student_smpl_hgcn`) = `train_student_abl_no_hgcn.yml` + `smpl_hgcn: True` → so trực tiếp với `student_abl_no_hgcn` (chỉ khác 1 yếu tố) và breg. Lifter cũ (Markov v1, có 3dpw-train). Chỉ số chính: lỗi J29 đầu chi (log, ~138 mm), MPVPE, PA.
- Vẽ kiểm tra: `tools/vis_hyperedges.py` (T-pose SMPL thật, import đúng danh sách siêu cạnh model dùng; 3 hình + bảng khớp → part/body). Người dùng kiểm tra hình 2026-11-12: siêu cạnh đúng thiết kế (đường zigzag ở body thân chỉ là lỗi vẽ polyline).
- **Log smpl_hgcn (người dùng gửi 2026-11-12 13:00, tới epoch 17/22 — CHƯA phải `last`; lifter cũ)**. TB ep11–17 (smpl_hgcn vs breg cùng epoch): 3DPW MPJPE 73.58 vs 73.54, PA ~48.29 vs ~48.25, MPVPE 90.60 vs 90.56; Crowd MPJPE 72.27 vs 72.25; OC ~80.7 vs ~80.7. J29 đầu chi (log) ep17 136.6 mm vs breg 138.0 mm (~1 mm, trong dao động). ep17: 3DPW 73.45 / 48.23 / 90.01, OC 80.77 / 50.77 / 95.69, PC 63.21 / 66.47 / 95.92, Crowd 72.27 / 52.45 / 95.43.
  - Nhận xét sơ bộ: HyperGCN SMPL-24 KHÔNG cải thiện (≈ breg ≈ no_hgcn), kể cả chỉ số nó nhắm tới (J29 đầu chi, MPVPE). Giả thuyết: head bề mặt bị giới hạn bởi THÔNG TIN (đặc trưng ảnh ResNet-50 F8 8×8, crop 256) chứ không bởi cấu trúc/tính nhất quán; vị trí đồ thị không phải nút thắt. Chờ `last`.

## 27. ImgLifter KHÔNG 3dpw-train (2026-11-12, người dùng yêu cầu, đã tạo YAML, chưa chạy)
- `config/train_img_lifter_dense48_markov_no3dpw.yaml` (exp `img_lifter_dense48_markov_no3dpw`) = `train_img_lifter_dense48_3dpw_markov.yaml`, CHỈ bỏ `3dpw-train` (ratio 0.25 × 4 bộ, giống Refiner dày / Student). Refiner dày vốn KHÔNG train 3dpw → không cần train lại. Không sửa code.
- Mục đích: giao thức công bằng với bảng chính JOTR / DPMesh → báo 3DPW-OC / PC TOÀN BỘ sequence. Sau đó Student: chép `train_student.yml`, đổi `imglift_checkpoint` + `pw3d_ocpc_test_only: False`.
- Kỳ vọng (giả thuyết): LIFT kém đi vài mm so với Markov v1 (72.53 trên 3DPW). Kết quả: chưa có.
- Lifter no3dpw: người dùng báo train xong (2026-11-12 11:19). Đã tạo `config/train_student_no3dpw.yml` (exp `student_no3dpw`) = `train_student.yml`, chỉ đổi `imglift_checkpoint` → `img_lifter_dense48_markov_no3dpw/last` và `pw3d_ocpc_test_only: False`. Kết quả eval lifter: chưa có.
- **Eval lifter no3dpw `last`** (người dùng gửi 2026-11-12 12:18, `eval_img_lifter.py --ocpc_mode all`, K=10 S=4, mean, MPJPE17 / PA; OC/PC TOÀN BỘ): 3DPW 74.80 / 49.87 | OC (n 20243) 75.49 / 53.47 | PC (n 2218) 85.49 / 64.06 | Crowd 83.22 / 56.47. oracle min-K 67.25 / 67.42 / 77.00 / 75.19. Khớp sót/có: 3DPW 91.1 / 71.5, Crowd 110.6 / 77.4.
  - So Markov v1 CÓ 3dpw-train (§13, cùng K10 S4): 3DPW 65.80 / 43.78 → **+9.0 / +6.1**; Crowd 64.63 / 47.62 → **+18.6 / +8.9** (Crowd KHÔNG nằm trong 3dpw-train). OC/PC không so được (khác tập).
  - Ngang các lifter cũ không 3dpw (§6: 3DPW 73–85, Crowd 85–100) → lợi ích lớn trước đây chủ yếu đến từ 3dpw-train.
  - Giả thuyết (chưa kiểm chứng): 3dpw-train dạy lifter nhiễu 2D THẬT của detector (OpenPose) + miền ngoài trời, trong khi 4 bộ còn lại dùng GT + PoseFix (nhiễu tổng hợp) → Crowd (HHRNet) hưởng lợi nhiều dù không rò rỉ.
  - Ước lượng thô (giả thuyết): LIFT 14 khớp ≈ MPJPE17 + ~6.5 (độ lệch ở lifter cũ) → 3DPW ~81, Crowd ~89; mesh Student ≈ LIFT + ~1 → kém DPMesh (73.6 / 79.9) và JOTR (76.4 / 82.4). Cần xác nhận bằng `ik_upper_bound.py` với lifter mới (eval, không train) trước khi train Student 22 epoch.
- **ik_upper_bound với lifter no3dpw** (người dùng gửi 2026-11-12 12:39, K5 S2, mới có 3DPW), MPJPE / PA / MPVPE 14 khớp: LIFT **82.00 / 53.83**; C (β TB) 83.41 / 55.57 / 104.84; D (β_xương) 84.34 / 55.71 / 105.19. Ước lượng LIFT ~81 ĐÚNG. So lifter cũ (có 3dpw-train): LIFT 72.53 → 82.00 (+9.5), C 73.18 → 83.41. Dự đoán Student (D − 1…3 mm, giả thuyết): ~81–83 / ~54 / ~102 → ngang 3DCrowdNet (81.7 / 51.5 / 98.3), kém JOTR (76.4) và DPMesh (73.6) ~8–9 mm. OC / PC / Crowd: chờ.
- **Log train lifter no3dpw theo epoch** (người dùng gửi 2026-11-12 13:30; log bắt đầu từ giữa ep3, thiếu phần đầu in kích thước bộ; đánh giá nhanh EMA, K=5 S=2, mean, 3DPW chỉ 3000 mẫu đầu → KHÔNG so với số eval đầy đủ ở trên), MPJPE17 / PA:
  - Train `MPJPE x0 (mọi t)`: ep4 66.36 → ep10 60.04 → ep17 57.24 → ep19 54.10 (sau giảm lr ở ep18) → ep25 52.19; bone 13.45 → 6.26 mm. Lỗi 2D Refiner (train) ~8.9–9.0 px, không đổi.
  - Test gần như PHẲNG từ ep5: 3DPW ep5 85.94 / 50.71 → ep10 85.53 / 50.53 → ep17 85.80 / 50.61 → ep25 85.96 / 50.72 (dao động 85.2–86.3); PC ep4 84.54 / 63.29 → ep25 84.74 / 63.79; Crowd ep4 82.98 / 55.99 → ep25 83.00 / 56.36. oracle min-K 3DPW ~79–80 suốt quá trình. Khớp sót / có (ep25): Crowd 109.9 / 77.3.
  - Nhận xét: train giảm ~14 mm trong khi test không đổi từ ep5 (≈ 320k mẫu) → giả thuyết "lifter train chưa đủ lâu / chưa đủ lượt dữ liệu" KHÔNG được ủng hộ (đường test phẳng từ rất lâu TRƯỚC lần giảm lr đầu). Lưu ý: lỗi train tính trung bình mọi mức nhiễu t nên không so trực tiếp độ lớn với test (lấy mẫu từ nhiễu thuần); chỉ so xu hướng.
  - Giả thuyết còn lại (chưa kiểm chứng): phần học thêm sau ep5 chỉ khớp phân bố train (2D GT + PoseFix, ảnh H36M / MuCo) và không chuyển sang test (2D OpenPose / HHRNet, ảnh ngoài trời); hoặc thông tin độ sâu từ backbone Refiner (chỉ học 2D, đóng băng) đã cạn.
- **Oracle 2D GT cho lifter no3dpw (2026-11-12, người dùng duyệt, đã tạo YAML, chưa chạy)**: `config/eval_img_lifter_no3dpw_gt2d.yaml` = YAML lifter no3dpw + `DATASET.input_2d_gt: True` (chỉ đánh giá, không sửa code): 2D test thay bằng GT 2D (mesh chiếu) → Refiner dày → ImgLifter. Mục đích: tách "lệch 2D train/test" với "ảnh + độ sâu". Mốc OpenPose (K10 S4): 3DPW 74.80 / 49.87, OC 75.49 / 53.47, PC 85.49 / 64.06, Crowd 83.22 / 56.47. Đọc: tụt mạnh → nút thắt là 2D; gần như không đổi → nút thắt là ảnh / độ sâu. Oracle lạc quan (GT cho cả khớp bị che). **Kết quả 3DPW** (người dùng gửi 2026-11-12 14:42, n 35515, K10 S4; khớp sót 0.9% xác nhận mask = GT): mean **76.16 / 48.05** (OpenPose 74.80 / 49.87 → MPJPE KÉM 1.4, PA tốt hơn 1.8), jpma 77.31 / 49.98, oracle min-K 67.86 / 46.12, lệch 2D 4.63 px (sàn 3.01). → **2D đầu vào KHÔNG phải nút thắt**: 2D hoàn hảo vẫn 76 mm. Khớp với lỗi trên tập train in-the-wild (70–78). Nút thắt là suy ra độ sâu / tư thế 3D từ ảnh (đặc trưng Refiner chỉ học 2D, đóng băng). Hệ quả: detector tốt hơn (ViTPose), NKR teacher 2D GT, sửa nhiễu 2D train đều ít giá trị. MPJPE kém đi có thể do 2D sạch nằm ngoài phân bố train (luôn có PoseFix) — giả thuyết. Ghi chú: oracle 2D GT cũ (§5, ~59 mm) dùng lifter có 3dpw-train / train bằng GT 2D → không so trực tiếp.
- **Lỗi lifter no3dpw trên chính tập TRAIN** (người dùng gửi 2026-11-12 13:59, `eval_img_lifter.py --train_sets Human36M,MuCo,MSCOCO,CrowdPose --max_samples 3000`, K10 S4, 2D = GT + PoseFix có augmentation như lúc train, GT = `orig_joint_cam` (mesh NeuralAnnot), MPJPE17 / PA, mean | oracle min-K | sàn chiếu `gt3d_san` (px) | tỷ lệ khớp sót):
  - Human36M 33.46 / 26.46 | 29.11 | 2.86 | 1.4%
  - MuCo 72.28 / 57.88 | 62.74 | 8.53 | 20.2%
  - MSCOCO 77.56 / 41.56 | 60.17 | 5.78 | 46.7%
  - CrowdPose 69.97 / 34.60 | 53.39 | 4.96 | 35.0%
  - So test (§27, K10 S4): 3DPW 74.80 / 49.87, Crowd 83.22 / 56.47. jpma kém mean ở cả 4 bộ train (như test).
  - Nhận xét: (1) MPJPE trên ảnh in-the-wild của CHÍNH tập train (COCO 77.6, CrowdPose 70.0) ≈ 3DPW test (74.8) → KHÔNG có khoảng cách train/test lớn về MPJPE; lifter chưa khớp được ngay cả dữ liệu train ngoài phòng lab → giả thuyết "lệch miền 2D train/test là nguyên nhân chính" bị YẾU đi; nghiêng về thiếu thông tin / thiếu năng lực trên ảnh in-the-wild (H36M 33.5 cho thấy mô hình khớp được khi ảnh trong lab và nhãn chính xác). (2) PA vẫn có khoảng cách: COCO 41.6 / CrowdPose 34.6 (train) vs 3DPW 49.9 / Crowd 56.5 (test) → có phần không tổng quát hóa ở hình dạng pose. (3) MPJPE − PA rất lớn trên COCO / CrowdPose (~36 mm, H36M ~7 mm) → phần lớn lỗi là hướng / tỷ lệ tổng thể (nhãn pseudo-GT, camera ước lượng). (4) MuCo bất thường: PA 57.9 trên chính tập train (tệ hơn PA 3DPW test 49.9) và sàn chiếu 8.53 px (gấp 3 H36M) → nghi vấn (CHƯA kiểm chứng): nhãn mesh NeuralAnnot của MuCo lệch 2D, hoặc phối cảnh mạnh / ảnh ghép che nhiều của MuCo; MuCo chiếm 25% dữ liệu train lifter.
- **Ablate ảnh lifter no3dpw** (người dùng gửi 2026-11-12 15:12, `eval_img_lifter.py --ablate_img`, 3DPW n 35515, K10 S4, OpenPose; f8 và feat_j = 0, giữ 2D Refiner): mean **247.70 / 129.60**, jpma 239.37 / 136.45, oracle min-K 224.56, lệch 2D **16.08 px** (bình thường ~4.6–5.0). → Lifter SỤP hoàn toàn, kể cả không còn bám 2D đầu vào. Diễn giải: phép đặt đặc trưng = 0 nằm ngoài phân bố train (đặc trưng ảnh đi vào cả c_j, adaLN và cross-attn) → KHÔNG đo được "ảnh đóng góp bao nhiêu thông tin độ sâu", chỉ cho thấy lifter phụ thuộc cấu trúc vào đặc trưng ảnh (không phải bộ nâng 2D thuần). Dự đoán trước của Claude ("gần như không đổi") SAI. Kết luận chẩn đoán chính (§27: 2D GT vẫn 76 mm; train in-the-wild 70–78 mm) không đổi: đặc trưng ảnh lifter đang dùng không đủ để suy ra độ sâu / tư thế trên ảnh in-the-wild.

## 28. Student với lifter no3dpw: mốc + Student sửa pose (2026-11-12, người dùng duyệt, đã tạo YAML, chưa chạy)
- Người dùng chốt: GIỮ 2D như hiện tại (GT + PoseFix cho train), không đổi nguồn 2D lifter.
- **Kết quả `student_no3dpw` (người dùng nhớ lại 2026-11-14 10:16, chưa có log đầy đủ)**: 3DPW MPJPE ~**84** mm (14 khớp; chưa rõ epoch cuối hay tốt nhất; OC / PC / Crowd: chưa có). So mốc không học D (`ik_upper_bound`, lifter no3dpw + β_xương + IK) 84.34 và C (β TB) 83.41 → phần học của Student ≈ 0 với lifter này (giả thuyết, cần số chính xác). JOTR 76.4 (best-of-test) → chênh ~7.6 mm; dù trừ phần lợi do chọn theo test (vài mm, chưa đo) vẫn còn khoảng cách thật.
- GPU 0: `config/train_student_no3dpw.yml` (exp `student_no3dpw`) — Student chính thức, lifter no3dpw → số thật dưới giao thức công bằng.
- GPU 1: `config/train_student_no3dpw_delta17.yml` (exp `student_no3dpw_delta17`) — chỉ đổi `v7_delta17: True`: pose17 = lift + Linear(ctx) (khởi tạo 0), giám sát qua loss `ctx_3d` (v5_ctx_loss_w = 1). Câu hỏi: lifter yếu hơn (LIFT 3DPW 82.0) thì ảnh có sửa được pose thân không? Kết quả: chưa có.

## 29. ImgLifter fine-tune Refiner cùng lúc (2026-11-12, người dùng duyệt, đã code, chưa chạy)
- Lý do (§27): 2D GT vẫn 76 mm; lỗi trên ảnh in-the-wild của chính tập train 70–78 mm → nghi đặc trưng ảnh (backbone Refiner chỉ học loss 2D, đóng băng) thiếu thông tin độ sâu. JOTR / DPMesh đều train backbone bằng loss 3D (DPMesh: đổi backbone 4–6 mm).
- Code (backup `_to_delete/bak_lifter_ftref_20261112/`): `config.py` thêm `IMGLIFT.refiner_finetune` (False), `refiner_lr_scale` (0.5), `refiner_loss_w` (1.0); `img_lifter_tools.refiner_inputs_train` (có gradient; xy / log_b detach, f8 giữ gradient, feat_j vốn detach trong Refiner); `train_img_lifter.py`: khi bật → Refiner train() (BN backbone vẫn eval), 3 nhóm lr (lifter 2e-4 | head Refiner 1e-4 | backbone 1e-5), cộng loss 2D Refiner giống `train_refiner2d.py` target mesh (17 khớp + 48 điểm dày), EMA Refiner, eval nhanh dùng Refiner EMA, lưu `refiner_epoch_XX.pth.tar` / `refiner_last.pth.tar` và ghi vào `ck['refiner_ckpt']` (đường dẫn tương đối gốc repo) → eval / ik_upper_bound / Student nạp đúng Refiner mới; resume nạp cả Refiner. Tắt → y hệt cũ.
- YAML `config/train_img_lifter_dense48_markov_no3dpw_ftref.yaml` (exp `img_lifter_dense48_markov_no3dpw_ftref`) = YAML no3dpw + `refiner_finetune: True` (+ `DATASET.dense2d_vertex_file` bắt buộc cho nhãn điểm dày).
- So với lifter no3dpw (K10 S4): 3DPW 74.80 / 49.87, OC 75.49 / 53.47, PC 85.49 / 64.06, Crowd 83.22 / 56.47; tập train COCO 77.56 / 41.56, CrowdPose 69.97 / 34.60. Tiêu chí: 3DPW tốt hơn ≥ 2 mm → đi tiếp (Student với lifter mới); không → soát nhãn (MuCo) trước. Xem thêm log: đường test còn giảm sau epoch 5 không (trước đây phẳng từ ep5).
- **Log train ftref** (người dùng gửi 2026-11-12 22:16, đủ 25 epoch; đánh giá nhanh EMA, K5 S2, 3DPW 3000 mẫu đầu, MPJPE17 / PA), so cùng epoch với lifter no3dpw (Refiner đóng băng):
  - ep25: 3DPW **87.11 / 51.14** (đóng băng 85.96 / 50.72 → KÉM 1.2), PC 84.84 / 63.91 (84.74 / 63.79, ≈), Crowd **81.61 / 56.00** (83.00 / 56.36 → tốt hơn 1.4). oracle min-K 3DPW 81.36 (80.15).
  - Đường test lại PHẲNG từ ~ep5 (3DPW ep5 86.48 → ep25 87.11; Crowd 83.31 → 81.61 giảm nhẹ dần). Train x0 50.95 (đóng băng 52.19). Lỗi 2D Refiner (train, target mesh) 8.21 → 7.87 px → loss 3D KHÔNG phá định vị 2D.
  - Nhận xét: KHÔNG đạt tiêu chí (≥ 2 mm trên 3DPW); chênh ±1–1.5 mm, trái dấu giữa các tập → trong dao động. Giả thuyết "backbone đóng băng chỉ học 2D là nút thắt chính" KHÔNG được ủng hộ, ít nhất với cấu hình này. Giới hạn chưa loại trừ: lr backbone 1e-5 (JOTR dùng lr_backbone 1e-4, `JOTR/config.py`), gradient 3D chỉ vào backbone qua f8 (cross-attn 64 token), BN backbone đóng băng. Cả hai run đều phẳng từ ep5 → nghiêng về giới hạn dữ liệu / nhãn (giả thuyết). Chờ eval đầy đủ (K10 S4, OC/PC toàn bộ) để chốt.

## 30. Teacher 2D GT (2026-11-12, người dùng duyệt, đã tạo YAML, chưa chạy)
- Mục đích: Teacher cho KD nhẹ sau này (kiểu NKR của DPMesh): cùng kiến trúc Student v7 breg, chỉ khác 2D đầu vào sạch.
- `config/train_teacher_gt2d.yml` (exp `teacher_gt2d`) = `train_student.yml` + `DATASET.input_2d_gt: True` (split 'all'): GT 2D ở mọi bộ train VÀ test → Refiner → ImgLifter (3dpw-train, Markov v1) → Student. Không sửa code (key có sẵn, áp qua `Human36M17Dataset` cho cả 4 bộ train và các tập test).
- Thay cho kế hoạch Teacher GT 3D (người dùng đổi ý 16:33, chưa code).
- Mong đợi (giả thuyết): gần Student vì lifter no3dpw với GT 2D vẫn 76.16 (§27); có thể chênh ở PA / MPVPE (phần bề mặt). Số test của YAML này là ORACLE (test cũng GT 2D); test thêm ckpt bằng `train_student.yml` để xem Teacher với 2D detector.
- So với Student breg `last`: 3DPW 73.32 / 48.16 / 90.11, OC 80.54 / 50.60 / 95.45, PC 61.56 / 64.96 / 100.09, Crowd 72.30 / 52.53 / 95.18.
- KD sau này: `Student_Trainer` KD hiện tạo Teacher bằng `MODEL.name = 'teacher'` (ARTS báo lỗi) và đưa 2D detector cho Teacher → cần sửa để Teacher nhận GT 2D (`targets['orig_joint_img']`). Kết quả: chưa có.

## 31. Lifter TEACHER 2D GT cho KD ở lifter (2026-11-12, người dùng duyệt, đã tạo YAML, chưa chạy)
- Lý do: lifter train bằng 2D nhiễu không tận dụng được 2D chính xác (GT 2D lúc test 76.16 vs OpenPose 74.80, §27) → Teacher ở mức Student (§30) khó mạnh hơn Student. Lifter train hẳn bằng GT 2D mới là teacher có ý nghĩa (kiểu NKR DPMesh, áp vào lifter). Cận trên nâng 2D GT → 3D trên 3DPW trong tài liệu ~65 mm (Pose2Mesh, giả thuyết là teacher đạt gần đó).
- `config/train_img_lifter_dense48_markov_no3dpw_ftref_teacher_gt2d.yaml` (exp `img_lifter_dense48_markov_no3dpw_ftref_teacher_gt2d`) = YAML ftref (§29, CÙNG version: Refiner mở khóa, không 3dpw-train) + 3 key định nghĩa teacher sạch: `DATASET.input_2d_gt: True` (split 'all', train + test), `IMGLIFT.train_xy_noise: 0`, `IMGLIFT.input_drop_prob: 0`. Không sửa code. Eval nhanh mỗi epoch và eval cuối = oracle (GT 2D).
- Tiêu chí (MPJPE17 3DPW, K10 S4): ≤ ~65 → code KD ở lifter (bước 2: teacher đóng băng + student cùng t và cùng nhiễu, ép token ẩn từng khớp; chờ duyệt kế hoạch); ~72–76 → dừng hướng KD.
- §30 (Teacher 2D GT ở mức Student): Claude đề nghị tạm hoãn để dành GPU.
- **Log train teacher** (người dùng gửi 2026-11-12 22:29, tới ep17/25, ep18 đang chạy; 2000 iter / epoch: người dùng xác nhận 22:33 đã giảm batch trên server (thiếu VRAM) → batch 32 (suy từ 64000 mẫu / 2000 iter; file YAML local vẫn ghi 64), cùng số mẫu / epoch, gấp đôi số bước cập nhật so với ftref; đánh giá nhanh EMA, K5 S2, GT 2D cả lúc test = oracle, MPJPE17 / PA):
  - ep17: 3DPW **57.09 / 37.42**, PC **54.81 / 35.58**, Crowd **50.07 / 30.96** (oracle min-K 3DPW 54.16). Student lifter ftref (OpenPose) cùng epoch: 3DPW 86.82 / 51.02, PC 84.31 / 63.92, Crowd 81.54 / 56.07 → teacher hơn **~30 mm MPJPE, ~14–28 mm PA**.
  - Đường test KHÔNG phẳng từ ep5 như các run 2D nhiễu: 3DPW ep5 60.06 → ep10 58.26 → ep17 57.09. Train x0 33.6 mm (ep17), lỗi 2D Refiner (train) 0.88 px.
  - So oracle §27 (lifter train 2D NHIỄU, test GT 2D): 76.16 → lifter train 2D SẠCH, test GT 2D: ~57. → **SỬA kết luận §27**: "2D không phải nút thắt" là SAI; đúng là lifter train với nhiễu 2D nặng (PoseFix 1.0 + train_xy_noise 0.5 + bỏ khớp 0.1) KHÔNG tận dụng được 2D chính xác. Độ chính xác 2D đáng ~30 mm khi lifter được dạy dùng nó (giả thuyết (b) người dùng nêu 15:25 được ủng hộ).
  - Lưu ý: oracle lạc quan (GT 2D biết cả khớp bị che, không nhầm người). Crowd (50) < 3DPW (57) → lỗi Crowd hiện nay chủ yếu do 2D (nhầm người / che).
  - Đạt tiêu chí làm KD ở lifter (≤ ~65). Bước tiếp đề xuất (chưa duyệt): (A) lifter ftref với nhiễu train nhẹ hơn (1 YAML) để tìm điểm cân bằng nhiễu ↔ tận dụng 2D, test OpenPose; (B) KD ở lifter (teacher này đóng băng → student ftref, cùng t và cùng nhiễu, ép token ẩn).

## 32. Lifter PoseFix 0.5 + KD ở lifter (2026-11-12, người dùng duyệt, đã code, chưa chạy)
- Người dùng chốt: 2 run, cả hai batch 32 (VRAM).
- **A. `config/train_img_lifter_dense48_markov_no3dpw_ftref_pf05.yaml`** (exp `..._ftref_pf05`) = ftref + `posefix_scale` 1.0 → 0.5 + batch 64 → 32. Train từ đầu, Refiner mở khóa. Không sửa code. Lý do: §31 (lifter train 2D nhiễu nặng không tận dụng 2D chính xác); tiền lệ §6 (DiffLifter cũ PoseFix 0.5 hơn 1.0 ~4 mm). Rủi ro: PC / Crowd kém đi.
- **B. KD ở lifter `config/train_img_lifter_dense48_markov_no3dpw_ftref_kd.yaml`** (exp `..._ftref_kd`) = ftref + KD, fine-tune từ ftref `last` (`init_ckpt`), batch 32, 10 epoch, lr_step [6, 9].
  - Code (backup `_to_delete/bak_lifter_kd_20261112/`): `config.py` thêm `IMGLIFT.kd_teacher_ckpt` (''), `kd_w` (0.05), `init_ckpt` (''); `img_lifter.py`: `denoiser.forward` / `_x0` / `loss` nhận `return_hidden` (token ẩn 17 khớp lớp cuối, trước head), `loss` nhận `t`, `noise` từ ngoài — mặc định y hệt cũ; `train_img_lifter.py`: nạp init (lifter + Refiner raw / EMA, không nạp optimizer), nạp teacher (lifter + Refiner từ `ck['refiner_ckpt']`, đóng băng), mỗi batch teacher nhận GT 2D (`orig_joint_img`, mask trunc × valid, không nhiễu), student nhận 2D nhiễu, CÙNG t và CÙNG nhiễu; loss KD = MSE(layer_norm(h_s), layer_norm(h_t)) trung bình khớp × kênh, chỉ mẫu `is_valid_fit`, × `kd_w`; log `KD: mse ...` mỗi epoch.
  - Teacher = `img_lifter_dense48_markov_no3dpw_ftref_teacher_gt2d/last` (§31) — cần chạy xong ep25.
  - **+ InfoNCE** (người dùng yêu cầu 22:41, kiểu NKR DPMesh; backup `_to_delete/bak_lifter_kdnce_20261112/`): key `IMGLIFT.kd_nce_w` (0 = tắt), `kd_nce_tau` (0.07). Anchor = token ẩn student (b, j), dương = token teacher cùng (b, j), âm = mọi token teacher khác trong batch (≈ 32 × 17 − 1), cosine / τ, chỉ mẫu `is_valid_fit`. YAML KD (chưa chạy) cập nhật `kd_nce_w: 0.01`. Loss KD tổng = kd_w × MSE + kd_nce_w × InfoNCE.
  - So với ftref `last` (điểm khởi tạo). Đối chứng chặt (tùy chọn): cùng YAML, `kd_teacher_ckpt: ''`. Chỉnh `kd_w` theo log epoch 1 (mục tiêu: KD cùng bậc độ lớn với loss MPJPE x0 ~0.05).
- **Rà soát flow (Claude, 2026-11-12 22:43, đọc lại toàn bộ code, không chạy)**: cơ chế code khớp thiết kế (cùng t / nhiễu, đầu vào teacher trùng `input_2d_gt` lúc train teacher, teacher + Refiner của nó đều là bản EMA đóng băng, EMA / lưu ckpt / resume đúng). PHÁT HIỆN 1 lỗi THIẾT KẾ ở YAML B: student khởi tạo từ ftref (train độc lập với teacher) → không gian token ẩn của hai mạng không cùng hệ trục, MSE / InfoNCE trực tiếp (không projector) sẽ ép student xoay lại toàn bộ đặc trưng. DPMesh (`DPMesh/common/base_ssl.py`: `teacher.load_state_dict(student.state_dict())`) cho teacher và student CÙNG trọng số ban đầu. Đề xuất (chờ duyệt): đổi `init_ckpt` của YAML B sang `last` của teacher (YAML-only; ckpt teacher có đủ `raw_model_state` / `refiner_raw_state`). Khi đó `kd_w` / `kd_nce_w` cần chỉnh lại theo log epoch 1 (loss KD ban đầu nhỏ hơn nhiều).
- **Sửa InfoNCE cho giống DPMesh (2026-11-12 22:53, người dùng nhắc; backup `_to_delete/bak_lifter_kdnce2_20261112/`)**: DPMesh có lớp `SimCLR` (NT-Xent) và `MSELoss(student_feat, teacher_feat)` trong `DPMesh/common/nets/loss.py` (KHÔNG phải `nets/infonce.py` — file đó là contrastive khớp 3D thừa hưởng từ JOTR); vòng train NKR gọi chúng không có trong code phát hành. SimCLR của họ: mỗi MẪU 1 vector (flatten), đối xứng, âm = 2(N − 1) vector còn lại của cả hai nhánh, τ mặc định của lớp = 1.0 (giá trị thực dùng: không rõ). Thêm key `IMGLIFT.kd_nce_mode`: `sample` (mặc định, giống DPMesh) | `joint` (bản đầu của Claude: mức khớp, một chiều). YAML KD đặt `sample`, τ giữ 0.07.
- **Đổi khởi tạo student (người dùng duyệt 22:55)**: YAML KD `init_ckpt` → `last` của TEACHER (trước: ftref). Rủi ro cần theo dõi: Refiner của teacher chỉ quen 2D sạch (lỗi 2D train ~0.9 px = gần như chép đầu vào) → student phải học lại việc sửa 2D nhiễu với lr head 1e-4 / backbone 1e-5 trong 10 epoch; chỉ báo: `Refiner: loi ... px` phải về ~8 px (mức ftref).
- **Đánh giá loss KD (Claude, 22:55, suy luận — chưa có thí nghiệm)**: (1) MSE trên token ẩn lớp CUỐI ≈ chưng cất đầu ra, trùng vai với loss MPJPE (đã có GT 3D) → tín hiệu yếu. (2) InfoNCE trên token ẩn có đường tắt: mỗi mẫu một bước t khác nhau, student và teacher cùng mẫu dùng chung t → phân biệt mẫu bằng "dấu vết t" mà không cần nội dung. (3) NKR của DPMesh căn đặc trưng BACKBONE (nơi heatmap 2D được tiêm); tương ứng ở đây là `f8` của Refiner: không phụ thuộc t, gradient vào thẳng backbone, đúng nghĩa "đặc trưng với 2D nhiễu phải giống với 2D sạch". Đề xuất (chờ duyệt): thêm MSE + InfoNCE trên `f8`, giữ MSE token ẩn.
- **Người dùng chốt lại (22:59): student KD train TỪ ĐẦU** (không khởi tạo từ teacher, không từ ftref). YAML KD cập nhật: bỏ `init_ckpt`, `end_epoch` 25, `lr_step` [18, 23] (= ftref), batch 32; lifter ngẫu nhiên, Refiner từ `refiner2d_dense48_scratch` (cùng gốc với Refiner của teacher → `f8` hai bên gần cùng hệ trục). Hết rủi ro "Refiner teacher phải học lại sửa 2D nhiễu". So với ftref có lẫn yếu tố batch (64 vs 32); đối chứng chặt = cùng YAML, `kd_teacher_ckpt: ''`. Đề xuất KD trên `f8` vẫn CHỜ duyệt.
- **KD trên `f8` (người dùng duyệt 23:01; backup `_to_delete/bak_lifter_kdf8_20261112/`)**: key mới `IMGLIFT.kd_f8_w` (0 = tắt) và `kd_nce_feat` (`hidden` | `f8`). `train_img_lifter.py`: loss f8 = TB 64 ô của ‖L2norm_kênh(f8_student) − L2norm_kênh(f8_teacher)‖² (= 2 − 2cos), chỉ mẫu `is_valid_fit`, gradient vào backbone Refiner student (bắt buộc `refiner_finetune`); InfoNCE chuyển sang `f8` (hàm `nt_xent`, NT-Xent đối xứng, 1 vector / mẫu). YAML KD cuối cùng (train từ đầu, batch 32, 25 epoch): `kd_f8_w: 1.0` (chính), `kd_nce_w: 0.01` trên `f8`, `kd_w: 0.05` (MSE token ẩn, phụ). Trọng số là ước lượng, chỉnh theo log epoch 1. Dự đoán (giả thuyết): InfoNCE trên f8 gần bão hòa ngay (ảnh khác nhau vốn dễ phân biệt) → tín hiệu chính là MSE f8; nếu `KD f8` ở epoch 1 rất nhỏ (< ~0.02) nghĩa là f8 gần như không phụ thuộc heatmap 2D → KD f8 cũng yếu.
- **Teacher dừng sớm (người dùng, 2026-11-12 23:30)**: teacher §31 bị dừng trước epoch 25 (khoảng ep20–21, lr đã giảm còn 2e-5 từ ep18); lượt KD dùng `last.pth.tar` mới nhất lúc đó (epoch xem trong log KD: dòng `Nap ImgLifter ... (epoch X` và `Nap Refiner 2D .../refiner_last.pth.tar (epoch Y` — phải X = Y). Teacher chỉ là công cụ, không báo số của nó như kết quả chính; số teacher ghi theo đúng epoch đó.
  - Log KD (người dùng gửi 23:35): teacher lifter `last` **epoch 20**, Refiner teacher `refiner_last` **epoch 20** → khớp (X = Y = 20). Cấu hình KD in ra đúng YAML (kd_w 0.05, kd_f8_w 1.0, InfoNCE 0.01 τ 0.07 trên f8). Số đánh giá nhanh của teacher ep20: chưa có (ep17: 3DPW 57.09 / 37.42, PC 54.81 / 35.58, Crowd 50.07 / 30.96).
- **Log KD đủ 25 epoch** (người dùng gửi 2026-11-13 11:38; teacher ep20; batch 32, 26 phút / epoch; đánh giá nhanh K5 S2, OpenPose, 3DPW 3000 mẫu đầu, MPJPE17 / PA):
  - ep25: 3DPW **86.31 / 50.81**, PC **85.06 / 63.92**, Crowd **83.29 / 56.28**. So cùng ep25: ftref (batch 64, không KD) 87.11 / 51.14, 84.84 / 63.91, 81.61 / 56.00; lifter no3dpw Refiner đóng băng 85.96 / 50.72, 84.74 / 63.79, 83.00 / 56.36. → chênh ±1.7 mm, trái dấu giữa các tập → **KD KHÔNG cải thiện** (trong dao động).
  - Đường test lại phẳng từ ~ep6 (3DPW 84.2–86.3 suốt ep6–25). Giá trị thấp nhất giữa chừng (ep7 84.19) KHÔNG dùng (không chọn theo test).
  - Loss KD: `KD f8` 0.342 (ep1) → 0.275 (ep2) → 0.220 (ep7) → **0.192 (ep25)**, chững từ ~ep15 (2 − 2cos ≈ 0.19 → cos ≈ 0.90 từng ô); InfoNCE f8 0.88 → 0.55 (mức ngẫu nhiên 4.14); MSE token ẩn 0.168 → 0.022. Lỗi 2D Refiner (train) 8.32 → 7.92 px (≈ ftref) → KD không phá 2D.
  - Nhận xét: student KHÔNG tái tạo được đặc trưng "như khi có 2D sạch" (f8 chững ở cos ~0.90 dù loss KD là thành phần lớn nhất sau nhân trọng số) → giả thuyết: từ ảnh + 2D nhiễu không suy ra được phần thông tin mà GT 2D mang (vị trí khớp bị che / nhầm người) — KD không tạo thông tin mới. Khớp với mức lợi nhỏ của NKR trong DPMesh (1.3–3 mm).
- **Log A (PoseFix 0.5) đủ 25 epoch** (người dùng gửi 2026-11-13 12:01; batch 32; không KD; đánh giá nhanh K5 S2, OpenPose, MPJPE17 / PA):
  - ep25: 3DPW **87.28 / 51.15**, PC **84.54 / 63.51**, Crowd **82.43 / 56.25**. So ep25: ftref (PoseFix 1.0, batch 64) 87.11 / 51.14, 84.84 / 63.91, 81.61 / 56.00; KD 86.31 / 50.81, 85.06 / 63.92, 83.29 / 56.28 → **≈ nhau (±1 mm)**: PoseFix 0.5 KHÔNG cải thiện, cũng không làm PC / Crowd xấu đi.
  - PoseFix 0.5 có tác dụng thật lên dữ liệu: lỗi 2D Refiner train (sau nhiễu) 8.57 → 8.09 px (ftref 8.91 → 8.53), Refiner target mesh 7.95 → 7.49 px (ftref 8.21 → 7.87). Đường test lại phẳng từ ~ep3–5.
  - Nhận xét: 4 run (Refiner đóng băng, ftref, KD, PoseFix 0.5) đều kết thúc 3DPW 86–87, Crowd 81.6–83.3 (đánh giá nhanh) → mức nhiễu 2D lúc train trong khoảng PoseFix 0.5–1.0 không phải nút thắt; khoảng ~30 mm của teacher (§31) đến từ chất lượng 2D LÚC TEST (GT biết khớp bị che, không nhầm người), không phải từ cách lifter dùng 2D. Giả thuyết (b) 15:25 chỉ đúng một phần: lifter train 2D sạch tận dụng 2D chính xác, nhưng giảm nhiễu vừa phải không chuyển thành lợi ích với 2D OpenPose.
- Kết quả: chưa có.

## 33. Tái cấu trúc FuseKin v8: mạng end-to-end, ảnh làm chính — mốc M0 (2026-11-13, nhánh git `v8-e2e`, người dùng duyệt, đã code, chưa chạy)
- Lý do (người dùng chốt 17:43): Student no3dpw ra đúng dự đoán (~81–83 trên 3DPW), kém JOTR (cùng 4 bộ dữ liệu, cùng 2D OpenPose, cùng ResNet-50) ~6 mm → nguyên nhân là cách khai thác + mô hình, không phải dữ liệu. Nguyên nhân gốc (Claude): (1) pose 3D suy chủ yếu từ tọa độ 2D (lifter khuếch đại lỗi 2D: GT 2D 57 vs OpenPose 82); (2) pipeline rời rạc, Student v7 bị khóa vào pose lifter; (3) backbone gần như không học bằng loss 3D / mesh. Phát hiện thêm: ngân sách train Student (32k × 22 = 0.7 triệu mẫu) ít hơn JOTR (`JOTR/common/base.py`: 2 × (H36M + MuCo) ≈ 1.02 triệu / epoch × 20 ≈ 20 triệu) ~29 lần.
- Lộ trình: **M0** (backbone + heatmap + query lặp, không lifter / HyperGCN / IK; phải ≈ JOTR) → M1 (+ token pose lifter, gợi ý mềm) → M2 (+ HyperGCN giữa các lớp decoder) → M3 (+ khối 3D / contrastive, tùy chọn). Mỗi mốc 1 YAML.
- **M0 code** (backup `_to_delete/bak_v8_m0_20261113/`): file mới `lib/models/fusekin_v8.py` (`V8Net` = ResNet-50 SPIN + heatmap 2D tiêm stride 4, BN train, che 10% khớp 2D lúc train; `V8Head` = conv 1×1 → 256 + PE sin → encoder 3 lớp post-norm + decoder 3 lớp (3 token SMPL + 17 query khớp, `dec_norm` sau mỗi lớp như JOTR), hồi quy phần dư theo tầng: pose 24 × 6D (khởi tạo pose trung bình), β, camera, khớp 3D 17 × 3; SMPL như Student, bàn tay = 0). `config.py`: `MODEL.v8_e2e` (False), `v8_dim` 256, `v8_enc_layers` 3, `v8_dec_layers` 3, `v8_joint_drop` 0.1. `ARTS.get_model`: `v8_e2e` → `V8Net`. `main/test_student_3dpw_sets.py`: dựng model bằng `get_model` (không đổi hành vi cũ). `Student_Trainer` / `Tester` dùng nguyên: loss `_smpl_hard_losses` (mesh, khớp 3D từ mesh, tham số pose / shape, chiếu 2D) qua AWL cho cả 3 lớp decoder (`refine_history`) + loss khớp 3D của query (`ctx_3d`).
- YAML `config/train_v8_m0.yml` (exp `v8_m0`): batch 32, lr 1e-4 (backbone ×1.0 như JOTR), 20 epoch, giảm ×0.1 ở epoch 15, **256k mẫu / epoch (≈ 5.1 triệu, ~1/4 JOTR, người dùng chọn)**, tỷ lệ trộn như JOTR [0.30, 0.20, 0.40, 0.10]; không 3dpw; OC / PC toàn bộ; bật lại loss tham số pose.
- **4 GPU (người dùng 17:54)**: batch TỔNG 256 = 64 / GPU, giống JOTR (`JOTR/config.py` train_batch_size 64 / GPU, `train.sh` nproc = số GPU; paper: batch 256, lr 1e-4). Code dùng DataParallel → `TRAIN.batch_size` = batch tổng; BN riêng từng GPU (JOTR dùng SyncBN). `samples_per_epoch` giữ 256k → 1000 iter / epoch; lr giữ 1e-4; workers 8 → 16; TEST batch 128. DPMesh không làm mốc batch (backbone SD, 8 / GPU, lr 2e-4).
- Tiêu chí M0: 3DPW MPJPE 14 khớp ≤ ~78–79 (JOTR 76.4 / 48.7 / 92.6; JOTR không phần 3D, bảng 4: 77.8 trên OC). Không đạt → lỗi ở pipeline dữ liệu / loss / train, sửa trước khi thêm M1.
- **Log M0 ep1–6** (người dùng gửi 2026-11-13 23:25; 4 GPU, batch 256, lr 1e-4 chưa giảm; ~45 phút train + ~5 phút eval / epoch, 2.7 s / iter ≈ 95 mẫu / s → 20 epoch ≈ 16–17 giờ). Đánh giá đầy đủ mỗi epoch (14 khớp, MPJPE / PA / MPVPE; OC / PC toàn bộ):
  - 3DPW: ep1 113.61 / 73.14 / 129.88 → ep2 101.32 / 62.35 / 118.40 → ep3 100.14 / 58.79 / 116.40 → ep4 96.66 / 56.12 / 112.91 → ep5 **92.22 / 54.74 / 108.07** → ep6 96.58 / 56.50 / 115.36.
  - ep5: OC 92.80 / 60.19 / 108.32, PC 104.20 / 73.12 / 128.68, Crowd 97.89 / 60.52 / 116.41.
  - Train: loss tổng 5.51 → 3.22; khớp 3D của query (train) 218.7 → 98.9 mm; AWL tăng trọng số mesh / khớp / pose (0.62 → 2.0–2.4), giảm proj (0.45 → 0.33) — hành vi bình thường của AWL khi loss giảm.
  - Nhận xét: mới ~1.5 triệu mẫu (≈ 1.5 epoch JOTR), lr chưa giảm → chưa kết luận được; xu hướng giảm đều (dao động ±4 mm giữa epoch). Mốc đọc: sau lần giảm lr ở ep15. Giả thuyết (chưa kiểm chứng): nếu tới ep14 3DPW vẫn > ~88–90 thì sau giảm lr khó xuống ≤ 79 → khi đó tăng ngân sách trước khi nghi pipeline.
- **Log M0 ep14–16** (người dùng gửi 2026-11-14 09:05; lr giảm ×0.1 từ ep16): 3DPW ep14 90.94 / 54.00 / 107.33 → ep15 87.99 / 53.43 / 104.10 → ep16 88.64 / 52.98 / 105.37; OC 91.25 → 88.28 → 86.68; PC 102.82 → 98.90 → 97.32; Crowd 96.07 → 91.80 → 89.99. Khớp 3D query (train) 81.6 → 76.8 mm, còn giảm. → Dự đoán cuối (giả thuyết) 3DPW ~86–88: **M0 KHÔNG đạt tiêu chí ≤ 78–79**, và còn KÉM pipeline lifter cũ (Student no3dpw ~82) → hướng "ảnh làm chính" với cài đặt hiện tại CHƯA được xác nhận.
- **Phát hiện giao thức của mốc so sánh (Claude đọc code, 2026-11-14)**: JOTR (`JOTR/train.sh`, `assets/yaml/*.yml`, `train.py`): giai đoạn 1 = 18 epoch, lr 1e-4 KHÔNG giảm (`lr_dec_factor: 1`), 32 / GPU, AdamW wd 0.01, AMP, encoder 2 / decoder 4; giai đoạn 2 = fine-tune từ `snapshot_17`, lr 5e-5, contrastive, 10000 bước, **đánh giá cả 4 tập test MỖI 200 BƯỚC và lưu checkpoint tốt nhất RIÊNG cho từng tập** (`{tập}_best_ckpt.pth.tar`, đúng các file phát hành trong README). DPMesh (`DPMesh/train.py`): đánh giá 3DPW mỗi 4000 iter + cuối epoch, lưu best theo MPJPE test. → Số công bố là **chọn theo test (best-of-nhiều-lần-đánh-giá, riêng từng tập)**, KHÔNG phải epoch cuối như quy tắc dự án; dao động giữa các lần đánh giá của mình ±2–4 mm (M0 ep5 92.2 → ep6 96.6) → chênh lệch giao thức có thể vài mm (chưa đo được).

## 34. FuseKin v8 — mốc M2 (2026-11-14, nhánh `v8-e2e`, người dùng duyệt "làm M2 lun, bỏ qua M1", đã code, chưa chạy)
- Lý do: M0 (§33) ~88 trên 3DPW, kém pipeline cũ; người dùng không train thêm baseline, muốn sửa mô hình của mình. M2 = M0 + 4 phần riêng FuseKin, **train từ đầu** (không khởi tạo từ M0).
- Code (backup `_to_delete/bak_v8_m2_20261114/`: `fusekin_v8.py`, `config.py`, `base.py`):
  - `config.py`: `MODEL.v8_lifter`, `v8_hyp_tokens`, `v8_ik`, `v8_hgcn` (đều False; tắt hết = M0 y hệt).
  - `v8_lifter`: `V8Net` nạp Refiner + ImgLifter no3dpw (đóng băng, luôn eval; `lift()` chép `ARTS.lift_imglift`, K=5, S=2, gộp mean, nhiễu cố định lúc test). Lifter nhận mask detector ĐẦY ĐỦ (che 10% chỉ áp cho heatmap backbone). 17 query khớp: nội dung ban đầu = `Linear(3, D)(pose lifter)`; head khớp 3D lớp 0 = lifter + phần dư (init std 1e-3) → decoder SỬA pose lifter.
  - `v8_hyp_tokens`: 5 × 17 giả thuyết → `Linear(3, D)` + ID khớp + ID giả thuyết → LayerNorm → 85 token nối sau 64 token ảnh trong bộ nhớ decoder (PE = 0).
  - `v8_ik`: mỗi lớp decoder → IK HybrIK từ khớp đã sửa (như Student v7, readout = token khớp trực tiếp): J29 = M × pose17 + delta 7 điểm đầu chi (head riêng từng điểm, token [6, 3, 13, 16, 10, 6, 3]); twist = mặc định + `Linear(2D, 2)`(token cha, con theo `SMPL_BONE_PAIRS_H36M`) + bias riêng từng khớp; β = β_xương(pose17, no-grad) + `Linear(D, 10)`(token shape). Tất cả init 0. Bỏ head 6D / β → bắt buộc `v7_no_pose_param_loss` True (assert). Output `v7_j29` / `v7_dbeta` / `v7_beta0` → Student_Trainer tự cộng loss J29 7 điểm + |dβ|².
  - `v8_hgcn`: 1 khối HyperGCN (`HGraph.create_layers`, layer scale 1e-5, không adaptive fusion) trên 17 token khớp sau MỖI lớp decoder (3 khối riêng).
  - Cột LIFT lúc đánh giá = pose lifter đầu vào; `ctx_3d` = khớp đã sửa lớp cuối (loss `v5_ctx_loss_w`).
  - **Sửa lỗi nạp checkpoint V8** (`base.load_model_weights`): key `smpl_model.shape_head.` của M0 trùng `LEGACY_KEY_PARTS` (Student cũ) → bị bỏ khi nạp → test M0 bằng `test_student_3dpw_sets.py` sẽ dừng ở assert thiếu key. Nay không lọc legacy khi `model.mode == 'v8'`. Không ảnh hưởng Student cũ. (Số M0 trong §33 lấy từ đánh giá trong lúc train → không bị ảnh hưởng.)
- YAML `config/train_v8_m2.yml` (exp `v8_m2`): như M0 + 4 key; batch TỔNG 128 (32 / GPU, Refiner + lifter tốn VRAM), lr 5e-5 (tỷ lệ theo batch), 20 epoch × 256k mẫu, giảm ×0.1 ở ep15; `v7_bone_beta_reg` 0.001, `v7_dbeta_w` 0.01, `v7_j29_loss_w` 1.0; TEST batch 64.
- Chạy: `python main/train.py --cfg ./config/train_v8_m2.yml --exp_name v8_m2 --gpu 0,1,2,3`; test epoch cuối: `python -u main/test_student_3dpw_sets.py --cfg ./config/train_v8_m2.yml --ckpt experiment/v8_m2/checkpoint/last.pth.tar --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --ocpc_mode all --gpu 0`.
- Kỳ vọng (giả thuyết): ep1 đã ≈ mức lifter + IK (~84–86, vì mọi head init ≈ 0), sau đó phải xuống dưới Student no3dpw (~82–84). Nếu M2 không tốt hơn M0 / Student no3dpw → phần sửa từ ảnh không học được gì thêm. Ablation sau: tắt `v8_hgcn` = M1, tắt `v8_hyp_tokens`.
- **Cập nhật 2026-11-14 11:00**: người dùng đánh giá hướng M2 "chưa hay" → M2 KHÔNG chạy (code / YAML giữ nguyên để tham khảo). Thay bằng v9 (§35).

## 35. FuseKin v9 — Student hồi quy mesh từ ảnh + 2D + 3D, giao thức nhanh 3dpw-train → 3dpw (2026-11-14, người dùng duyệt, đã code, chưa chạy)
- Người dùng chốt: Student là mô hình HỒI QUY mesh dựa trên đặc trưng ảnh + 2D + 3D, KHÔNG phải "sửa lifter". Do giới hạn tài nguyên: train CHỈ 3dpw-train (2D OpenPose), test CHỈ 3dpw (OpenPose), không OC / PC / Crowd; ổn mới mở rộng dữ liệu. Lifter = ImgLifter no3dpw (đóng băng). Lịch lr: giảm 4 lần ×0.5.
- Ghi chú lịch sử (Claude rà lại): chọn giả thuyết bằng ảnh đã thử thời DiffLifter, không thành lợi ích mesh; oracle min-K của ImgLifter chỉ thấp hơn mean ~6–8 mm → không làm hướng chính.
- Giả thuyết (chưa kiểm chứng): lifter no3dpw không thấy 3DPW → lỗi của nó trên 3dpw-train giống lúc test → Student học được tình huống lifter sai thật (trước đây Student train trên chính dữ liệu lifter đã thấy → học "tin lifter").
- Code (backup `_to_delete/bak_v9_20261114/`: `ARTS.py`, `base.py`, `config.py`): file mới `lib/models/fusekin_v9.py`.
  - [1] 17 token bằng chứng: F16 (1024 × 16 × 16 của backbone, conv 1×1 → 256) lấy mẫu bilinear tại xy Refiner + MLP 2D (x, y OpenPose × mask, mask, x, y Refiner, log_b) + MLP 3D (pose lifter mean, độ lệch chuẩn 5 giả thuyết × 10) → LayerNorm. Vị trí = ID khớp + PE sin tại xy Refiner (cùng hệ với PE token ảnh). Modality dropout lúc train: 20% mẫu bỏ hết 3D, 10% khớp bỏ 3D (thay bằng vector học được).
  - [2] fuse 3 lớp: self-attn 17 khớp → cross-attn 64 token ảnh (F32 2048 × 8 × 8 → 256 + PE, qua 2 lớp encoder) → FFN → 1 khối HyperGCN HyperDiff (layer scale 1e-5). Khớp 3D phụ = Linear từ token khớp (loss `ctx_3d`, không dùng làm mesh).
  - [3] query SMPL: 24 query khớp SMPL = Linear(token cha, token con) theo `SMPL_BONE_PAIRS_H36M` + embedding; + token shape, token camera; decoder 3 lớp cross-attn vào [17 token khớp đã fuse, 64 token ảnh]; mỗi lớp hồi quy phần dư theo tầng 6D từng khớp (lớp 0 = pose TRUNG BÌNH + head), β (từ β trung bình), camera.
  - [4] SMPL → mesh; loss như M0 (mesh, khớp 3D từ mesh, tham số pose / shape, chiếu 2D qua AWL, cả 3 lớp decoder) + `ctx_3d`. Lifter: cột LIFT lúc đánh giá.
  - `config.py`: `MODEL.v9_e2e` (False), `v9_dim` 256, `v9_img_enc` 2, `v9_fuse_layers` 3, `v9_smpl_dec` 3, `v9_hgcn` True, `v9_drop3d_sample` 0.2, `v9_drop3d_joint` 0.1, `v9_joint_drop` 0.1. `ARTS.get_model`: `v9_e2e` → `V9Net`. `base.load_model_weights`: không lọc legacy cho cả `mode` 'v9'.
- YAML `config/train_v9_3dpw.yml` (exp `v9_3dpw`): 1 GPU, batch 64, 32k mẫu / epoch (500 iter) × 20 epoch; lr 1e-4 (backbone ×0.1), giảm ×0.5 ở ep 8 / 12 / 15 / 18; warmup 0.
- Chạy: `python main/train.py --cfg ./config/train_v9_3dpw.yml --exp_name v9_3dpw --gpu 0`; test: `python -u main/test_student_3dpw_sets.py --cfg ./config/train_v9_3dpw.yml --ckpt experiment/v9_3dpw/checkpoint/last.pth.tar --sets 3dpw --gpu 0`.
- Mốc đọc kết quả: cột LIFT (lifter no3dpw, 3DPW ~82 MPJPE 14 khớp theo các lần trước — cần lấy đúng số trong log). v9 có ích nếu mesh < LIFT rõ rệt.
- **Kết quả v9_3dpw** (người dùng gửi 2026-11-14 22:55, đánh giá trong lúc train, 3DPW 14 khớp, ~18.5 phút / epoch train, 2.22 s / iter):
  - ep19: 86.56 / 54.09 / 102.36; **ep20 (cuối): 86.36 / 54.29 / 102.03** (MPJPE / PA-MPJPE / MPVPE).
  - Train ep20: loss 2.18; ctx 3D 0.0086; smpl3d 0.020, mesh 0.026 (loss đã qua chuẩn hóa, không phải mm); lỗi lifter đầu vào trên train 74.8 mm.
  - Nhận xét (Claude, GIẢ THUYẾT): loss train rất nhỏ trong khi test 86 → khả năng cao QUÁ KHỚP (3dpw-train ít người / ít sequence, mô hình nhiều tham số + backbone train). Chưa có cột LIFT trên test và đường cong theo epoch để xác nhận; kém M0 (88, dữ liệu 4 bộ) chỉ ~1.6 mm dù dữ liệu khác hẳn → chưa so trực tiếp được.
  - **Test chính thức `last` + chẩn đoán tầng** (người dùng 23:16, `test_student_3dpw_sets.py --sets 3dpw`, n = 35515, 14 khớp): mesh 86.36 / 54.29 / 102.03 (trùng eval ep20). Tầng: **lift 82.03 / PA 53.84 → ctx (khớp đã fuse) 80.68 / PA 53.60 → mesh 86.36 / PA 54.29**.
  - Nhận xét (Claude): (1) khối fuse CÓ cải thiện khớp nhưng nhỏ: −1.35 MPJPE, −0.24 PA so với lifter. (2) Đường khớp → mesh bằng hồi quy 6D MẤT 5.7 mm MPJPE nhưng chỉ 0.7 mm PA → lỗi thêm chủ yếu là phần cứng (hướng toàn thân / tỉ lệ cơ thể), không phải tư thế khớp (giả thuyết, suy từ chênh MPJPE vs PA). → Xác nhận cần đường hình học khớp → mesh (IK từ khớp Student đã fuse).

## 36. v9 + reliability (độ tin cậy từng khớp) — chạy song song v9_3dpw trên server thứ 2 (2026-11-14 12:36, người dùng duyệt, đã code, chưa chạy)
- Ý tưởng (Claude đề xuất): lifter no3dpw sai thật trên 3dpw-train (log v9: `lift` 72–75 mm) và có GT → dạy Student ĐOÁN lỗi của từng nguồn ở từng khớp rồi dùng để định tuyến fuse. Tính mới (chưa tra tài liệu): giám sát độ tin cậy của lifter đóng băng bằng lỗi thật trên miền nó chưa thấy, định tuyến fuse theo khớp.
- Code (backup `_to_delete/bak_v9rel_20261114/`: `fusekin_v9.py`, `base.py`, `config.py`): key `MODEL.v9_reliability` (False), `v9_rel_w` (0.1).
  - `fusekin_v9.py`: `rel_head` MLP([ảnh tại khớp, e2d, e3d]) → 2 số / khớp = log lỗi dự đoán [3D lifter (m), 2D Refiner (đơn vị [-1, 1])], bias khởi tạo log 0.07 / log 0.06. Cổng g = sigmoid(w · (−log lỗi dự đoán, detach) + b) (w = 1, b = 0 lúc đầu → g ≈ 0.94) nhân vào e3d / e2d TRƯỚC modality dropout. Output thêm `v9_rel_pred`, `v9_rel_gate`, `v9_ref_xy`.
  - `base.py`: `_v9_rel_loss`: nhãn = log ‖lifter − GT 3D‖ (root-relative, khớp hợp lệ `fit_joint_trunc × is_valid_fit`) và log ‖xy Refiner − GT 2D‖ (`orig_joint_img`, mask `orig_joint_trunc × orig_joint_valid`), smooth-L1, × `v9_rel_w`. Log thêm `rel`, `r3d` (tương quan Pearson log lỗi dự đoán / thật, 3D), `g3`, `g2` (cổng trung bình).
- YAML `config/train_v9_3dpw_rel.yml` (exp `v9_3dpw_rel`) = `train_v9_3dpw.yml` + `v9_reliability: True`, `v9_rel_w: 0.1`. Chạy: `python main/train.py --cfg ./config/train_v9_3dpw_rel.yml --exp_name v9_3dpw_rel --gpu 0`.
- Theo dõi: `r3d` phải tăng dần (> ~0.5 là head đoán được lỗi); `g3` phải giảm ở khớp lỗi lớn (hiện chỉ log trung bình). So v9_3dpw ở epoch cuối. Kết quả: chưa có.

## 37. v9 + IK — mesh bằng IK từ khớp Student đã fuse (2026-11-14 23:26, người dùng duyệt "dùng IK", đã code, chưa chạy)
- Lý do: chẩn đoán v9 (§35): khớp đã fuse 80.68 / PA 53.60 (tốt hơn lifter 82.03) nhưng mesh 6D 86.36 / PA 54.29 → hồi quy 6D mất 5.7 mm, chủ yếu phần toàn cục. Các phương án đã bàn: A IK toàn phần (chọn), B root / β giải tích + 6D cục bộ, C IK + phần dư 6D, D hồi quy đỉnh mesh, E fitting unroll; JOTR dùng VPoser (root 6D riêng + latent 32 chiều qua decoder VPoser đóng băng) — FuseKin từng dùng VPoser ở kiến trúc cũ, lý do bỏ không ghi trong file này.
- Code (backup `_to_delete/bak_v9ik_20261114/`: `fusekin_v9.py`, `config.py`): key `MODEL.v9_ik` (False). Khi bật: J29 = M × khớp đã fuse (`ctx_3d`) + delta 7 điểm đầu chi (head riêng / điểm, từ token khớp, init 0); β = β_xương(khớp, no-grad) + dβ (CascadeHead từ token shape), clamp ±5; twist (cos, sin) += CascadeHead từ 23 query SMPL (khớp 1..23); camera như cũ; IK HybrIK mỗi lớp decoder (refine_history). Không tạo head 6D / shape khi bật. Output `v7_j29` / `v7_dbeta` / `v7_beta0` → Student_Trainer tự thêm loss J29 (7 điểm) + |dβ|². Assert `v7_no_pose_param_loss` True.
- Lưu ý (đọc code `hybrik_lbs.batch_inverse_kinematics_transform`): khớp vào IK bị `detach` → khớp đã fuse chỉ học từ loss `ctx_3d` (+ loss J29 qua M); loss mesh dạy twist / dβ / camera / delta. SVD hướng pelvis chạy trên CPU (`.cpu()`) → có thể chậm thêm.
- YAML `config/train_v9_3dpw_ik.yml` (exp `v9_3dpw_ik`) = `train_v9_3dpw.yml` + `v9_ik: True`, `v6_m_path`, `v7_bone_gn_iters` 3, `v7_bone_beta_reg` 0.001, `v7_dbeta_w` 0.01, `v7_j29_loss_w` 1.0, `v7_no_pose_param_loss: True`.
- Kỳ vọng (giả thuyết): mesh ≈ mức khớp đã fuse (~81) thay vì 86. Kết quả: chưa có.

## 38. v9 + IK + lấy mẫu lại theo phép chiếu (F1) — server thứ 3 (2026-11-14 23:33, người dùng duyệt, đã code, chưa chạy)
- Bối cảnh: người dùng chạy song song 3 run: `v9_3dpw_rel` (§36, đang chạy), `v9_3dpw_ik` (§37), `v9_3dpw_ik_rp` (mục này). Các ý tưởng đã bàn cho contribution fusion / HyperGCN: F1 lấy mẫu lại theo phép chiếu (chọn), F2 attention thiên lệch theo độ tin cậy, H1 HyperGCN điều biến theo độ tin cậy, H2 tham số hóa khớp theo xương. Chọn F1 vì nhắm vào nguồn lỗi lớn nhất (2D) và không phụ thuộc kết quả reliability.
- Code (backup `_to_delete/bak_v9rp_20261114/`: `fusekin_v9.py`, `base.py`, `config.py`): key `MODEL.v9_reproj` (False), `v9_reproj_w` (1.0).
  - Sau lớp fuse 1 và 2 (không làm sau lớp cuối): khớp 3D tạm = `j3d_head(token)` → fit weak-perspective (s, t) đóng kín vào vị trí 2D hiện tại, trọng số 1 / b² (b từ `log_b` Refiner) → `xy_chiếu` (vị trí hợp cấu trúc 3D). Vị trí mới = xy hiện tại + Linear([token, xy_chiếu − xy hiện tại (detach)]) (khởi tạo 0 → lúc đầu giữ xy Refiner), clamp ±1.2 → lấy mẫu lại F16 tại đó, cộng vào token qua Linear (khởi tạo 0), cập nhật PE vị trí.
  - `base._v9_reproj_loss`: smooth-L1 (β 0.05) giữa các vị trí lấy mẫu và GT 2D (`orig_joint_img`, mask `orig_joint_trunc × orig_joint_valid`), trung bình theo lớp, × `v9_reproj_w`. Log `rp2d: <px Refiner> -> <px lớp cuối>` (crop 256).
- YAML `config/train_v9_3dpw_ik_rp.yml` (exp `v9_3dpw_ik_rp`) = `train_v9_3dpw_ik.yml` + `v9_reproj: True`, `v9_reproj_w: 1.0`.
- Theo dõi: `rp2d` lớp cuối phải < px Refiner (vị trí lấy mẫu sửa được 2D); tầng `ctx` lúc test phải < `ctx` của v9_3dpw_ik. Kết quả: chưa có.

## 39. Chẩn đoán lifter no3dpw trên 3DPW-test với 2D OpenPose (2026-11-15 18:49, người dùng chạy và báo)
- Lệnh: `python main/eval_img_lifter.py --cfg config/train_img_lifter_dense48_markov_no3dpw.yaml --ckpt experiment/img_lifter_dense48_markov_no3dpw/checkpoint/last.pth.tar --sets 3dpw --K 5 --S 2 --gpu 0`. Lifter epoch 25, Refiner dense48 scratch epoch 20, 2D OpenPose, n = 35515, K = 5, S = 2. Lưu `experiment/_eval/eval_img_lifter_11-15_18-36-19/eval_img_lifter.json`.
- Kết quả (người dùng báo, **MPJPE17**, root-relative, mm): mean 74.66 / PA 49.72; jpma 74.68 / 50.08; một giả thuyết đơn lẻ (hyp_don_le) 74.96 / 49.84; oracle min-K 71.07 / 48.88. Độ lệch 2D (fit weak-perspective, px crop 256): mean 4.62, jpma 4.48, giả thuyết đơn lẻ 4.64, oracle 4.54; mức sàn `gt3d_san` 3.01 px. Khớp detector sót / có (mean): 90.97 / 71.38 mm (tỉ lệ khớp sót 16.7%); jpma 91.62 / 71.28; oracle 85.29 / 68.21.
- Người dùng nhớ lại (chưa có log, chưa ghi nhận như kết quả): lifter train / test với GT 2D cho MPJPE hơn 40 mm một chút. Cần chạy `config/eval_img_lifter_no3dpw_gt2d.yaml` để có số thật.
- Nhận xét (Claude, giả thuyết, chưa đo trực tiếp):
  1. Trung bình K = 5 giả thuyết chỉ hơn một giả thuyết đơn 0.3 mm → các giả thuyết gần giống nhau, lifter hoạt động gần như bộ hồi quy xác định; oracle min-K chỉ thấp hơn mean 3.6 mm.
  2. Độ lệch 2D của pose lifter chỉ hơn mức sàn khoảng 1.6 px trong khi MPJPE17 = 74.66 mm → phần lớn lỗi có thể nằm theo độ sâu, không phải trong mặt phẳng ảnh. Cần lỗi theo trục (x, y, độ sâu) để xác nhận.
  3. Khớp có detector vẫn sai 71 mm, khớp sót chỉ hơn ~20 mm → lỗi không chủ yếu do khớp bị sót.
- Lưu ý thang đo: số này là MPJPE17; cột `lift` 82.03 trong §35 cùng thang với mesh (14 khớp). Không so trực tiếp hai số (quy ước dự án). Mối quan hệ 17 ↔ 14 khớp là suy luận, chưa kiểm tra.
- Chưa có: (a) lifter với GT 2D (`eval_img_lifter_no3dpw_gt2d.yaml`); (b) `eval_refiner2d.py` (lỗi 2D của Refiner so với OpenPose); (c) lỗi lifter theo trục và theo từng khớp (cần script mới, chưa viết, chờ người dùng duyệt).
- Ngữ cảnh thảo luận cùng ngày (không phải kết quả): ablation DPMesh (người dùng gửi bảng, chưa đối chiếu paper) cho thấy chỉ ảnh 87.6 / 54.9 → thêm heatmap concat 75.1 / 49.8 → thêm joint làm prompt 73.6 / 47.4 (3DPW); heatmap cộng vào z0 thay vì concat 100.3 / 62.0. JOTR (số trên OC, nhãn cột bảng bị đảo nên chỉ dùng xu hướng): thêm feature 3D −0.8 MPJPE, contrastive J2N + J2J −1.3.

## 40. Lifter no3dpw với GT 2D + lỗi 2D của Refiner trên 3DPW-test (2026-11-15 19:08, người dùng chạy và báo)
- (a) Lifter với GT 2D: `python main/eval_img_lifter.py --cfg config/eval_img_lifter_no3dpw_gt2d.yaml --ckpt experiment/img_lifter_dense48_markov_no3dpw/checkpoint/last.pth.tar --sets 3dpw --K 5 --S 2 --gpu 0` (n = 35515; lưu `experiment/_eval/eval_img_lifter_11-15_18-56-30/`). MPJPE17 / PA: mean **76.12 / 47.95**; jpma 76.42 / 48.63; một giả thuyết đơn 76.48 / 48.10; oracle min-K 72.15 / 47.07. Độ lệch 2D (px): 4.64 / 4.50 / 4.66 / 4.56 (mức sàn 3.01). Khớp sót / có (mean): 152.80 / 75.46 mm, tỉ lệ khớp sót 0.9%.
  - So với §39 (cùng lifter, 2D OpenPose): mean 74.66 / 49.72 → GT 2D **không giảm MPJPE17 (tăng 1.46 mm)**, PA tốt hơn 1.77 mm. Số "hơn 40 mm" người dùng nhớ **không tái hiện** với checkpoint này (có thể là thiết lập khác: lifter train bằng GT 2D / tập khác). Đính chính §39: dòng "người dùng nhớ lại" không còn dùng làm cơ sở.
- (b) Refiner 2D: `python main/eval_refiner2d.py --cfg config/train_refiner2d_jotr_mesh_dense48_scratch.yaml --ckpt experiment/refiner2d_dense48_scratch/checkpoint/last.pth.tar --sets 3dpw --gpu 0` (n = 35515; lưu `experiment/_eval/eval_refiner2d_11-15_19-05-22/`). 12 khớp tay chân, lỗi tính theo % kích thước người, "xa" = lỗi > 0.2 × kích thước người:
  - op: lỗi 5.55%, xa 0.21%, bỏ 4.87%, mẫu có khớp xa 1.71%. refine: lỗi 3.25%, xa 0.07%, bỏ 4.87%, mẫu có khớp xa 0.49%. refine_fill: 3.41 / 0.28 / 0.84 / 2.67. op_fill: 5.62 / 0.41 / 0.84 / 3.86. op_fill_all: 5.67 / 0.53 / 0.00 / 5.04.
  - op → refine: sửa 605 khớp xa, làm hỏng 55 khớp; điền 77.2% khớp bị sót (b ≤ 3.0); corr(log b, log lỗi) = 0.209. Điểm dày (đỉnh mesh): lỗi 3.75%, xa 0.52%, 1,681,651 điểm.
  - Lỗi theo khớp (op / refine / |refine − op|, % kích thước người): Pelvis 9.34 / 2.67 / 8.02; R_Hip 10.19 / 3.02 / 8.84; R_Knee 6.11 / 3.02 / 4.68; R_Ankle 5.31 / 2.99 / 3.83; L_Hip 9.01 / 2.93 / 7.91; L_Knee 5.18 / 2.97 / 3.93; L_Ankle 4.78 / 3.12 / 3.21; Neck 6.37 / 3.29 / 3.46; Nose 4.99 / 3.67 / 6.35; L_Shoulder 3.60 / 3.50 / 1.85; L_Elbow 5.21 / 3.55 / 3.10; L_Wrist 3.90 / 3.64 / 2.35; R_Shoulder 3.38 / 3.20 / 2.08; R_Elbow 4.95 / 3.36 / 3.06; R_Wrist 4.41 / 3.81 / 2.98.
- Nhận xét (Claude, giả thuyết): (1) với lifter + Refiner hiện tại, 2D chính xác hơn không giảm lỗi 3D → nút thắt không phải chất lượng 2D; đổi detector (ViTPose) cho lifter nhiều khả năng ít giúp. (2) Lỗi lifter nghiêng về độ sâu: pose chiếu lệch 2D chỉ hơn mức sàn ~1.6 px ở cả hai kiểu đầu vào, trong khi MPJPE17 ~75 mm; cần lỗi theo trục để xác nhận (script chưa viết). (3) Phần lớn khoảng cách op → refine nằm ở hông / pelvis (8–9% kích thước người): phù hợp với việc Refiner đổi định nghĩa khớp detector sang định nghĩa mesh (giả thuyết, chưa xác minh định nghĩa). (4) Có thể GT 2D (định nghĩa mesh) lệch nhẹ so với đầu vào mà Refiner quen (định nghĩa detector) → giải thích một phần việc không cải thiện; chưa kiểm chứng.
- **Đính chính §40 (2026-11-15 19:12, theo người dùng):** (1) Nhận xét (1) ở trên SAI về logic: lifter no3dpw được train với 2D lỗi nên đã học ít tin vào tọa độ 2D; đưa GT 2D vào nó (76.12) không nói gì về mức quan trọng của chất lượng 2D. (2) Số "hơn 40 mm" của người dùng là lifter train VÀ test bằng GT 2D (thiết lập khác, chưa có log, chưa ghi nhận như kết quả); đó mới là trần khi lifter được phép tin 2D. Giả thuyết (Claude): độ sâu suy ra từ chênh lệch nhỏ trong 2D nên rất nhạy với sai số vài % kích thước người; GT 2D nhất quán hình học và cùng định nghĩa khớp với nhãn, nên detector thật (kể cả ViTPose) sẽ không đạt mức đó; số thực nằm giữa ~40 và ~75 mm, chưa biết. (3) Nhận xét (2), (3) về lỗi nghiêng về độ sâu và về hông / pelvis (Refiner đổi định nghĩa khớp) vẫn là giả thuyết như cũ. (4) Cần: log lifter train GT 2D; lỗi 2D của detector khác (vd ViTPose) trên 3DPW-test theo cùng bảng `eval_refiner2d.py` để biết dư địa trước khi train lại lifter với nhiễu tương ứng.

## 41. Lifter train + test bằng GT 2D (trần khi lifter được tin 2D) (2026-11-15 19:16, người dùng gửi log train)
- Config `config/train_img_lifter_dense48_markov_no3dpw_ftref_teacher_gt2d.yaml`: ImgLifter dense48 + Markov (gate b), num_ctx 48, Refiner fine-tune cùng lifter (lr head 1e-4, backbone 1e-5), w_bone 0.5, w_angle 0.1, lr 2e-4, lr_step [18, 23], 25 epoch, batch 32. Log người dùng gửi dừng ở epoch 20 (chưa có `last`).
- Đánh giá nhanh mỗi epoch (EMA, K 5, S 2, **tập con**: 3DPW n = 3000, PC n = 2218, Crowd n = 1923), MPJPE17 / PA (mean), người dùng báo:
  - ep20: **3DPW 55.39 / 36.60**, PC 53.61 / 34.83, Crowd 49.25 / 30.54. Oracle min-K: 52.50 / 51.66 / 47.83. Tỉ lệ khớp sót 1.2% / 0.9% / 0.3% (đầu vào test là GT 2D).
  - Diễn biến 3DPW mean: ep1 86.71 → ep2 68.45 → ep5 60.06 → ep10 58.26 → ep17 57.09 → ep18 (lr ×0.1) 56.06 → ep20 55.39.
  - Train ep20: MPJPE x0 (mọi t) 28.46 mm, lỗi 2D Refiner 0.82 px (đầu vào GT) / 2.16 px so target mesh.
- Đính chính số "hơn 40 mm" (§39, §40): số ~28–40 mm là **loss train** (MPJPE x0 mọi t); trên 3DPW test với GT 2D lifter đạt ~55 MPJPE17 (tập con 3000 mẫu).
- Nhận xét (Claude, giả thuyết): (1) Với 2D hoàn hảo và lifter được train để tin 2D, MPJPE17 3DPW ~55 so với 74.66 (OpenPose, toàn bộ 35515 mẫu, §39) → dư địa từ chất lượng 2D ~19 mm (không cùng tập mẫu, chỉ ước lượng). (2) Ngay cả GT 2D vẫn còn ~55 mm → phần còn lại là mơ hồ độ sâu / hướng mà 2D không giải được. (3) Detector thật (kể cả tốt hơn OpenPose) sẽ nằm giữa 55 và 75.
- Người dùng: detector Halpe (đã thử trước đây) cho kết quả tốt hơn OpenPose nhưng vẫn lỗi khi bị che → không dùng nữa (số chưa có trong mục này).
- **Đính chính §41 (Claude, 2026-11-15 19:19):** (1) Run này CHÍNH LÀ teacher §31 (đã ghi từ 2026-11-12); §39–§41 lặp lại phát hiện §27 / §31 / §32 — Claude không đọc lại lịch sử trước khi phân tích. (2) "Dư địa ~19 mm" là SAI: so sai tập mẫu. Đánh giá nhanh dùng 3000 mẫu đầu 3DPW; trên CÙNG tập con, lifter OpenPose cho ~86–87 (§32) so với teacher ~55 → khoảng cách ~30 mm như §31 đã ghi. 74.66 (§39) là toàn bộ 35515 mẫu, không so trực tiếp được. (3) Kết luận đã có từ §32 vẫn đứng: khoảng ~30 mm đến từ chất lượng 2D LÚC TEST (khớp bị che / nhầm người), không phải cách lifter dùng 2D; KD (f8 + token + InfoNCE) và PoseFix 0.5 không giúp.

## 42. Kết quả v9_3dpw_ik (§37) và v9_3dpw_ik_rp (§38) — đủ 20 epoch (2026-11-15 19:22, người dùng gửi log train)
- Giao thức nhanh (train 3dpw-train, test 3dpw-test toàn bộ, đánh giá mỗi epoch trong lúc train, 14 khớp, MPJPE / PA / MPVPE). Báo epoch cuối (ep20, `last`).
  - **v9_3dpw_ik ep20: 82.59 / 54.99 / 100.22.** Ba epoch cuối: ep18 80.09, ep19 84.11, ep20 82.59 (dao động ±2 mm giữa epoch).
  - **v9_3dpw_ik_rp ep20: 80.41 / 54.14 / 97.11.** Ba epoch cuối: ep18 80.83, ep19 82.37, ep20 80.41.
  - Mốc v9_3dpw (6D, §35) ep20: 86.36 / 54.29 / 102.03.
- Log train: `lift` (lỗi lifter trên batch train) ~75 mm ở cả hai run (không đổi, lifter đóng băng). ik_rp: `rp2d` (lỗi 2D trên 3dpw-train, px crop 256) Refiner → vị trí lấy mẫu lớp cuối: ep4 8.0 → 5.0, ep5 6.4 → 3.7, ep9 6.6 → 3.1, ep20 7.4 → 3.3 (giảm ~một nửa). J29 lỗi train ep20: ik 76.0 mm, ik_rp 67.7 mm.
- Nhận xét (Claude; 1 seed, dao động giữa epoch ±2 mm nên chênh ≤ 2 mm chưa chắc chắn):
  1. IK so với 6D: MPJPE −3.8, MPVPE −1.8, PA +0.7 → IK sửa phần toàn cục như dự đoán §35/§37 nhưng không cải thiện dáng.
  2. F1 so với IK: MPJPE −2.2, PA −0.85, MPVPE −3.1; đường test ổn định hơn ở các epoch cuối. Trên train F1 sửa được 2D tốt hơn Refiner khoảng một nửa (`rp2d`). Chưa biết trên TEST 2D có được sửa hay không (rp2d chỉ log lúc train).
  3. Cần test chính thức `last` bằng `test_student_3dpw_sets.py` để có tầng lift → ctx → mesh: kiểm tra (a) mesh − ctx với IK đã thu hẹp so với 5.7 mm của 6D chưa; (b) ctx của ik_rp có thấp hơn ik không.
- v9_3dpw_rel (§36): chưa có kết quả.
- **Test chính thức `last` (người dùng 19:38, `test_student_3dpw_sets.py --sets 3dpw`, n = 35515, 14 khớp, MPJPE / PA):**
  - v9_3dpw_ik: mesh 82.59 / 54.99 / 100.22 (trùng eval ep20). Tầng: **lift 82.03 / 53.84 → ctx 82.49 / 55.98 → mesh 82.59 / 54.99.**
  - v9_3dpw_ik_rp: mesh 80.41 / 54.14 / 97.11. Tầng: **lift 82.03 / 53.84 → ctx 81.94 / 56.26 → mesh 80.41 / 54.14.**
  - So v9 (6D, §35): lift 82.03 / 53.84 → ctx 80.68 / 53.60 → mesh 86.36 / 54.29.
- Nhận xét (Claude, 1 seed):
  1. IK đóng khoảng ctx → mesh: 6D +5.7 mm; IK +0.1; IK + F1 −1.5 (mesh tốt hơn khớp đã fuse, nhờ delta đầu chi / twist / β).
  2. Nhưng với IK, khớp đã fuse (ctx) KHÔNG tốt hơn lifter: ik +0.46 MPJPE, PA +2.1 (xấu hơn); ik_rp −0.09, PA +2.4. Bản 6D ctx tốt hơn lifter 1.35 / PA −0.24. Giả thuyết: khi bật IK, khớp vào IK bị detach nên token khớp mất gradient từ loss mesh, chỉ còn loss ctx_3d + J29 → khối fuse làm dáng khớp xấu đi (PA). Chưa kiểm chứng.
  3. F1 trên test: ctx chỉ −0.55 so với ik (train `rp2d` giảm một nửa) → lợi ích sửa 2D lúc train gần như không chuyển sang test (có thể quá khớp 3dpw-train, hoặc 2D test không được sửa — chưa đo).
  4. Tổng: mesh tốt nhất (80.41) chỉ hơn lifter 1.6 MPJPE; PA mesh (54.14) vẫn xấu hơn lifter (53.84). Ở giao thức 3dpw-only, Student chưa lấy thêm được thông tin dáng ngoài lifter.

## 43. Đọc code DPMesh (2026-11-15 20:00, chuẩn bị lifter dùng lõi DPMesh; chưa code)
- Bối cảnh: người dùng muốn thử lifter mới = encoder ảnh của DPMesh (Stable Diffusion + 2D + PoseFix) + decoder của ImgLifter hiện tại (self-attn + Markov + cross-attn ảnh), ra 17 khớp 3D. Chưa chốt thiết kế.
- Backbone (`DPMesh/common/vpd/vpdencoder_useattn.py`, `cldm/cldm.py`, `cldm/module_utils.py`, YAML `common/vpd/cldm_v15.yaml`): SD 1.5 `ControlLDM`. Ảnh 256 → VAE encoder (đóng băng, `.mode()`, không thêm nhiễu) → latent 4×32×32; t = 1, một bước. ControlNet: concat heatmap 30 khớp (32×32, σ 2.5, nhân mask khớp, train che ngẫu nhiên 10% khớp) vào latent ở đầu vào, ra residual cộng vào skip của U-Net. "Prompt" cross-attn = Linear(2 → 768) trên tọa độ khớp 2D (thay CLIP; CLIP vẫn được tạo nhưng đóng băng). U-Net SD đóng băng + LoRA r = 64 (train). Đặc trưng lấy từ decoder U-Net (4×4, 8×8, 16×16, 32×32) + bản đồ cross-attention trung bình → gộp về **2048×8×8** (`outmodules`).
- Head mesh: conv → khối 3D 8×8×8 (neo độ sâu tương đối) + transformer encoder; transformer enc-dec 2 / 4 lớp, 15 query khớp + pose / shape / cam, VPoser đóng băng, cascade theo lớp.
- Train (`configs/main_train.yml`): Human36M + MuCo (3D) + MSCOCO + CrowdPose (2D), 256×256, batch 8, 20 epoch, lr 2e-4 (fine_tune.yml 5e-5).
- Điểm cần lưu ý (Claude đọc code, chưa chạy):
  1. **Dataloader train chưa phát hành** (README; `dataloader/` chỉ có PW3D). Phải tự ghép dataloader FuseKin.
  2. Optimizer (`common/base.py` `get_optimizer`): train LoRA U-Net + `outmodules` + phần ngoài backbone; dòng thêm **ControlNet (`control_model`) bị comment** → trong code phát hành ControlNet KHÔNG được cập nhật. Không có code nạp trọng số SD / ControlNet gốc (đều comment) → trọng số chỉ đến từ checkpoint DPMesh. Chưa rõ ControlNet được train ở giai đoạn nào; nếu train lại từ SD gốc cần tự quyết (đề xuất: train ControlNet).
  3. Checkpoint phát hành là `3dpw_best`, `3dpw-oc_best`, `3dpw-pc_best`, `3dpw-crowd_best` (mỗi tập một checkpoint, tên "best") — chưa có trên máy local (`DPMesh/checkpoints` không tồn tại).
  4. Phụ thuộc: `ldm`, `cldm`, `lora_diffusion`, CLIP (FrozenCLIPEmbedder), SD 1.5.

## 44. SD lifter mức A0 — code xong, chưa chạy (2026-11-15 21:xx, người dùng duyệt thiết kế; Claude viết code, chưa chạy / chưa kiểm tra cú pháp)
- **Thiết kế:** giữ nguyên decoder ImgLifter (self-attn 17 khớp + 48 token ngữ cảnh, Markov, cross-attn, FFN, diffusion x0) và Refiner dày **đóng băng** (`refiner2d_dense48_scratch`): xy / log_b / feat_j như lifter no3dpw. **Chỉ thay `f8`** (64 token ảnh cho cross-attn) từ ResNet-50 của Refiner sang đặc trưng Stable Diffusion 1.5 + ControlNet theo DPMesh (`lib/models/sd_feature.py`).
- **SD encoder (theo DPMesh `vpdencoder_useattn.py` + `model.py`):** ảnh 256 RGB [0,1] → VAE encoder (no_grad, `mode()`, không nhân 0.18215 như DPMesh) → latent 4×32×32; heatmap 17 kênh 32×32 (σ 2.5) từ xy Refiner (khớp det = 0 → 0; che 10% = `input_drop_prob` sẵn có); prompt `Linear(2→768)` trên tọa độ/32; ControlNet nhận cat(latent, heatmap) (không input_hint_block), t = 1; U-Net đóng băng + LoRA r64 (to_q/k/v/out của mọi CrossAttention + GEGLU.proj); đặc trưng output_blocks 1/4/7/cuối + bản đồ cross-attention (up+down, 8×8 và 16×16) → `outmodules` prior **[320, 657, 2577]** (DPMesh 30 khớp: [320, 670, 2590]) → f8 2048×8×8.
- **Trọng số:** `control_sd15_openpose.pth` (ControlNet 1.0, gồm SD 1.5 + ControlNet), giống dòng (bị comment) của DPMesh. Conv đầu ControlNet đổi 4 → 4+17 kênh: `sd_ctrl_in_init: copy_latent` chép 4 kênh latent + bias, kênh heatmap khởi tạo mặc định PyTorch (**khác DPMesh**, DPMesh bỏ hẳn trọng số này → ngẫu nhiên). Không CLIP.
- **Mức A0** (`sd_train_control: none`): VAE, U-Net, ControlNet (cả conv đầu) đóng băng; train LoRA + outmodules + kp_lifting + lifter. Mức A (`in_zero`) / B (`full`) đã có key, chưa chạy.
- **Train:** AdamW lr **hằng** 2e-4 cho cả lifter và SD (DPMesh `main_train.yml` lr_dec_factor 1; lưu ý: `train.py` DPMesh ghi đè lr bằng argparse mặc định 1e-4 nếu không truyền `--lr` → lr thật của DPMesh chưa xác định), wd 0.01, **20 epoch**, batch 16 × `grad_accum` 4 = 64 hiệu dụng (1000 bước/epoch như lifter cũ). Autocast bf16, trọng số đóng băng lưu bf16. Dữ liệu Human36M/MuCo/MSCOCO/CrowdPose (không 3dpw-train).
- **Yếu tố gây nhiễu khi so với lifter no3dpw (§39: 74.66 / 49.72, K5 S2, 3DPW đủ):** lifter cũ 25 epoch, lr ×0.1 ở 18/23; bản SD 20 epoch lr hằng → chênh < ~2 mm không kết luận.
- **Chẩn đoán sớm:** `sd_eval_ablate: True` → mỗi epoch thêm quick-eval 3DPW (3000 mẫu) với f8 SD = 0 (giữ feat_j). Nếu gần bằng bản đầy đủ → SD không đóng góp → dừng sớm.
- **File:** mới `lib/models/sd_encoder/{blocks,unet,vae,lora}.py`, `lib/models/sd_feature.py`, `config/train_img_lifter_sd_a0.yaml`; sửa `lib/core/config.py` (key `IMGLIFT.sd_*`, `grad_accum`, mặc định tắt / 1), `lib/utils/img_lifter_tools.py`, `main/train_img_lifter.py`, `main/eval_img_lifter.py` (backup `_to_delete/bak_sdlifter_20261115/`).
- **Chưa có kết quả.**
- **Quyết định (2026-11-15 21:59, người dùng):** run `img_lifter_sd_a0` đã chạy (1.12 it/s ≈ 60 phút/epoch, ≈ 21–23 giờ cho 20 epoch; bộ nhớ dùng 36 GB / còn ~85 GB). **Hướng chính vẫn dùng lifter cũ** (no3dpw); SD lifter chỉ thay vào khi thực sự hiệu nghiệm (tiêu chí: 3DPW đủ, epoch cuối, tốt hơn ≥ 2 mm so với 74.66 / 49.72 và `--ablate_sd` cho thấy SD có đóng góp).

## 45. Kết quả v9_3dpw_rel (v9 6D + reliability, resume ep16 → 20) (2026-11-15 22:24, người dùng gửi log train)
- Config `train_v9_3dpw_rel.yml` = `train_v9_3dpw.yml` + `v9_reliability` (w 0.1); **pose 6D, không IK, không F1**. Bị dừng ep16 do lỗi máy, resume `--resume_training`.
- Eval 3DPW (35515 mẫu, 14 khớp, eval trong vòng train) theo epoch, MPJPE / PA / MPVPE:
  ep16 85.09 / 53.76 / 101.18 · ep17 85.97 / 54.68 / 102.55 · ep18 84.53 / 53.66 / 100.65 · ep19 85.24 / 53.93 / 101.16 · **ep20 (last) 85.29 / 53.55 / 101.27**.
- So v9 6D (§42, last): 86.36 / 54.29 / 102.03 → **−1.07 / −0.74 / −0.76 mm**. Dao động ep16–20 của chính run này 84.53–85.97 → chênh **nằm trong nhiễu** (1 seed).
- Thống kê train (ep20): `r3d 0.91` = tương quan Pearson log lỗi 3D lifter dự đoán / thật **trên tập train**; `g3 0.90`, `g2 0.96` = cổng trung bình cho 3D / 2D (gần 1 → gần như không giảm trọng số).
- Nhận xét (giả thuyết, Claude): head dự đoán được lỗi lifter trên train (r3d cao) nhưng cổng gần mở hoàn toàn → mô hình ít dùng thông tin tin cậy; cải thiện ≤ ~1 mm, không đủ kết luận. r3d trên test chưa đo. Chưa có số tách lift / ctx / mesh (cần `test_student_3dpw_sets.py`).

## 46. Kế hoạch v9_full_ik_rp — IK + F1 trên giao thức đủ dữ liệu (2026-11-15 22:26, người dùng duyệt; chưa chạy)
- `config/train_v9_full_ik_rp.yml` = `train_v9_3dpw_ik_rp.yml` (§42, 3dpw-only last 80.41 / 54.14 / 97.11), **chỉ đổi dữ liệu**: train Human36M / MuCo / MSCOCO / CrowdPose (0.25 mỗi bộ, 2D GT + PoseFix), không 3dpw-train; `samples_per_epoch` 64000 (1000 iter/epoch, batch 64). Giữ kiến trúc, loss, lr 1e-4 (backbone ×0.1, ×0.5 ở 8/12/15/18), 20 epoch, lifter no3dpw đóng băng.
- Cả pipeline không thấy 3DPW → báo 3DPW / OC / PC (toàn bộ sequence) / Crowd ở `last.pth.tar`, **so được với bảng chính JOTR / DPMesh**. Dùng làm baseline cho các ablation tiếp (reliability, H1, F2, bỏ detach trước IK).
- Ước lượng (chưa đo trên máy này): ~37 phút/epoch → ~12 giờ.

## 47. Kết quả v9_full_ik_rp (§46) — IK + F1, đủ dữ liệu, last ep20 (2026-11-16 19:25, người dùng chạy test và báo)
| tập | n | MPJPE | PA | MPVPE | LIFT | LIFT-PA | lift → token/ctx → mesh |
|---|---:|---:|---:|---:|---:|---:|---|
| 3DPW | 35515 | 82.45 | 55.23 | 101.06 | 82.03 | 53.84 | 82.03 → 81.43 → 82.45 |
| 3DPW-OC (toàn bộ) | 20243 | 86.35 | 59.09 | 104.47 | 83.25 | 57.75 | 83.25 → 84.76 → 86.35 |
| 3DPW-PC (toàn bộ) | 2218 | 98.70 | 69.51 | 123.11 | 94.31 | 69.47 | 94.31 → 98.66 → 98.70 |
| 3DPW-Crowd | 1923 | 92.81 | 60.15 | 116.42 | 90.47 | 60.16 | 90.47 → 91.04 → 92.81 |
- Script ghi nhãn "(toàn bộ, rò rỉ)" cho OC/PC, nhưng ở đây cả pipeline không train 3DPW → không rò rỉ; so được với bảng §2.
- So DPMesh (§2): 3DPW +8.9, OC +15.5, PC +16.5, Crowd +12.9 mm. So 3dpw-only IK+F1 (§42, 80.41 / 54.14 / 97.11): 3DPW xấu hơn 2.0 mm (giao thức khác, chỉ tham khảo).
- Quan sát: **mesh xấu hơn lifter trên cả 4 tập** (MPJPE +0.4 / +3.1 / +4.4 / +2.3; PA 3DPW +1.4). Trên OC / PC / Crowd, fusion (token/ctx) đã xấu hơn lift (+1.5 / +4.4 / +0.6) → Student không sửa được lifter ở vùng che khuất mà còn làm xấu đi; trên 3DPW gần như chép lifter (−0.6 ở ctx).
- Giả thuyết (Claude, chưa kiểm chứng): Student vẫn bám pose 3D của lifter; nhánh ảnh + 2D không đủ mạnh để vượt lifter, nhất là khi bị che.

## 48. Chẩn đoán v9_full_ik_rp TẮT 3D lúc test (`--drop3d`, token 3D = miss3d mọi mẫu / khớp) (2026-11-16 20:05, người dùng chạy và báo)
| tập | có 3D (§47) MPJPE / PA | **tắt 3D** MPJPE / PA / MPVPE | chênh MPJPE / PA | DPMesh MPJPE |
|---|---|---|---|---:|
| 3DPW | 82.45 / 55.23 | 89.59 / 56.25 / 106.95 | +7.1 / +1.0 | 73.6 |
| OC | 86.35 / 59.09 | 91.98 / 59.99 / 109.04 | +5.6 / +0.9 | 70.9 |
| PC | 98.70 / 69.51 | 109.79 / 74.54 / 133.48 | +11.1 / +5.0 | 82.2 |
| Crowd | 92.81 / 60.15 | 103.03 / 62.16 / 122.74 | +10.2 / +2.0 | 79.9 |
- Chuỗi tầng (tắt 3D): 3DPW token/ctx 89.91 → mesh 89.59; OC 91.06 → 91.98; PC 111.25 → 109.79; Crowd 104.04 → 103.03.
- Code: key `MODEL.v9_test_drop3d` (mặc định tắt), cờ `--drop3d` trong `main/test_student_3dpw_sets.py` (backup `_to_delete/bak_v9_testdrop3d_20261116/`).
- Lưu ý: lúc train chỉ 20% mẫu bỏ 3D → số này có thể đánh giá thấp nhánh ảnh + 2D.
- Nhận xét (giả thuyết, Claude): (1) nhánh ảnh + 2D của Student kém DPMesh 16–28 mm MPJPE và ~9 mm PA (3DPW 56.25 vs 47.4) → nút thắt chính là năng lực ảnh + 2D, không chỉ cách dùng 3D; (2) trên 3DPW / OC, bỏ 3D làm MPJPE xấu 6–7 mm nhưng PA chỉ ~1 mm → 3D lifter chủ yếu đóng góp phần bị PA loại bỏ (hướng toàn cục / tỉ lệ), ít đóng góp hình dạng tư thế; trên PC / Crowd 3D giúp cả PA (+5 / +2 mm).

## 49. Kết quả SD lifter A0 (`img_lifter_sd_a0`, §44) — last ep20, K10 S4, OC/PC toàn bộ (2026-11-17 01:24, người dùng chạy và báo)
| tập | SD A0 mean MPJPE17 / PA | lifter no3dpw last (§ dòng 387, K10 S4) | chênh | oracle min-K SD / cũ | khớp sót / có (SD) |
|---|---|---|---|---|---|
| 3DPW (35515) | 73.65 / 48.46 | 74.80 / 49.87 | −1.15 / −1.41 | 66.60 / 67.25 | 91.61 / 70.04 |
| OC (20243) | 72.78 / 51.24 | 75.49 / 53.47 | −2.71 / −2.23 | 65.19 / 67.42 | 89.78 / 69.54 |
| PC (2218) | 83.36 / 62.96 | 85.49 / 64.06 | −2.13 / −1.10 | 76.07 / 77.00 | 111.19 / 74.32 |
| Crowd (1923) | 80.73 / 53.90 | 83.22 / 56.47 | −2.49 / −2.57 | 73.30 / 75.19 | 107.38 / 75.09 |
- jpma / hyp_don_le SD: 3DPW 74.37 / 74.54, OC 73.39 / 74.26, PC 84.41 / 85.10, Crowd 81.56 / 82.49. Lệch 2D 4.55 / 5.05 / 6.12 / 5.70 px.
- Đánh giá theo tiêu chí §44: 3DPW −1.15 (**chưa đạt** ≥ 2 mm); OC / PC / Crowd −2.1 đến −2.7 (đạt), cải thiện lớn hơn trên tập che khuất. Khớp detector sót gần như không đổi (3DPW 91.6 vs 91.1 cũ).
- Yếu tố gây nhiễu còn nguyên: lịch khác (SD 20 epoch lr hằng, batch 16×4; cũ 25 epoch, lr ×0.1 ở 18/23), 1 seed mỗi bên.
- **`--ablate_sd`** (f8 SD = 0, giữ feat_j Refiner; người dùng báo 2026-11-17 10:35), mean MPJPE17 / PA: 3DPW 79.54 / 51.47 (+5.89 / +3.01 so với đủ SD) | OC 82.85 / 56.82 (+10.07 / +5.58) | PC 86.95 / 65.35 (+3.59 / +2.39) | Crowd 87.27 / 58.23 (+6.54 / +4.33). Khớp sót / có 3DPW 99.46 / 75.53 (đủ SD 91.61 / 70.04).
- Nhận xét: lifter **phụ thuộc rõ vào f8 SD** (bỏ đi xấu 3.6–10 mm, lớn nhất ở OC), cả khớp sót lẫn khớp có. Lưu ý: f8 = 0 nằm ngoài phân bố train (lifter chưa từng thấy) → ablate đo mức phụ thuộc, KHÔNG đo phần hơn ResNet; phần hơn ResNet cần run đối chứng §50.

## 50. Đối chứng SD A0: lifter ResNet cùng lịch SD (`img_lifter_sd_a0_ctrl`) (2026-11-17 01:26, người dùng duyệt; chưa chạy)
- `config/train_img_lifter_sd_a0_ctrl.yaml` = `train_img_lifter_sd_a0.yaml`, chỉ `sd_enable: False` (f8 từ ResNet-50 Refiner). Cùng 20 epoch, lr hằng 2e-4, batch 16 × grad_accum 4, dữ liệu, decoder.
- Đọc: SD A0 (§49) − đối chứng = phần do SD; đối chứng − lifter cũ (74.80 / 75.49 / 85.49 / 83.22) = phần do lịch train.
- **Kết quả đối chứng** (người dùng báo 2026-11-17 11:36; dòng tiêu đề không có "f8 = SD" → đúng run không SD; last, K10 S4, OC/PC toàn bộ), mean MPJPE17 / PA:
  | tập | đối chứng (ResNet, lịch SD) | SD A0 (§49) | SD − đối chứng | đối chứng − lifter cũ |
  |---|---|---|---|---|
  | 3DPW | 75.08 / 49.79 | 73.65 / 48.46 | **−1.43 / −1.33** | +0.28 / −0.08 |
  | OC | 75.47 / 53.41 | 72.78 / 51.24 | **−2.69 / −2.17** | −0.02 / −0.06 |
  | PC | 84.07 / 63.60 | 83.36 / 62.96 | −0.71 / −0.64 | −1.42 / −0.46 |
  | Crowd | 81.61 / 55.57 | 80.73 / 53.90 | −0.88 / −1.67 | −1.61 / −0.90 |
  Oracle min-K đối chứng: 67.06 / 67.06 / 76.01 / 73.40. Khớp sót / có 3DPW 92.10 / 71.66.
- Nhận xét (Claude): đổi lịch train gần như không đổi 3DPW / OC (≤ 0.3 mm), giúp PC / Crowd ~1.5 mm (2 tập nhỏ, nhiễu lớn). **Phần do SD**: 3DPW −1.4, OC −2.7 (2 tập lớn, đáng tin hơn), PC −0.7, Crowd −0.9. → SD có lợi thật nhưng vừa phải; 3DPW chưa đạt ≥ 2 mm. Chi phí: SD 1.29 tỷ tham số, ~60 phút / epoch.

## 51. Detector ViTPose cho 3DPW (2026-11-17, người dùng muốn thử detector khác; chưa chạy)
- Bối cảnh: SD lifter chỉ giúp ~1.4–2.7 mm (§50); người dùng muốn thử 2D tốt hơn OpenPose.
- Môi trường: server `arts` (mmpose chính thức 0.29 không có ViTPose) → cài repo ViTPose gốc (`ViTAE-Transformer/ViTPose`, fork mmpose 0.x, backbone `ViT`), nới giới hạn mmcv 1.5.0 → 1.7.0 trong `ViTPose/mmpose/__init__.py` (mmcv 1.7.0 đang có; cài mmcv-full 1.5.0 thất bại do ổ `/` đầy). Trọng số `vitpose-h-multi-coco.pth` (như ARTS) ở `pose_detector/`.
- Tool mới `tools/extract_vitpose_3dpw.py`: ViTPose (API như `ARTS/main/run_demo.py`) → COCO-17 đổi sang **định dạng Halpe-26** (Neck = trung điểm vai, Hip = trung điểm hông, đỉnh đầu + bàn chân score 0 → H36M Head_top bị bỏ như OpenPose), dùng lại hàm chọn bbox / vòng lặp / vẽ hình của `extract_halpe26_3dpw.py`. Ra `3DPW_{split}_halpe26_vitpose{,_opbox}.json`. Dùng trong FuseKin: `pw3d_2d_source: 'halpe26'` + `pw3d_halpe_suffix: '_vitpose'` (không sửa dataset).
- Kế hoạch: (1) thử ít ảnh + xem hình; (2) trích test / oc / pc / validation, bbox `gt` và `openpose_best`; (3) so độ chính xác 2D với OpenPose / Halpe (`compare_2d_sources.py`) trước khi đụng tới lifter.
- **Bổ sung (2026-11-17 14:03, người dùng yêu cầu):** trích ViTPose trên **3DPW train / validation / test / oc / pc** với 2 kiểu bbox: (1) `gt` = bbox annotation (đuôi `_vitpose`); (2) `gt2d` = bbox ôm 17 khớp 2D GT H36M (từ mesh GT, gồm cả khớp bị che) ×1.25 (đuôi `_vitpose_gt2dbox`). Tool mới `tools/dump_pw3d_gt2d.py` (env jotr_new, xuất `3DPW_{tag}_gt2d_h36m.json`, cùng định nghĩa 2D GT với nhánh đánh giá FuseKin); `extract_vitpose_3dpw.py` thêm `--bbox_source gt2d` (backup `_to_delete/bak_vitpose_20261117/`). Ghi chú: bbox (2) dùng 2D GT của tập test. Dataset PW3D chỉ đọc `pw3d_halpe_suffix` cho split ≠ train → file train có đuôi chưa được dataset dùng.

## 52. Chẩn đoán D1 / D2 theo `GOPY.MD` (2026-11-17, người dùng duyệt "code luôn"; đã code, chưa chạy)
- Bối cảnh: `GOPY.MD` (góp ý ngoài) nêu: Student (`V9Head`) không nhận `cam_bbox` (đã xác nhận trên code: `cam_bbox` chỉ vào lifter); HybrIK detach khớp đầu vào (`hybrik_lbs.py:732`, đã xác nhận); fusion làm xấu lifter trên OC / PC / Crowd. Claude đánh giá: phần "mâu thuẫn view_rot kéo ngược chiều" bị phóng đại; các mức "lợi ích kỳ vọng" là ước đoán.
- **D1** (`--view_rot fwd|inv`): xoay mesh / ctx / token của Student quanh gốc bằng `view_rot(dx, dy)` từ `cam_bbox` (hoặc nghịch đảo) trước khi đo. Nếu `--drop3d` + `fwd` giảm MPJPE rõ (PA gần như không đổi) → Student thiếu `cam_bbox` dự đoán trong hệ camera "nhìn thẳng tâm bbox".
- **D2** (`--blend a,b,...`, `--blend_mode global|opmask`, `--oracle_lc`): trộn khớp 3D trước IK `j3d = w·lift + (1−w)·ctx` (key `MODEL.v9_test_blend` mặc định −1 = tắt, `v9_test_blend_mode`); `opmask`: khớp OpenPose sót dùng lifter. `oracle_lc` = mỗi khớp chọn lift / ctx gần GT hơn (chỉ đo trần, không báo cáo). w = 0 phải tái tạo đúng §47 (kiểm tra).
- Code: `lib/core/config.py` (2 key), `lib/models/fusekin_v9.py` (trộn chỉ khi test + key bật), `lib/utils/jotr_evaluation.py` (`post_fn` tùy chọn, `extra_stages`), `main/test_student_3dpw_sets.py` (cờ mới). Backup `_to_delete/bak_d1d2_20261117/`.
- **Bổ sung (2026-11-17 16:54, người dùng chốt):** bản chính cho 3DPW dùng bbox OpenPose (`--bbox_source openpose_best`, đuôi `_vitpose_opbox`). Tool mới `tools/extract_vitpose_train.py`: dùng lại nguyên các tool Halpe `extract_halpe26_{h36m,muco,coco}.py` (chọn người, lọc frame, bbox GT annotation, chia shard / gộp), chỉ thay model bằng ViTPose và thêm hậu tố `_vitpose` vào tên file (`<Name>_<split>_halpe26_vitpose.npz`). 4 bộ này không có OpenPose → bbox GT annotation. Dataset FuseKin chưa đọc file có hậu tố.
- **Kết quả D1** (người dùng báo 2026-11-17 23:21; `v9_full_ik_rp` last, MPJPE / PA / MPVPE 14 khớp; PA không đổi khi xoay → kiểm tra code đúng):
  | tập | tắt 3D (§48) | tắt 3D + `fwd` | tắt 3D + `inv` | có 3D (§47) | có 3D + `fwd` |
  |---|---|---|---|---|---|
  | 3DPW | 89.59 | 91.70 (+2.11) | 97.23 (+7.64) | 82.45 | 89.82 (+7.37) |
  | OC | 91.98 | 93.22 (+1.24) | 100.11 (+8.13) | 86.35 | 95.47 (+9.12) |
  | PC | 109.79 | 111.84 (+2.05) | 112.66 (+2.87) | 98.70 | 102.93 (+4.23) |
  | Crowd | 103.03 | 102.41 (−0.62) | 107.96 (+4.93) | 92.81 | 96.02 (+3.21) |
- Nhận xét (Claude): phép thử đủ nhạy (xoay toàn bộ `view_rot` lên Student có 3D làm xấu +3–9 mm). Khi tắt 3D, xoay `fwd` **không cải thiện** (xấu +1–2 mm, chỉ Crowd −0.6) → Student tắt 3D **không** dự đoán trong hệ camera "nhìn thẳng tâm bbox"; dự đoán đã gần hệ camera thật (`inv` xấu nhiều hơn `fwd` nhiều → lệch, nếu có, chỉ là một phần nhỏ góc). → **Giả thuyết GOPY "thiếu `cam_bbox` gây +4–6 mm khi tắt 3D" KHÔNG được ủng hộ** bởi D1; khoảng MPJPE − PA khi tắt 3D đến từ sai lệch toàn cục khác (hướng / tỉ lệ), không phải đúng góc `view_rot`. Đưa `cam_bbox` vào `V9Head` (H3) vẫn có thể có ích nhưng kỳ vọng thấp hơn GOPY ước.
- **Kết quả D2** (người dùng báo 2026-11-17 23:21; `v9_full_ik_rp` last, 14 khớp; `w = 0` tái tạo đúng §47 → code đúng). MPJPE (ctx = khớp đưa vào IK | mesh):
  | tập | lift | w=0: ctx / mesh | w=0.5 global: ctx / mesh | w=0.5 opmask: ctx / mesh | w=1: ctx(=lift) / mesh | oracle_lc (w=0) |
  |---|---|---|---|---|---|---|
  | 3DPW | 82.03 | 81.43 / 82.45 | 80.19 / 81.87 | 80.17 / 81.79 | 82.03 / 83.43 | 74.88 |
  | OC | 83.25 | 84.76 / 86.35 | 82.15 / 84.71 | 82.18 / 84.86 | 83.25 / 86.04 | 75.64 |
  | PC | 94.31 | 98.66 / 98.70 | 94.95 / 96.48 | 95.05 / 96.58 | 94.31 / 96.51 | 89.03 |
  | Crowd | 90.47 | 91.04 / 92.81 | 89.45 / 92.35 | 89.64 / 92.30 | 90.47 / 94.36 | 83.85 |
  PA mesh: w=0 55.23 / 59.09 / 69.51 / 60.15; w=0.5 global 54.50 / 58.43 / 68.85 / 59.69; w=1 55.21 / 59.46 / 69.45 / 60.62.
- Nhận xét (Claude): (1) **Bước IK mất 1.4–3.9 mm** ngay cả khi đầu vào là pose lifter (w=1: 3DPW +1.4, OC +2.8, PC +2.2, Crowd +3.9) → đường M + IK + twist / dβ là nguồn lỗi riêng (ủng hộ H1). (2) **ctx và lift bổ sung cho nhau:** trung bình 50/50 tốt hơn CẢ HAI ở 3DPW / OC / Crowd (ctx 80.19 vs 82.03 / 81.43; OC 82.15 vs 83.25 / 84.76), mesh tốt hơn w=0 0.5–2.2 mm; oracle chọn từng khớp thấp hơn lift ~7 mm (3DPW / OC) → cổng học theo khớp có tiềm năng lớn (ủng hộ H2). (3) Cổng theo OpenPose sót (`opmask`) ≈ trung bình toàn cục → cờ khớp sót không phải tín hiệu tốt để chọn nguồn. (4) Trộn lúc test chỉ là chẩn đoán, không phải phương pháp báo cáo.

## 53. H1 + H2 — code xong, chưa chạy (2026-11-17 23:52, người dùng duyệt "code đi"; thiết kế `PLAN_H1H2.md` mục 6, sau 3 vòng trao đổi với Gemini trong `GOPY.MD`)
- **H1b** `MODEL.v9_ik_grad`: file mới `lib/models/hybrik_ik_grad.py` — IK khả vi: hướng xương con = xương dự đoán (không detach), bỏ `big_diff`, Pelvis giải tích (Rodrigues 2 bước), không SVD / `.cpu()` / assert, không gán in-place. Phát hiện khi code: trong FuseKin `children_map[9]` (Spine3) bị ghi đè = Neck → Spine3 là khớp 1 con, **không** đi qua nhánh 3 con / SVD (đề xuất giải tích Spine3 của GOPY không cần).
- **H1a** `MODEL.v9_ik_res29`: mỗi lớp decoder cộng phần dư J29 (24 khớp từ 24 query, 5 điểm lá từ query khớp cha; khởi tạo 0); 22 điểm thân `0.04·tanh`, 7 điểm đầu chi tự do; mỗi lớp IK dùng J29 của lớp đó; loss J29 đủ 29 điểm mọi lớp (`v9_j29_all_w`, hệ số `v3_refine_gamma`) + phạt `v9_res29_reg_w`·mean(Δ_thân²). Thay `ik_dlimb` khi bật.
- **H2** `MODEL.v9_gate`: cổng `g = σ(MLP([jt.detach(), ‖lift − j3d_s‖, lỗi chiếu lift, lỗi chiếu j3d_s, log_b, op_mask]))·keep`; IK nhận `g.detach()·lift + (1 − g.detach())·j3d_s.detach()`; `ctx_3d` = j3d_s (loss ctx riêng); BCE với `g*` chuẩn hóa theo từng khớp trên batch (`base._v9_gate_loss`, `v9_gate_w` 0.1). Test: tầng chẩn đoán `mix` (khớp đã trộn).
- **H2b** `MODEL.v9_vis_aux`: lượt fuse thứ hai (chung trọng số) với token 3D = miss3d → `j3d_vis`, loss GT 3D (`v9_vis_aux_w` 0.5); test: tầng `vis`.
- YAML mới (copy `train_v9_full_ik_rp.yml`, chỉ thêm key): `train_v9_full_h1.yml` (E1), `_h2` (E2), `_h1h2` (E3), `_h1h2_vis` (E4). Mốc: `v9_full_ik_rp` last 82.45 / 86.35 / 98.70 / 92.81.
- Sửa: `lib/core/config.py` (9 key, mặc định tắt), `lib/models/fusekin_v9.py` (vòng fuse thành hàm `run_fuse` để dùng lại; D2 `--blend` giờ chỉ đổi khớp vào IK, tầng `ctx` giữ j3d_s, thêm tầng `mix`), `lib/core/base.py` (loss + log). Backup `_to_delete/bak_h1h2_20261117/`.
