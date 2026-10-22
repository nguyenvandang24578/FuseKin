# FuseKin — kết quả và tiến độ thí nghiệm

Cập nhật **2026-10-22**. Số bên dưới lấy từ kết quả người dùng đã cung cấp trong trao đổi và tài liệu lịch sử của dự án; không phải các phép đo vừa được chạy lại. Chưa có đủ đường dẫn checkpoint/seed/epoch cho mọi số, nên không tự gán thêm.

Kiến trúc và giao thức hiện tại: [01_PROJECT_CONTEXT.md](01_PROJECT_CONTEXT.md). Lệnh chạy: [README.md](README.md).

## 1. Trạng thái hiện tại

| Nhánh | Config | Trạng thái / câu hỏi |
|---|---|---|
| Lifter baseline | `finetune_difflifter_r25_cam.yaml` | Đã có checkpoint, dùng làm lifter đóng băng cho các YAML Student diffusion |
| Student diffusion baseline | `train_student_v2_diff.yml` | Mốc S=1, K=1; người dùng báo mesh khoảng 78.2 / 94.4 mm |
| Selector theo khớp bằng ảnh | `train_student_v2_diff_sel.yml` | Có kết quả: pose tốt hơn nhưng mesh chưa tốt hơn; xem mục 3 |
| Augmentation bằng pose diffusion | `train_student_v2_diff_aug.yml` | **Đang train trên server theo người dùng**; chưa có kết quả cuối |
| Spread vào fusion | `train_student_v2_diff_spread.yml` | Có code/YAML; chưa có kết quả được xác nhận trong đợt cập nhật này |
| GCN + attention song song | `finetune_difflifter_r25_cam_gcn_attn.yaml` | **Đã code, đọc lại source; chưa chạy train/test** |
| Prior SimCC/GMM | `probe_halpe26_simcc*.yaml` | Đã probe 32 người tập val; **tạm hoãn** phát triển prior |
| Teacher / KD / hint | Các YAML Teacher/KD hiện có | Hướng phụ, hạ ưu tiên; không tiếp tục thêm biến thể nếu chưa có lý do mới |

Đường dẫn config trong bảng tính từ `config/`. Không suy ra một run đã hoàn tất chỉ vì có YAML hoặc checkpoint được nhắc trong comment.

## 2. Nhiều giả thuyết có chứa nghiệm tốt hơn không?

**Có, theo kết quả oracle người dùng cung cấp trên 3DPW test.** Đây là lỗi pose input của lifter: **H36M-17, root-relative, mm**.

| Cách lấy pose | MPJPE |
|---|---:|
| S=1 | 72.1 |
| Oracle cả pose, K=5 | 64.8 |
| Oracle cả pose, K=10 | 60.9 |
| Oracle cả pose, K=20 | 57.6 |
| Oracle từng khớp, K=20 | 39.9 |

Oracle chọn bằng GT, nên không phải kết quả có thể dùng khi suy luận thực tế. Oracle từng khớp còn có thể ghép các khớp từ những giả thuyết khác nhau.

Các pose oracle cũng cải thiện được mesh. Bảng này đo **FuseKin qua `dataset.evaluate`: MPJPE / MPVPE, mm**.

| Đầu vào fusion | Student gốc | Student train cùng selector |
|---|---:|---:|
| S=1, bỏ qua selector | 78.2 / 94.4 | 76.1 / 92.8 |
| Oracle cả pose | 75.1 / 90.5 | 70.8 / 86.4 |
| Oracle từng khớp | 71.3 / 86.2 | 65.4 / 80.4 |

Kết luận trong phạm vi phép đo: nhóm giả thuyết chứa nghiệm hữu ích và Student có thể hưởng lợi nếu lấy được pose phù hợp. Chưa đủ để kết luận tăng K hoặc dùng mean tự động cải thiện mesh.

## 3. Selector bằng ảnh đã giúp gì?

Theo kết quả người dùng báo:

- Pose input test: MPJPE **72.4 → 67.3 mm** sau selector.
- Mesh: MPJPE **78.6 mm**, kém hơn **76.1 mm** khi cùng Student đó bỏ selector và nhận S=1.
- Pose input train được báo khoảng **50–60 mm**, test khoảng **67 mm**. Có khoảng cách train/test; chẩn đoán đang xét việc fusion phụ thuộc quá mức vào pose selector.

Mốc 72.4 và mốc 72.1 ở mục 2 thuộc các phép đo đã báo riêng; chưa có đủ thông tin seed/checkpoint để gộp chúng thành cùng một run.

**Kết luận:** ảnh cung cấp tín hiệu giúp gộp giả thuyết, nhưng cách train hiện tại chưa biến lợi ích pose thành lợi ích mesh. Khoảng cách train/test và tác động của phân phối pose input cần được tách tiếp; không chỉ dựa vào oracle để coi selector đã thành công.

Các hướng chặn gradient từ mesh vào selector, giảm sức chứa hoặc thêm nhiễu vào pose đã chọn mới là đề xuất, **chưa triển khai trong source hiện tại**. Ưu tiên trước mắt của người dùng đã chuyển sang denoiser GCN + attention.

## 4. Bản augmentation đang train để kiểm tra điều gì?

`train_student_v2_diff_aug.yml` giữ Student baseline, không selector/KD/Teacher:

- Train: mỗi mẫu có xác suất 0.5 dùng một giả thuyết S=5, K=1 ngẫu nhiên thay pose S=1.
- Eval: S=1, K=1 như baseline; lifter vẫn đóng băng.
- Câu hỏi: Student có tốt lên chỉ nhờ thấy pose đa dạng trong train hay cần cơ chế chọn bằng ảnh?

Khi có log, so với baseline cùng dữ liệu/lịch train/checkpoint selection. Ghi MPJPE/MPVPE mesh và seed/epoch; không quy mọi thay đổi cho selector vì bản aug không có selector.

## 5. Thí nghiệm denoiser mới

Baseline: `finetune_difflifter_r25_cam.yaml`.

Thay đổi duy nhất về thiết kế thí nghiệm: bật `DIFFLIFT.gcn_attn=True`, thay block attention thuần bằng GCN skeleton + attention song song, concatenate và chiếu về D trước FFN/residual. Giữ D=256, depth=8, diffusion pred_x0/DDIM, 4 dataset, tỷ lệ 0.25 mỗi tập, 15 epoch, batch 512, lr 3e-4, EMA 0.999 và augmentation như baseline.

Thư mục mới: `experiment/finetune_difflifter_r25_cam_gcn_attn/`; train từ đầu, đánh giá **`best_epoch.bin`** theo `3dpw-val`. Nhiều tham số hơn baseline, nên đây chưa phải ablation cân bằng số tham số.

Khi người dùng chạy xong:

1. So MPJPE/PA-MPJPE 17 khớp của baseline và bản mới với cùng S/K trên 3DPW và H36M.
2. Dùng `inspect_difflifter.py` xem lỗi chiều sâu, xoay toàn thân, độ đa dạng và oracle; không chỉ nhìn lỗi trung bình.
3. Khi có bằng chứng lifter cải thiện mới lập thí nghiệm Student riêng. Các YAML Student đang có vẫn trỏ checkpoint `r25_cam`, không tự chuyển sang GCN.

## 6. SimCC — kết quả giữ lại để nghiên cứu sau

Probe RTMPose Halpe-26 trên **32 người validation**: raw SimCC có số âm; decode từ bin→crop→ảnh khớp decoder với sai số cực đại khoảng **0.000122 px**. Điều này xác nhận phép biến đổi tọa độ, không xác nhận xác suất bất định đã đúng.

| Softmax temperature | Sigma median x/y, pixel ảnh gốc | Entropy chuẩn hóa median x/y |
|---|---|---|
| 1.0 | 238.84 / 329.47 | 0.9956 / 0.9961 |
| 0.1 | 22.98 / 38.97 | 0.4834 / 0.4951 |
| 0.05 | 5.05 / 6.14 | 0.4116 / 0.4243 |

Đây là so sánh normalization trên cùng raw, chưa hiệu chuẩn bằng GT. Không chọn T=0.05 chỉ vì sigma nhỏ hơn; chưa có conditional prior SimCC/GMM tích hợp trong DiffLifter.

## 7. Mốc lịch sử còn hữu ích

Các số dưới đây là **tham khảo lịch sử**, không thay thế mốc cùng giao thức của thí nghiệm mới.

| Mesh v2 lịch sử | MPJPE / MPVPE, mm |
|---|---:|
| Teacher GT 3D | 33.90 / 46.94 |
| Student MotionBERT scratch | 79.65 / 95.22 |
| Student chỉ khởi tạo từ Teacher | 79.14 / 95.64 |
| Student feature KD | 78.49 / 94.23 |
| Student KD + contrastive, theo ghi chép cũ | 78.66 / 94.65 |
| Student S-hint | 81.41 / 97.95 |

Các run lịch sử được ghi là chọn theo test; chênh lệch nhỏ chưa có thống kê nhiều seed. Content ablation trên tập con n=1600 từng cho ảnh mean thay ảnh thật làm MPJPE tăng khoảng 20.18 mm ở Teacher và 18.54 mm ở Student: cả hai có dùng ảnh. Điều này không tự chứng minh KD sẽ hiệu quả.

| Lifter lịch sử | Train có 3DPW | 3DPW / H36M MPJPE 17 khớp, mm |
|---|---|---:|
| MotionBERT Halpe | Có | 80.7 / 88.5 |
| PMCE finetune, epoch 4 | Có | 75.3 / 48.9 |
| DiffLifter `v3_mix` cũ | Có | 76.1 / 60.9 |
| DiffLifter `v4_cam` cũ | Có | 75.5 / 53.2 |

`v3_mix`/`v4_cam` dùng code kiến trúc lịch sử, không phải cấu hình đang được source hiện tại hỗ trợ nguyên trạng. Kết quả phân tích cũ từng cho thấy lỗi xoay toàn thân/chiều sâu nổi bật và có mẫu sai nhưng spread thấp; đó là động cơ nghiên cứu, không phải số đo của denoiser GCN mới.

## 8. Cách cập nhật tiếp

Mỗi kết quả mới cần ghi: tên run/YAML, checkpoint và epoch, seed, dữ liệu train, tập chọn best, tập đánh giá, S/K và thước đo 17 hay 14 khớp. Nếu thiếu thì ghi thiếu, không suy từ comment hoặc tên thư mục.

Các YAML Student diffusion hiện dùng 3DPW-train và chọn best trên 3DPW test. Lifter `r25_cam` không train 3DPW nhưng chọn best trên 3DPW-val. Ghi rõ giới hạn này khi so với giao thức không dùng 3DPW; xem mục 6 của tài liệu bối cảnh.
