# FuseKin — kết quả và tiến độ thí nghiệm

Cập nhật **2026-10-23**. Số bên dưới lấy từ kết quả người dùng đã cung cấp trong trao đổi và tài liệu lịch sử của dự án; không phải các phép đo vừa được chạy lại. Chưa có đủ đường dẫn checkpoint/seed/epoch cho mọi số, nên không tự gán thêm.

Kiến trúc và giao thức hiện tại: [01_PROJECT_CONTEXT.md](01_PROJECT_CONTEXT.md). Lệnh chạy: [README.md](README.md).

## 1. Trạng thái hiện tại

| Nhánh | Config | Trạng thái / câu hỏi |
|---|---|---|
| Lifter baseline | `finetune_difflifter_r25_cam.yaml` | Đã có checkpoint, dùng làm lifter đóng băng cho các YAML Student diffusion |
| Student diffusion baseline | `train_student_v2_diff.yml` | Mốc S=1, K=1; người dùng báo mesh khoảng 78.2 / 94.4 mm |
| Selector theo khớp bằng ảnh | `train_student_v2_diff_sel.yml` | Có kết quả: pose tốt hơn nhưng mesh chưa tốt hơn; xem mục 3 |
| Augmentation bằng pose diffusion | `train_student_v2_diff_aug.yml` | Người dùng báo không hiệu quả; chưa có số liệu/checkpoint chi tiết |
| Spread vào fusion | `train_student_v2_diff_spread.yml` | Người dùng báo best trên 3DPW: MPJPE 77.59 / MPVPE 94.17 mm; công cụ chẩn đoán đã code, chưa chạy; xem mục 9 |
| GCN + attention song song | `finetune_difflifter_r25_cam_gcn_attn.yaml` | Đã đối chiếu hai best checkpoint epoch 6, S=1/K=1: chỉ H36M giảm lỗi, 3DPW/OC/PC tăng; chưa thay baseline, xem mục 5 |
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

## 4. Bản augmentation đã thử để kiểm tra điều gì?

`train_student_v2_diff_aug.yml` giữ Student baseline, không selector/KD/Teacher:

- Train: mỗi mẫu có xác suất 0.5 dùng một giả thuyết S=5, K=1 ngẫu nhiên thay pose S=1.
- Eval: S=1, K=1 như baseline; lifter vẫn đóng băng.
- Câu hỏi: Student có tốt lên chỉ nhờ thấy pose đa dạng trong train hay cần cơ chế chọn bằng ảnh?

Cập nhật 2026-10-23: người dùng báo augmentation không hiệu quả; chưa có số MPJPE/MPVPE, epoch hoặc checkpoint chi tiết. Khi bổ sung log, đối chiếu baseline cùng dữ liệu/lịch train/checkpoint selection; bản aug không có selector. Trạng thái mới nhất ở mục 9.

## 5. Thí nghiệm denoiser mới

Baseline: `finetune_difflifter_r25_cam.yaml`.

Thay đổi duy nhất về thiết kế thí nghiệm: bật `DIFFLIFT.gcn_attn=True`, thay block attention thuần bằng GCN skeleton + attention song song, concatenate và chiếu về D trước FFN/residual. Giữ D=256, depth=8, diffusion pred_x0/DDIM, 4 dataset, tỷ lệ 0.25 mỗi tập, 15 epoch, batch 512, lr 3e-4, EMA 0.999 và augmentation như baseline.

Thư mục mới: `experiment/finetune_difflifter_r25_cam_gcn_attn/`; train từ đầu, đánh giá **`best_epoch.bin`** theo `3dpw-val`. Nhiều tham số hơn baseline, nên đây chưa phải ablation cân bằng số tham số.

### Đối chứng người dùng gửi ngày 2026-10-23

Hai bảng tổng kết được gửi sau hướng dẫn chạy `main/eval_difflifter_3dpw.py`. Header xác nhận cả hai dùng **best checkpoint epoch 6, S=1, K=1**; số mẫu từng tập trùng nhau:

- Attention baseline: `experiment/finetune_difflifter_r25_cam/best_epoch.bin`.
- GCN + attention: `experiment/finetune_difflifter_r25_cam_gcn_attn/best_epoch.bin`.

| Tập | Số mẫu | Attention | GCN + attention | Chênh GCN − baseline |
|---|---:|---:|---:|---:|
| 3DPW test | 35,515 | 72.18 | 73.70 | +1.52 |
| Human36M test | 2,181 | 62.46 | 61.82 | -0.64 |
| 3DPW-OC | 20,243 | 82.14 | 82.30 | +0.16 |
| 3DPW-PC | 2,218 | 93.87 | 96.23 | +2.36 |

MPJPE lifter H36M-17 root-relative, mm; thấp hơn tốt hơn. Lệnh đã hướng dẫn dùng cùng batch=512/seed eval=0; các header không in hai tham số này hoặc seed train, nên chưa xác nhận độc lập và chưa có kết quả nhiều seed.

**Kết luận cho hai run đã đo:** GCN + attention chưa cải thiện so với attention baseline ở S=1/K=1; H36M giảm 0.64 mm nhưng 3DPW tăng 1.52 mm, PC tăng 2.36 mm; OC tăng nhẹ 0.16 mm. Giữ `r25_cam` làm mốc chính và lifter của Student. Chưa có bằng chứng để chuyển FuseKin sang checkpoint GCN hoặc quy kết GCN nói chung không phù hợp.

Log test ban đầu của bản GCN có PA-MPJPE 3DPW/H36M/OC/PC lần lượt **50.21 / 47.93 / 54.47 / 60.87 mm**; MPJPE H36M khi đó là 61.81 mm. Dùng 61.82 mm của evaluator đối chứng trong bảng trên; không tự suy nguyên nhân chênh 0.01 mm khi chưa đối chiếu chi tiết evaluator. Chưa có PA-MPJPE baseline trong cặp log mới.

Với K=1, min-K bằng MPJPE là điều hiển nhiên. Chất lượng đa giả thuyết, spread và oracle của hai backbone chưa được so sánh. Mốc 72.1 mm cũ ở mục 2 không thay thế mốc baseline trực tiếp 72.18 mm mới xác nhận.

Bước tiếp theo:

1. Nếu tiếp tục xét GCN cho mục tiêu đa giả thuyết, so cùng S=5/K=20 và oracle trên hai checkpoint; dùng `inspect_difflifter.py` để phân tích sâu hơn.
2. Hướng conditioning từng block (adaLN) đã được trao đổi, chưa code/chưa được duyệt triển khai. Nếu kiểm tra tác dụng độc lập của conditioning, dùng attention baseline làm đối chứng trước; không mặc định GCN là backbone tốt hơn.
3. Chỉ lập thí nghiệm Student riêng khi có bằng chứng cải thiện phù hợp mục tiêu. Các YAML Student vẫn trỏ checkpoint `r25_cam`, không tự chuyển sang GCN.

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

## 9. Spread - kết quả người dùng báo và công cụ chẩn đoán (2026-10-23)

Người dùng gửi dòng log: `chon best theo: ['3dpw'] | MPJPE 77.59 | MPVPE 94.17`.
Đây là metric mesh qua evaluator (MPJPE 14 khớp), đơn vị mm. Chưa có epoch, seed train và đường dẫn checkpoint thực tế của log; không tự gán các thông tin đó. YAML tương ứng vẫn train `3dpw-train`, chọn best trên **3DPW test**: đây là kết quả phát triển, không phải đánh giá cuối bằng test độc lập.

Mốc baseline lịch sử trong YAML là 78.23 / 94.37 mm, nên spread giảm 0.64 / 0.20 mm trong phép đối chiếu số đã báo. Chưa chạy lại hai best checkpoint cùng evaluator/batch size để xác nhận đối chứng; chênh lệch nhỏ của một run không chứng minh lợi ích ổn định. Augmentation: người dùng báo không hiệu quả, chưa cung cấp số chi tiết.

`main/inspect_student_spread.py` đã code, **chưa chạy/chưa có kết quả kiểm chứng**. Chỉ suy luận trên một checkpoint Student spread, không train lại hoặc đổi kiến trúc:

- `real`: gọi `ARTS.forward_student` như lúc eval; hook lấy chính feature ảnh, pose S=1 và std/embedding thực đã dùng. Hook được tháo sau mỗi lần gọi.
- `off`: dùng lại ảnh/pose/PE2D, bỏ toàn bộ embedding spread, bao gồm bias của Linear. Không tương đương chỉ đưa std=0.
- `shuffle`: đổi cả tensor std `(17,3)` giữa các mẫu trong cùng batch; giữ thứ tự khớp. Không có mẫu tự nhận spread của chính mình. Batch cuối chỉ một mẫu không tính shuffle; các delta chỉ tính trên cặp có đủ metric.
- Tất cả metric mesh gọi `dataset.evaluate`; tương quan spread với lỗi lifter dùng H36M-17, bỏ pelvis (16 khớp còn lại). Báo cả Spearman gộp và theo từng khớp.
- Xuất `samples.csv`, `joints.csv`, `joint_summary.csv`, `quartiles.csv`, `summary.json`, `inspect_log.txt`, `overview.png` và `vis/*.png`. Hình gồm crop + 2D tô màu theo spread, S=1/GT/K pose, mesh real/off/GT nhìn trước/ngang; đây là hình trực giao đã căn root, không phải overlay camera lên ảnh.
- Mẫu improved/worsened chọn theo MPJPE **off - real trên cùng checkpoint**; thêm mẫu ngẫu nhiên, giới hạn bộ nhớ mesh chỉ giữ các mẫu cần vẽ. Không được hiểu đây là ảnh so với checkpoint baseline khác.
- Kiểm tra ở batch đầu: replay embedding thật phải khớp output mesh của wrapper (sai khác tối đa 1e-6 mét), sai thì dừng. Checkpoint nạp strict; thiếu spread/backbone hay sai kiến trúc thì báo lỗi.

Giữ một GPU và batch size như phép đo cần đối chiếu (YAML hiện tại 32), vì wrapper đặt lại seed DDIM=0 mỗi batch; thay batch/GPU có thể đổi nhiễu của mỗi mẫu. `--seed` chỉ đổi lựa chọn donor/mẫu vẽ, không thay seed nội bộ lifter. `--max_batches` chỉ là kết quả tập con. `--num_vis 0` bỏ hình và không cần matplotlib.

Cách đọc: real tốt hơn shuffle hỗ trợ việc model dùng spread phù hợp từng mẫu; real tốt hơn off cho thấy phụ thuộc nhánh spread. Bỏ nhánh là can thiệp khác phân phối train, không thay thế đối chứng retrain baseline. Shuffle trong batch có thể yếu do các frame gần nhau có spread tương tự; chênh lệch nhỏ không đủ kết luận model bỏ qua spread. Chưa dùng kết quả chẩn đoán để sửa gate/kiến trúc.
