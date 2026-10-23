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
| Spread vào fusion | `train_student_v2_diff_spread.yml` | Đã có log chẩn đoán toàn bộ 35,515 mẫu do người dùng gửi: real/off/shuffle gần như ngang nhau; chưa thấy lợi ích rõ của spread lúc inference, xem mục 9 |
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

`main/inspect_student_spread.py` đã code và **người dùng đã gửi kết quả chạy đủ 35,515 mẫu 3DPW**; bảng kết quả ở cuối mục này. Đây không phải phép chạy lại do trợ lý thực hiện. Chỉ suy luận trên một checkpoint Student spread, không train lại hoặc đổi kiến trúc:

- `real`: gọi `ARTS.forward_student` như lúc eval; hook lấy chính feature ảnh, pose S=1 và std/embedding thực đã dùng. Hook được tháo sau mỗi lần gọi.
- `off`: dùng lại ảnh/pose/PE2D, bỏ toàn bộ embedding spread, bao gồm bias của Linear. Không tương đương chỉ đưa std=0.
- `shuffle`: đổi cả tensor std `(17,3)` giữa các mẫu trong cùng batch; giữ thứ tự khớp. Không có mẫu tự nhận spread của chính mình. Batch cuối chỉ một mẫu không tính shuffle; các delta chỉ tính trên cặp có đủ metric.
- Tất cả metric mesh gọi `dataset.evaluate`; tương quan spread với lỗi lifter dùng H36M-17, bỏ pelvis (16 khớp còn lại). Báo cả Spearman gộp và theo từng khớp.
- Xuất `samples.csv`, `joints.csv`, `joint_summary.csv`, `quartiles.csv`, `summary.json`, `inspect_log.txt`, `overview.png` và `vis/*.png`. Hình gồm crop + 2D tô màu theo spread, S=1/GT/K pose, mesh real/off/GT nhìn trước/ngang; đây là hình trực giao đã căn root, không phải overlay camera lên ảnh.
- Mẫu improved/worsened chọn theo MPJPE **off - real trên cùng checkpoint**; thêm mẫu ngẫu nhiên, giới hạn bộ nhớ mesh chỉ giữ các mẫu cần vẽ. Không được hiểu đây là ảnh so với checkpoint baseline khác.
- Kiểm tra ở batch đầu: replay embedding thật phải khớp output mesh của wrapper (sai khác tối đa 1e-6 mét), sai thì dừng. Checkpoint nạp strict; thiếu spread/backbone hay sai kiến trúc thì báo lỗi.

Giữ một GPU và batch size như phép đo cần đối chiếu (YAML hiện tại 32), vì wrapper đặt lại seed DDIM=0 mỗi batch; thay batch/GPU có thể đổi nhiễu của mỗi mẫu. `--seed` chỉ đổi lựa chọn donor/mẫu vẽ, không thay seed nội bộ lifter. `--max_batches` chỉ là kết quả tập con. `--num_vis 0` bỏ hình và không cần matplotlib.

Cách đọc: real tốt hơn shuffle hỗ trợ việc model dùng spread phù hợp từng mẫu; real tốt hơn off cho thấy phụ thuộc nhánh spread. Bỏ nhánh là can thiệp khác phân phối train, không thay thế đối chứng retrain baseline. Shuffle trong batch có thể yếu do các frame gần nhau có spread tương tự; chênh lệch nhỏ không đủ kết luận model bỏ qua spread. Chưa dùng kết quả chẩn đoán để sửa gate/kiến trúc.

### Log chẩn đoán người dùng gửi sau khi chạy

Đầu ra: `experiment/student_v2_diff_spread/checkpoint/inspect_spread_3dpw_20261023_165208_598609/`. Log ghi đủ 35,515 mẫu cho cả real/off/shuffle và đã lưu 12 hình. Đoạn trích không có header checkpoint/epoch/seed/batch size hoặc dòng kiểm tra replay; chưa xác nhận độc lập các giá trị này. Ban đầu chỉ phân tích log; sau đó đã xem trực tiếp ba hình mẫu và biểu đồ tổng hợp người dùng gửi (ghi nhận bên dưới), chưa có CSV đầy đủ.

| Điều kiện trên cùng checkpoint | MPJPE (14) | PA-MPJPE | MPVPE |
|---|---:|---:|---:|
| real | 77.594 | 53.186 | 94.169 |
| off | 77.570 | 53.077 | 94.258 |
| shuffle trong batch | 77.592 | 53.173 | 94.162 |

Đơn vị mm. Delta dùng dòng thống kê ghép cặp chưa làm tròn: real - off = **+0.023 / +0.109 / -0.089**; real - shuffle = **+0.002 / +0.013 / +0.008** (MPJPE/PA/MPVPE). Chênh giữa các số đã làm tròn trong bảng có thể lệch 0.001 mm với mean delta in riêng.

Real tốt hơn off trên 49.2% mẫu về MPJPE và 53.4% về MPVPE khi yêu cầu cải thiện >0.1 mm; xấu đi lần lượt 42.8% và 39.9%; phần còn lại 8.0% và 6.6%. Median delta real-off MPJPE=-0.080, MPVPE=-0.188 mm. Tỷ lệ tốt hơn không nói lên độ lớn lợi ích; các mẫu xấu đi có thể triệt tiêu mức tăng của đa số mẫu. Ngưỡng 0.1 mm là ngưỡng mô tả của script, không phải kiểm định ý nghĩa thống kê.

| Nhóm | n | Spread trung bình (mm) | MPJPE real | MPJPE off | Gain off-real (mm) |
|---|---:|---:|---:|---:|---:|
| Q1 | 8,879 | 26.28 | 65.596 | 65.603 | +0.007 |
| Q2 | 8,878 | 33.85 | 70.377 | 70.462 | +0.085 |
| Q3 | 8,879 | 40.62 | 79.552 | 79.613 | +0.061 |
| Q4 | 8,879 | 53.95 | 94.849 | 94.602 | -0.247 |

Spread cao đi kèm lỗi mesh cao hơn giữa các nhóm trong log, song nhánh cộng embedding không giảm được lỗi nhóm Q4; MPJPE còn tăng 0.247 mm khi bật spread. Không đồng nhất Q4 với nhóm che khuất vì chưa kiểm tra nhãn/ảnh.

Spearman(spread, lỗi pose S=1) gộp 16 khớp không pelvis = **0.5591**. Theo từng khớp chỉ **0.091–0.311**: R/L ankle=0.091/0.103; L/R wrist=0.311/0.258; L/R elbow=0.250/0.262. Con số gộp có thể chịu ảnh hưởng của khác biệt loại khớp; chưa có bằng chứng để xem spread là confidence lỗi từng khớp đã hiệu chuẩn.

**Kết luận trong phạm vi run này:** chưa thấy lợi ích rõ của thông tin spread đúng mẫu với cách cộng token hiện tại. Off giữ chất lượng gần real; shuffle trong batch cũng gần real. Không kết luận model hoàn toàn bỏ qua spread vì có khác biệt từng mẫu, và donor trong batch có thể là các frame gần nhau. Mức 77.59/94.17 thấp hơn baseline lịch sử 78.23/94.37 không thể quy ngay cho việc tận dụng bất định lúc inference: tắt nhánh vẫn gần như giữ nguyên kết quả; tác động trong train và biến thiên giữa các run chưa được tách.

**Bước tiếp:** xem `overview.png`, ảnh improved/worsened/random và phân phối delta trong `samples.csv` để biết các ca xấu/tốt có ý nghĩa thực tế hay chỉ thay đổi rất nhỏ. Nếu cần kết luận sâu về phụ thuộc spread theo mẫu, đề xuất phép hoán đổi khác sequence/người vì shuffle cục bộ có thể quá nhẹ; chưa code/chưa duyệt thay đổi đó. Chưa thêm gate hoặc mở run train mới chỉ từ bảng này. Mọi số trên vẫn thuộc giao thức chọn best theo 3DPW test đã ghi rõ ở trên.

### Đối chiếu bốn ảnh người dùng gửi

Đã xem trực tiếp `image(2).png`, `image(3).png`, `image(4).png`, `image(5).png` sau khi đọc ảnh đính kèm. Đây là ba mẫu riêng và một biểu đồ tổng hợp, không phải toàn bộ 12 hình. Số trong bảng là nhãn trên ảnh; gain dùng nhãn đã tính trước khi làm tròn metric.

| Ảnh / mẫu | Nhóm | MPJPE real / off, mm | Gain off-real trên nhãn, mm | Quan sát |
|---|---|---:|---:|---|
| image(2), #2749, flat_guitar_01 | random | 131.12 / 129.44 | -1.69 | missing=0; real/off khá giống nhau, còn lệch tư thế chân/tay so với GT |
| image(3), #8785, downtown_runForBus_00 | worsened | 94.31 / 87.63 | -6.67 | missing=1; real/off giữ tư thế tổng thể tương tự, real tăng MPVPE từ 108.27 lên 121.33 mm |
| image(4), #17500, downtown_car_00 | improved | 469.33 / 481.27 | +11.93 | missing=16; crop chủ yếu là nội thất xe và một phần tay. Cả real/off vẫn lệch rõ hướng toàn thân và bố trí chi so với GT; giảm số không đồng nghĩa cứu được ca này |

Biểu đồ image(5): phân phối MPJPE off-real tập trung gần 0, có đuôi hai phía; Q4 có gain âm như bảng log. Scatter spread/lỗi phân tán rộng: ở spread tương tự vẫn có lỗi S=1 rất khác nhau. Màu spread của từng hình mẫu được chuẩn hóa riêng, không so độ sáng giữa các ảnh như một thang mm chung.

Các hình củng cố kết luận nhánh cộng std hiện chỉ gây điều chỉnh nhỏ ở nhiều mẫu, chưa cho thấy khả năng sửa các ca sai lớn. `missing=0` chỉ nói mask hợp lệ, không chứng minh detector đúng hoặc 3D không còn mơ hồ. Các pose nhìn trước/ngang đều root-relative; không coi chúng là overlay camera lên ảnh.

Bước chẩn đoán đề xuất tiếp theo là đo oracle/min-K theo từng mẫu xấu (GT chỉ dùng phân tích) để tách: K có chứa pose hợp GT mà fusion chưa dùng, hay cả nhóm giả thuyết đã sai theo đầu vào 2D. Ba giá trị std mất thông tin hướng sửa và cấu trúc từng nghiệm; thêm gate không tự tạo được nghiệm đúng khi cả K đều sai. Chưa code hoặc chạy phép min-K theo các ID này, chưa đổi kiến trúc hay mở run mới.

## 10. Kiểm tra Teacher GT trên các mẫu khó - đã code, chờ chạy

Người dùng chuyển ưu tiên sang kiểm tra Teacher đang dùng cho KD trước khi đổi fusion đa giả thuyết. Chưa duyệt/chưa triển khai fusion K=5 hoặc gate. Chưa có kết quả Teacher mới.

`main/inspect_student_spread.py --mode teacher` dùng YAML/checkpoint Teacher GT thuần (không T-hint, không trộn lifter); hỗ trợ `--sample_ids 2749 8785 17500` để kiểm tra đúng ba mẫu đã xem. `--spread_csv` đọc CSV spread đã chạy để xác minh sequence/frame và ann_id của từng mẫu, đồng thời in metric Student real/off/shuffle đã lưu. Không suy vị trí mẫu từ ann_id. Chế độ chọn tập con gọi evaluator theo index dataset gốc, không dùng offset liên tiếp của Subset để tránh so sai annotation.

GT 3D lấy từ GT mesh qua H36M regressor, root-relative tại pelvis, mét, giống evaluator Teacher hiện có. Đối chiếu `result['teacher_input_joints']` với GT trước khi báo số; không lifter hoặc corruption GT khi eval. PE2D giữ detector Halpe/mask như run train vì GT input của Teacher là **3D**, không tự biến PE thành GT 2D.

Xuất MPJPE/PA-MPJPE/MPVPE mesh bằng `dataset.evaluate` (14 khớp), CSV từng mẫu và hình Teacher/GT trước-ngang. Chẩn đoán khớp/biểu đồ trong hình dùng H36M-17 bỏ pelvis và được ghi rõ. Có thể bỏ `--sample_ids` để chạy toàn tập, lưu hình nhóm Teacher worst/best/random; chỉ số tổng kết khi chọn ba mẫu được ghi rõ là tập con. Lệnh cụ thể ở README.

YAML `config/train_student_kd_e1_v2_diff.yml` đang trỏ Teacher tới `experiment/teacher_gt_v2/checkpoint/best.pth.tar`; source Teacher GT là `config/train_teacher_k16_v2.yml` (teacher_hint=False, teacher_lift_alpha_max=0, train corruption=0.3 chỉ áp dụng khi train; img_probe=True). Đường dẫn là giá trị đọc từ YAML local, chưa xác nhận file trên server hoặc YAML thật của checkpoint người dùng. Nạp strict sẽ báo lỗi nếu checkpoint/YAML không khớp. Alias lịch sử teacher_model→smpl_model được đổi tên trước khi strict load, không lọc sai shape hoặc bỏ backbone.

Câu hỏi: với GT 3D đúng, Teacher có tái tạo được ca guitar/chạy/trong xe không? Teacher tốt cho thấy pose sạch có thể hỗ trợ đường Teacher, chưa đảm bảo Student sẽ học được qua KD. Teacher vẫn sai cần xét đường GT17→mesh, fusion/head, ảnh và PE2D; chưa thể quy riêng lỗi cho ảnh. Đây là chẩn đoán oracle dùng GT, vẫn thuộc giao thức chọn best theo 3DPW test của YAML hiện có.
