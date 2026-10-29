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

## 10. Kiểm tra Teacher GT trên các mẫu khó - đã có ảnh mẫu #17500

Người dùng chuyển ưu tiên sang kiểm tra Teacher đang dùng cho KD trước khi đổi fusion đa giả thuyết. Chưa duyệt/chưa triển khai fusion K=5 hoặc gate. Người dùng đã gửi ảnh Teacher mẫu #17500; chưa có log tổng kết ba mẫu hoặc toàn tập.

`main/inspect_student_spread.py --mode teacher` dùng YAML/checkpoint Teacher GT thuần (không T-hint, không trộn lifter); hỗ trợ `--sample_ids 2749 8785 17500` để kiểm tra đúng ba mẫu đã xem. `--spread_csv` đọc CSV spread đã chạy để xác minh sequence/frame và ann_id của từng mẫu, đồng thời in metric Student real/off/shuffle đã lưu. Không suy vị trí mẫu từ ann_id. Chế độ chọn tập con gọi evaluator theo index dataset gốc, không dùng offset liên tiếp của Subset để tránh so sai annotation.

GT 3D lấy từ GT mesh qua H36M regressor, root-relative tại pelvis, mét, giống evaluator Teacher hiện có. Đối chiếu `result['teacher_input_joints']` với GT trước khi báo số; không lifter hoặc corruption GT khi eval. PE2D giữ detector Halpe/mask như run train vì GT input của Teacher là **3D**, không tự biến PE thành GT 2D.

Xuất MPJPE/PA-MPJPE/MPVPE mesh bằng `dataset.evaluate` (14 khớp), CSV từng mẫu và hình Teacher/GT trước-ngang. Chẩn đoán khớp/biểu đồ trong hình dùng H36M-17 bỏ pelvis và được ghi rõ. Có thể bỏ `--sample_ids` để chạy toàn tập, lưu hình nhóm Teacher worst/best/random; chỉ số tổng kết khi chọn ba mẫu được ghi rõ là tập con. Lệnh cụ thể ở README.

YAML `config/train_student_kd_e1_v2_diff.yml` đang trỏ Teacher tới `experiment/teacher_gt_v2/checkpoint/best.pth.tar`; source Teacher GT là `config/train_teacher_k16_v2.yml` (teacher_hint=False, teacher_lift_alpha_max=0, train corruption=0.3 chỉ áp dụng khi train; img_probe=True). Đường dẫn là giá trị đọc từ YAML local, chưa xác nhận file trên server hoặc YAML thật của checkpoint người dùng. Nạp strict sẽ báo lỗi nếu checkpoint/YAML không khớp. Alias lịch sử teacher_model→smpl_model được đổi tên trước khi strict load, không lọc sai shape hoặc bỏ backbone.

Câu hỏi: với GT 3D đúng, Teacher có tái tạo được ca guitar/chạy/trong xe không? Teacher tốt cho thấy pose sạch có thể hỗ trợ đường Teacher, chưa đảm bảo Student sẽ học được qua KD. Teacher vẫn sai cần xét đường GT17→mesh, fusion/head, ảnh và PE2D; chưa thể quy riêng lỗi cho ảnh. Đây là chẩn đoán oracle dùng GT, vẫn thuộc giao thức chọn best theo 3DPW test của YAML hiện có.

### Ảnh Teacher mẫu #17500 và hướng KD người dùng đề nghị

Đã xem trực tiếp ảnh `image(6).png` người dùng gửi: Teacher GT, sample #17500, downtown_car_00/image_00794.jpg, selected subset, GT input error=0 mm. Nhãn cho biết Teacher MPJPE(14)=**153.026**, PA-MPJPE=**111.654**, MPVPE=**170.524** mm. Student spread cùng mẫu (CSV cũ trên hình): MPJPE=469.334, MPVPE=450.284 mm. Teacher tái tạo được dạng ngồi/hướng toàn thân tốt hơn Student, nhưng tay/chân vẫn lệch GT rõ; chưa coi Teacher là nghiệm chính xác. Chỉ có một hình mẫu, không suy ra mức giảm chung cho ba mẫu hoặc toàn tập, không suy ra hiệu quả KD.

Người dùng muốn giữ spread và chuyển sang train KD, chỉnh tín hiệu để Teacher dạy được ca khó. Source E1 (`train_student_kd_e1_v2_diff.yml`) hiện KD MSE qua projector ở feature joint/global sau fusion, w_proj=0, w_joint=w_global=1, kd_noise_weight=False; Student có sẵn hard loss GT joint/SMPL/mesh. Có cờ kd_noise_weight nhưng cơ chế hiện có chỉ cân theo lỗi pose lifter so với GT, chưa xét chất lượng output Teacher so với Student. Các YAML KD này chưa bật spread; YAML spread hiện chưa bật KD, nên chưa có kết quả xác nhận run kết hợp spread+KD.

Đề xuất đang thảo luận, **chưa code/chưa duyệt công thức cuối**: giữ nguyên các điểm feature KD, chỉ thay luật trọng số per-joint bằng mức khó Student kết hợp lợi thế Teacher. Lỗi Student/Teacher phải tính từ cùng H36M regressor của mesh, cùng GT/root/đơn vị; không so lỗi input lifter của bên này với lỗi mesh của bên kia. Teacher kém hơn ở khớp nào thì giảm/tắt KD khớp đó; khớp Student sai nhiều và Teacher tốt hơn thì tăng KD có giới hạn. Trọng số stop-gradient, chỉ dùng GT của train; mask hợp lệ GT độc lập với mask 2D detector. KD global dùng trọng số mẫu tổng hợp từ khớp. Không thêm KD output hoặc ép attention giống nhau trong cùng lần thay đổi đầu.

Đối chứng cần thiết: spread + KD feature đồng đều, so với cùng cấu hình spread + KD có trọng số; giữ cùng seed/khởi tạo/loss/lịch train và tập chọn best, đặc biệt không bật init_student_from_teacher ở một run mà tắt ở run kia. Giữ spread trong cả hai run là lựa chọn thí nghiệm, chưa khẳng định spread có lợi; sau KD vẫn so real/off/shuffle để xem nhánh spread có được dùng hơn không. Tín hiệu Teacher dùng GT không luôn suy ra được từ ảnh bị che và 2D hỏng; Teacher tốt hơn không bảo đảm Student sao chép được toàn bộ feature đặc quyền.

## 11. Spread + KD feature E1 - đã có log người dùng gửi (2026-10-23)

Người dùng duyệt chuẩn bị bản spread + KD hiện có làm đối chứng trước. YAML mới: `config/train_student_kd_e1_v2_diff_spread.yml`, run dự kiến `student_kd_e1_v2_diff_spread`, seed lệnh chạy 123. So với `train_student_kd_e1_v2_diff.yml`, chỉ thêm spread=True, steps=5, k=20; mọi giá trị cấu hình E1 còn lại được giữ nguyên. Không sửa loss/trainer/kiến trúc hay YAML cũ, không chạy code dự án tại local.

Giữ pose chính S=1/K=1; Teacher GT checkpoint theo E1; Student `smpl_model` khởi tạo từ Teacher, spread Linear khởi tạo 0. KD feature MSE ở joint/global sau fusion, hệ số mỗi nhánh=1, proj=0, hard_w=1, lambda=1, không noise weighting/InfoNCE. Run mới không lấy checkpoint spread cũ làm điểm khởi đầu. Mục tiêu là đo bản kết hợp trước khi thay luật trọng số KD cho ca khó; chưa triển khai luật trọng số mới.

Đối chứng trực tiếp để tách tác động thêm spread là E1 diffusion có cùng seed, khởi tạo Teacher, dữ liệu, lịch train và evaluator. So với run spread-only trước đó còn khác cả KD và khởi tạo Teacher, không quy chênh lệch riêng cho loss KD. Bản KD có trọng số về sau cần giữ nguyên cấu hình/khởi tạo của YAML mới này và chỉ đổi luật trọng số.

Giữ train 3DPW-train, chọn best theo 3DPW test như run hiện có, nên chỉ coi là kết quả phát triển. Trạng thái lúc tạo YAML là chờ chạy; log người dùng gửi sau đó được ghi bên dưới. Lệnh chạy ở README. Thư mục kết quả mới do trainer tạo trên server khi chạy, không tạo thư mục rỗng ở local.

### Kết quả KD người dùng báo và rà soát source

Người dùng báo best theo `['3dpw']`: **MPJPE 77.12 / MPVPE 93.73 mm**. Theo mạch trao đổi đây là bản spread + KD E1; chưa có YAML sao lưu của run, epoch, seed, PA-MPJPE, đường dẫn checkpoint hoặc log loss để xác nhận độc lập. So với scratch có spread 77.594 / 94.169, giảm khoảng **0.474 / 0.439 mm**; một run chưa đủ kết luận mức cải thiện ổn định hoặc tác dụng riêng của KD.

Rà soát tĩnh `Student_Trainer.train/_kd_dist/_ensure_projectors`, `ARTS.forward_student/forward_teacher`, fusion và optimizer: không thấy Student bị detach ngoài ý muốn trong đường E1 mặc định v1. Backbone/lifter đóng băng có chủ đích; `spread_proj` nằm ngoài no_grad, feature sau fusion đi vào projector KD và loss.backward; projector được thêm vào optimizer. Teacher eval/no_grad và target feature detach là đúng. Đây là kiểm tra source, chưa phải kiểm chứng gradient lúc chạy.

Các yếu tố cần tách khi so sánh:

- Scratch (`student_use_kd=False`) thực tế dùng `0.5*hard_loss + privileged_loss`; E1 feat dùng `1.0*hard_loss + 1.0*kd_loss + privileged_loss`. Vì vậy không chỉ thêm loss KD, mà còn đổi tỷ lệ hard/privileged. Hard loss đi qua AWL học được, không thể quy ngay mức chênh metric cho một hệ số.
- E1 khởi tạo `smpl_model` từ Teacher, scratch không dùng Teacher. Chưa có đối chứng cùng khởi tạo/hệ số loss để tách tác dụng của KD.
- YAML spread+KD local hiện có `lr_step=[5,10,15,22,27]`, khác bản lúc tạo `[15,22,27]` và khác YAML scratch spread hiện tại. Với scheduler MultiStepLR đang gọi cuối epoch, nếu run mới dùng đúng lịch local này, epoch 11-15 có LR 2.5e-5 thay vì 1e-4 của scratch, cuối lịch cũng thấp hơn bốn lần. Chưa xác nhận YAML thật trên server; không kết luận việc giảm LR sớm là nguyên nhân. Không sửa lịch/run hiện có.
- Feature KD cân đều các khớp GT hợp lệ (`fit_joint_trunc * is_valid_fit`), không theo lỗi mesh Student hoặc độ tốt tương đối của Teacher; KD global lấy mean token. Chưa có cơ chế ưu tiên ca khó mà Teacher làm tốt hơn. Mask KD là mask GT, không phải mask confidence Halpe.
- `_kd_dist` so sánh feature sau LayerNorm qua Linear học được; KD loss nhỏ chưa chứng minh mesh gần Teacher hoặc fusion dùng ảnh nhiều hơn. Khả năng projector hấp thụ khác biệt và gradient KD yếu/xung đột với hard loss mới là giả thuyết, cần log/đo gradient, chưa kết luận từ source.

Ưu tiên tiếp theo: xác minh YAML của run và đọc log đầu/giữa/cuối gồm J/G KD, hard_total, privileged, LR, epoch best và AWL; đối chiếu sai số từng mẫu/nhóm khó của checkpoint KD với scratch trước khi quy kết trung bình hoặc tăng lambda. Nếu bổ sung chẩn đoán, đo gradient KD/hard ở fusion và spread cùng chất lượng Teacher/Student trên cùng GT; chưa viết/chạy chẩn đoán này, chưa đổi luật trọng số KD. Giữ giới hạn giao thức chọn best trên 3DPW test.

### Bổ sung log epoch 14-22: sửa cách đọc mốc best

Người dùng gửi log liên tục epoch 14-22. Trong đoạn này, epoch **18** đồng thời tốt nhất cả MPJPE và MPVPE: **76.76 / 51.03 / 93.57 mm** (MPJPE14 / PA-MPJPE / MPVPE). Epoch 20 là **77.12 / 51.33 / 93.73**, không phải best của đoạn. Mốc epoch 18 giảm khoảng 0.834 / 2.156 / 0.599 mm so với scratch spread trong inspector (77.594 / 53.186 / 94.169); đây là đối chiếu số log, chưa chạy lại hai checkpoint cùng evaluator. Chưa có log trước epoch 14, sau epoch 22 hoặc header TEST CUOI xác nhận best.pth.tar toàn run.

Đã đối chiếu `_select_errors` và `main/train.py`: dòng `chon best theo` in metric của epoch hiện tại và tên tập dùng chọn checkpoint, không báo best tích lũy hoặc xác nhận vừa ghi best.pth.tar. `is_best` chỉ bật khi MPJPE hoặc MPVPE lập mức thấp mới so với toàn bộ lịch sử. Epoch 19-22 không vượt epoch 18 ở cả hai metric, nên không thay best bằng các epoch này theo code hiện tại. Best toàn run còn phụ thuộc epoch ngoài đoạn log.

Diễn biến KD trung bình: epoch 14 J=0.0272, G=0.0111 (tổng 0.0383); epoch 18 J=0.0248, G=0.0092 (0.0340); epoch 22 J=0.0244, G=0.0090 (0.0334). Có giảm rồi chậm lại trong đoạn này; chưa biết diễn biến giai đoạn đầu. P khoảng 2.08 không được cộng vào loss vì kd_w_proj=0, không phải KD đang chịu sai số 2.08. nw=1 và adaptive w=1 đúng với E1 không cân ca khó. Lift error khoảng 79.4 mm là lỗi đầu vào DiffLifter đóng băng, không phải lỗi mesh Student; nhãn MotionBERT trong log là tên cũ, việc lift error không giảm không chứng minh KD không học.

LR epoch 15=2.5e-5, epoch 16-22=1.25e-5, phù hợp lịch giảm sớm trong YAML local, nhưng chưa có YAML sao lưu run để xác nhận toàn bộ config. Sau khi giảm LR, epoch 18 vẫn cải thiện, nên log không chứng minh LR nhỏ là nguyên nhân KD kém. Từ epoch 18-22, train KD/hard tổng giảm nhẹ còn metric đánh giá dao động (MPJPE 76.76-78.40); không thấy dấu hiệu loss nổ/NaN trong đoạn.

AWL tăng trọng số mesh 2.756->3.892 và joint3D 2.826->4.108 (epoch14->22). Tổng loss có thành phần `log(1+p^2)` của AWL không phụ thuộc output model, vì vậy không được lấy KD~0.03 chia total~2.4 để kết luận gradient KD chỉ 1% hoặc chắc chắn quá yếu. Cần đo norm/hướng gradient trên cùng tham số và so feature trước/sau projector để kiểm tra giả thuyết này; chưa đo/chưa code thêm. Không tự tăng lambda, bật proj KD hoặc đổi lịch dựa riêng vào các độ lớn loss khác loại.

Bước tiếp: xác nhận epoch và kết quả TEST CUOI của best checkpoint; so sai số từng mẫu/nhóm khó KD với scratch và Teacher để biết lợi ích có tập trung đúng ca khó. Chỉ bổ sung chẩn đoán gradient hoặc đổi trọng số KD trong phạm vi được duyệt; chưa sửa source/YAML train của run này.

## 12. So scratch/KD trên cùng mẫu - đã có log toàn tập (2026-10-23)

Người dùng đồng ý tạm dừng train để so các checkpoint hiện có; epoch 18 đang là ứng viên best theo log đã gửi, chưa đọc trực tiếp checkpoint server. Người dùng yêu cầu rà `./main` trước khi thêm file. Đã đọc `compare_students.py`, đường chạy của `compare_teacher_student.py`, `diagnose_kd.py`, `kd_feature_analysis.py`, `compare_checkpoint_drift.py` và inspector hiện tại. `compare_students.py` có metric tổng nhiều Student, các công cụ feature cũ còn giả định MotionBERT/T-hint hoặc gọi forward thiếu cam_bbox/đường spread; không dùng nguyên trạng để kết luận thí nghiệm này.

Đã mở rộng **`main/inspect_student_spread.py --mode compare`**, không tạo file Python mới. Chạy hai Student spread với YAML riêng và Teacher GT trên cùng loader, strict load, ghi epoch và tùy chọn `--expected_epoch 18`. Dùng lại evaluator mesh 14 khớp của dataset, helper capture/visual hiện có. Mỗi YAML bắt đầu từ config default độc lập, cfg đúng được kích hoạt khi build/forward; không trộn key Teacher vào Student. Kiểm tra pose S1/std hai Student giống nhau trong 1e-6 mét và GT input Teacher khớp trong 0.001 mm. Không sửa source train, YAML cũ hoặc các YAML JOTR người dùng đang tạo.

Đầu ra: `compare_log.txt`, `summary.json`, `samples.csv`, `groups.csv`, ba YAML sao lưu, `overview.png`, `vis/` cạnh checkpoint KD trong thư mục mới. Gain = lỗi scratch - KD, dương là KD tốt hơn. Nhóm Q1–Q4 và 10% khó nhất cố định theo MPJPE scratch, thêm nhóm theo số keypoint thiếu và nhóm Teacher tốt hơn scratch >1 mm. Báo mean/median gain, phân vị 10/90, tỷ lệ tốt/xấu hơn ngưỡng mô tả 1 mm cho MPJPE/PA-MPJPE/MPVPE. Không suy ý nghĩa thống kê từ một run hoặc coi frame độc lập.

Visual: GT/scratch/KD/Teacher trước/ngang; ba ID #2749/#8785/#17500, mẫu cải thiện/xấu đi nhiều và mẫu ngẫu nhiên trong 10% khó. Mode compare dùng `--sample_ids` để chọn thêm hình, vẫn eval cả tập. Chọn hình sau khi có metric toàn lượt, nạp lại nguyên batch gốc để không đổi seed DDIM theo vị trí; xác minh ảnh/ann_id và metric replay trong 0.01 mm. Chỉ giữ mesh theo batch, không giữ mesh toàn tập. `--num_vis 0` bỏ matplotlib/hình; `--max_batches` phải được đọc là kết quả tập con.

Lệnh chính xác ở README, theo tên run đã chuẩn bị; người dùng cần dùng YAML thực tế của checkpoint nếu đã đổi lịch trên server. **Mới code và rà source/diff, chưa chạy/import/kiểm cú pháp Python hoặc đánh giá trên GPU.** Phép so này vẫn còn khác khởi tạo Teacher, hệ số hard loss và lịch LR giữa hai run; chỉ đo lợi ích tổng của bản KD. Teacher GT là oracle. Các checkpoint chọn theo 3DPW test chỉ phục vụ phân tích phát triển, không chọn/tinh chỉnh model tiếp bằng test. Chưa thay luật trọng số KD, chưa train biến thể mới.

### Kết quả ghép cặp người dùng gửi: 35,515 mẫu

Đã nhận bảng MPJPE14 theo nhóm từ mode compare. Sai khác đầu vào hai Student: pose_max_abs_m=0, std_max_abs_m=0; Teacher GT input error=0 mm. Đây là kết quả người dùng chạy, không phải phép chạy local của trợ lý. Chưa có header checkpoint/epoch, bảng PA-MPJPE/MPVPE toàn tập, CSV đầy đủ hoặc dòng xác nhận visual replay. Ảnh đính kèm `image(7).png` đã được xác định nhưng chưa xem trực tiếp pixel ở lượt phân tích này; nhận xét dưới đây chỉ dựa trên log.

| Nhóm theo lỗi scratch / detector | n | Scratch MPJPE | KD MPJPE | Teacher GT MPJPE | Gain scratch-KD | Median gain | Tốt/xấu >1 mm |
|---|---:|---:|---:|---:|---:|---:|---|
| Toàn tập | 35515 | 77.593 | 76.763 | 33.502 | +0.830 | +0.727 | 48.9% / 43.4% |
| Q1 | 8879 | 46.007 | 48.766 | 24.638 | -2.759 | -1.945 | 33.5% / 55.4% |
| Q2 | 8879 | 63.911 | 63.787 | 30.308 | +0.124 | +0.498 | 48.2% / 44.5% |
| Q3 | 8879 | 81.796 | 79.605 | 35.122 | +2.190 | +2.659 | 55.7% / 37.9% |
| Q4 | 8878 | 118.664 | 114.897 | 43.941 | +3.766 | +3.834 | 58.3% / 36.1% |
| 10% lỗi scratch cao nhất | 3552 | 142.483 | 137.729 | 49.144 | +4.754 | +4.714 | 61.6% / 32.5% |
| Không thiếu keypoint | 34835 | 77.168 | 76.342 | 33.375 | +0.826 | +0.709 | 48.9% / 43.5% |
| Thiếu 1–4 keypoint | 678 | 98.436 | 97.124 | 39.820 | +1.313 | +1.554 | 52.9% / 41.0% |
| Thiếu 9–17 keypoint | 2 | 415.397 | 500.669 | 102.552 | -85.272 | -85.272 | 0% / 100% |
| Teacher có lợi thế >1 mm | 35452 | 77.638 | 76.775 | 33.464 | +0.863 | +0.747 | 49.0% / 43.3% |
| Teacher không có lợi thế >1 mm | 63 | 52.362 | 69.789 | 54.830 | -17.427 | -15.883 | 0% / 100% |

Đơn vị mm. `hard10_teacher_better` trùng toàn bộ nhóm hard10 (n=3552), không phải một tập kiểm chứng độc lập. Không có dòng missing_5_8 trong log; không tự suy kết quả nhóm đó. Giá trị gain dùng số in riêng có thể lệch 0.001 mm so với trừ hai mean đã làm tròn.

Diễn giải: run KD cải thiện trung bình 0.830 mm và có gain lớn hơn ở các nhóm scratch sai nhiều, song Q1 xấu đi. Đây là phép so cả run, còn khác khởi tạo/hard loss/LR, chưa tách tác động riêng của loss KD hoặc spread. Nhóm chia theo chính lỗi scratch có thể tạo một phần xu hướng Q1 xấu/Q4 tốt, kể cả khi hai model chỉ khác phân bố sai số; chưa coi là bằng chứng KD đã học cơ chế nhận biết ca khó. Cần phân nhóm thêm theo lỗi pose S1 đầu vào chung (đã có s1_mpjpe16_mm trong samples.csv), số keypoint thiếu hoặc nhãn/sequence cố định, không chỉ theo output scratch.

Quy đổi đóng góp vào gain toàn tập từ các mean đã in: Q1 khoảng -0.690 mm; Q4 +0.941 mm; hard10 +0.475 mm (hard10 nằm trong Q4, không cộng hai nhóm này). Nhóm Teacher không có lợi thế >1 mm chỉ chiếm 63/35515=0.177%, đóng góp -0.031 mm. Vì vậy không thể quy phần lớn thiệt hại Q1 cho việc Teacher kém; Teacher trung bình ở Q1 vẫn tốt hơn nhiều (24.638 so với scratch 46.007). Nhóm thiếu 9–17 chỉ n=2, đóng góp -0.0048 mm toàn tập: cần xem từng ca thất bại, không khái quát cho mọi ca che khuất.

Teacher có lợi thế >1 mm trên 99.823% mẫu so với checkpoint scratch. Chỉ thêm gate theo điều kiện này sẽ tác động rất ít mẫu trong phép phân tích; không đủ bằng chứng rằng nó giải quyết được phần lớn hạn chế. Đây là so Teacher với scratch cố định, không thay thế phân bố Teacher so với Student đang thay đổi trong train. Trên hard10, khoảng cách lỗi scratch–Teacher là 93.339 mm, KD thu hẹp 4.754 mm (~5.1%); khoảng cách này là mô tả oracle, không phải phần cải thiện chắc chắn Student có thể đạt được từ ảnh/2D.

Ưu tiên tiếp: dùng samples.csv hiện có để kiểm tra nhóm theo lỗi S1 và xem gain ở mẫu scratch vốn tốt; sau đó mới chốt trọng số KD. Nếu tiếp tục sửa train, cần xét độ khó và khả năng bảo toàn mẫu đã tốt, không chỉ tăng lambda hoặc gate Teacher kém. Chưa sửa công thức KD, chưa tạo run mới. Các kết quả vẫn chọn checkpoint trên 3DPW test, chỉ là phân tích phát triển của một run; không suy ý nghĩa thống kê.

## 13. Content ablation Teacher GT - kết quả người dùng gửi (2026-10-24)

Đã đọc `main/content_ablation_test.py`, đối chiếu nhánh Teacher alpha=0 với `ARTS.forward_teacher` và cách in bảng tổng kết. Người dùng gửi dòng: `Teacher a=0.0 [GT sach] 30.55 +20.56 +28.26 DUNG anh`. Cột tương ứng: MPJPE ảnh thật=30.55 mm, mean-real MPJPE=+20.56 mm, mean-real MPVPE=+28.26 mm. Suy ra MPJPE khi dùng feature ảnh trung bình batch khoảng 51.11 mm. Chưa có n, batch/seed, đường dẫn checkpoint, epoch, kết quả blind, MPVPE tuyệt đối hoặc toàn bộ header để xác nhận độc lập; không tự gán các giá trị đó.

Script giữ GT 3D/PE2D, thay feature ảnh trước fusion; mean lấy trung bình feature trong batch, không phải ảnh xám. Phép can thiệp cho thấy ảnh đúng mẫu giúp Teacher GT dự đoán tốt hơn trong đánh giá này, không ủng hộ giả thuyết Teacher hoàn toàn bỏ qua ảnh khi pose sạch. Mốc lịch sử n=1600 mean-real Teacher +20.18 mm ở mục 7 là kết quả cũ riêng; không đồng nhất checkpoint hoặc tập con với dòng mới.

Metric ở đây là H36M-17 root-relative (bao gồm pelvis), MPVPE can pelvis H36M; không so tuyệt đối 30.55/51.11 với MPJPE14=33.502 của mode compare. Ngưỡng nhãn DUNG anh trong script là heuristic 3 mm, không phải kiểm định thống kê. Mean feature vẫn là can thiệp có thể khác phân phối; chênh lệch không đo trực tiếp tỷ lệ phụ thuộc ảnh.

Kết quả chưa xác định block fusion nào mang tín hiệu ảnh hữu ích, cũng chưa chứng minh Teacher sửa được pose S1 nhiễu hoặc Student có thể học tín hiệu đó qua KD. Chưa có lý do từ kết quả này để train lại Teacher chỉ nhằm ép dùng ảnh; có cơ sở nghiên cứu KD hướng vào nhánh đọc ảnh, nhưng chưa triển khai/chốt công thức image-update KD. Nhánh Student của content_ablation_test hiện gọi smpl_model trực tiếp mà không truyền extra_tok spread; không dùng nguyên trạng để kết luận về Student spread/KD. Không sửa source hoặc chạy code dự án ở local.

## 14. KD E1 + gate spread - đã code, chờ train (2026-10-24)

Người dùng duyệt thử gate theo từng joint và sẽ train lại KD. Phạm vi đã thực hiện: thay cách dùng spread trong fusion, giữ nguyên tín hiệu/loss KD E1 hiện tại; chưa chuyển sang KD image-update. Không sửa các YAML/run cũ hoặc repo tham khảo, không chạy Python dự án, train/test/import/syntax check tại local.

Hai YAML/run mới:

| Bản | YAML trong config/ | exp_name |
|---|---|---|
| Gate nhận spread thật | train_student_kd_e1_v2_diff_spread_gate.yml | student_kd_e1_v2_diff_spread_gate |
| Cùng gate, std đầu vào=0 | train_student_kd_e1_v2_diff_spread_gate_zero.yml | student_kd_e1_v2_diff_spread_gate_zero |

Đã đối chiếu tĩnh các giá trị YAML: mọi giá trị của KD spread local được giữ nguyên, chỉ thêm ba key gate đã khai báo trong config.py; hai YAML mới chỉ khác difflift_spread_gate_zero. Cả hai dùng S1/K1, spread S5/K20, seed lệnh chạy 123, Teacher GT cũ, init_student_from_teacher=True, hard_w=lambda=1, KD joint/global=1, proj=0, không noise weighting/InfoNCE; LR `[5,10,15,22,27]`, 30 epoch. Bản zero vẫn tính K/std để giữ cùng hình dạng tính toán và RNG giữa hai run, chỉ che std tại gate. Thư mục kết quả mới do trainer tạo khi chạy, không dùng resume của run cộng token cũ.

Gate: ở mỗi trong 3 block, nhận LN token joint sau self-attn, LN image_update sau out_joint và log1p(std_xyz*10), qua MLP hidden64 ra scalar mỗi joint. `g=2*sigmoid`, khởi tạo g=1; residual `coef1*joint + coef2*(g*image_update)`, giữ FFN và các hệ số cũ. Bật gate thì không cộng embedding spread. Gate dùng cả joint/ảnh, nên g khác 1 không chứng minh spread hữu ích. Đối chứng zero tách đóng góp của std khỏi phần gate dựa vào joint/ảnh. Không ép spread cao luôn dùng ảnh nhiều hơn; spread chưa được hiệu chuẩn thành lỗi joint.

Source đã đổi: config.py khai báo mặc định tắt; ARTS.__init__/forward_student nối std; JointImageSpreadGate và constructor/forward các lớp fusion trong teacher_student.py; Student_Trainer thêm log mean/std/min/max và kiểm tra key khởi tạo Teacher (không sửa công thức loss); main/train.py lưu metadata gate và xác minh lúc test cuối; inspector load_model/capture_real/replay hỗ trợ checkpoint gate, giữ strict load. Metadata zero_std/hidden giúp chặn nạp nhầm cấu hình real/zero cùng shape. Teacher không tạo gate dù được dựng từ YAML Student KD; checkpoint Teacher giữ nguyên.

Inspector: mode spread real dùng gate đã học, off ép gate=1, shuffle đổi std đúng mẫu trước gate. Với bản zero, shuffle không thay thông tin gate; off vẫn có thể khác vì gate đã học từ joint/ảnh. Các checkpoint cộng token cũ giữ nghĩa off/shuffle cũ. Mode compare scratch/KD dùng builder mới và vẫn kiểm tra raw pose/std chung. Gate mới chặn kd_res_mode=v2 vì đường fusion phụ cũ chưa truyền std; content_ablation_test.py nhánh Student cũng chưa nối gate và sẽ báo thiếu std. Không dùng hai đường đó để âm thầm bỏ gate.

Tiêu chí sau train: so real với zero trên cùng evaluator/batch và seed, cả metric tổng lẫn từng mẫu/nhóm lỗi S1 chung; kết hợp kiểm tra inference real/off/shuffle. So với KD cộng token 76.763 mm chỉ là so hai cách dùng spread, không tách riêng thông tin std. Nếu chênh nhỏ cần thêm seed, không suy ý nghĩa thống kê từ một run. Cả hai run giữ train 3DPW-train/chọn best 3DPW test để đối chiếu phát triển; không coi là đánh giá cuối độc lập. Chưa có kết quả gate thực nghiệm.

## 15. Chan doan spread tren KD cu - log nguoi dung moi gui

Nguoi dung gui ket qua du 35,515 mau. Real MPJPE/PA-MPJPE/MPVPE = 76.763/51.030/93.574 mm, trung moc KD E18 da bao. Doan log khong kem header checkpoint/YAML/epoch; theo mach trao doi day la KD cong embedding spread cu, KHONG phai ket qua gate moi. Chua xac minh doc lap checkpoint, seed hay CSV.

| Dieu kien | MPJPE (14) | PA-MPJPE | MPVPE |
|---|---:|---:|---:|
| real | 76.763 | 51.030 | 93.574 |
| off | 76.851 | 51.039 | 93.786 |
| shuffle trong batch | 76.751 | 51.014 | 93.546 |

Don vi mm. Gain off-real = +0.088/+0.009/+0.212; real kem shuffle +0.012/+0.017/+0.028. Median real-off MPJPE=-0.162, MPVPE=-0.296 mm; tot/xau hon >0.1 mm MPJPE=52.7%/39.4%, MPVPE=57.0%/36.5%. Chenh lech mean rat nho; khong suy y nghia thong ke tu mot run.

| Nhom theo spread | n | Spread TB (mm) | MPJPE real | MPJPE off | Gain off-real (mm) |
|---|---:|---:|---:|---:|---:|
| Q1 | 8879 | 26.28 | 65.397 | 65.689 | +0.292 |
| Q2 | 8878 | 33.85 | 69.367 | 69.613 | +0.245 |
| Q3 | 8879 | 40.62 | 77.840 | 77.956 | +0.116 |
| Q4 | 8879 | 53.95 | 94.446 | 94.146 | -0.300 |

Spearman gop 16 khop voi loi S1=0.5591, tung khop 0.091-0.311, nhu log scratch. Day la thuoc tinh std/dau vao, khong phai bang chung fusion su dung hieu qua. Q4 la spread cao, khong dong nhat voi kho nhat theo GT hay che khuat.

Ket luan: nhanh embedding co tac dong nho khi bo, nhung chua thay loi ich ro cua std dung mau; real khong tot hon shuffle, Q4 con xau hon off. Off bo ca bias/embedding va khac phan phoi train; shuffle chi trong batch co the gap frame gan nhau nen chua du ket luan spread hoan toan vo ich. Khong quy gain KD so scratch 0.830 mm cho spread, va khong dung ablation inference de tach tac dong spread khi train.

Gate o muc 14 van la gia thuyet da code/cho chay, KHONG duoc log nay kiem chung. Buoc tiep giu hai run gate real/zero cung kien truc, seed, loss, LR; chi khac std dau vao. Neu real khong cai thien nhat quan so zero thi chua co co so giu std, co the giu gate dua vao joint/anh neu doi chung rieng cho thay loi ich. Khong doi them loss KD trong cap thu nghiem nay. Giao thuc chon best theo 3DPW test chi phuc vu phat trien nhu cac moc truoc.
## 16. Don nhanh cong spread, giu gate KD - da sua code, cho chay

Nguoi dung ban dau yeu cau bo spread/gate, sau do doi uu tien train gate KD va xac nhan pham vi: **chi bo spread, giu GATE**. Pham vi duoc hieu theo lua chon da hoi: bo nhanh cong embedding spread cu; van giu std xyz lam dau vao gate. Khong thay kien truc gate hay loss KD da chot.

Da xoa config/train_student_v2_diff_spread.yml va config/train_student_kd_e1_v2_diff_spread.yml. Hai YAML gate real/zero giu nguyen ten va noi dung. ARTS.__init__/forward_student bo spread_proj/diff_spread va chi sinh K/std khi gate bat; FusionPose2Mesh.forward bo extra_tok rieng nhanh cu, hint_z giu nguyen. Cac key difflift_spread/steps/k giu cho lay std cua gate, config.py ghi ro spread=True phai kem gate=True. Loader validate_spread_gate_checkpoint tu choi checkpoint co spread_proj thay vi de strict=False bo qua nhanh cu. Muon tai hien run spread cu can source/YAML tuong ung cua run do; khong xoa checkpoint, ket qua hay lich su.

Inspector hien co giu ten main/inspect_student_spread.py: mode gate mac dinh, mode spread la alias cho gate; capture_real/replay bo hook/project embedding, off=gate1, shuffle doi std, bo cot spread_embedding_norm. Teacher GT va doc CSV cu giu nguyen. Mode compare ho tro Student DiffLifter thuan/gate, luon so pose S1; chi so std khi ca hai gate, neu khong ghi std_compared=False/std_max_abs_m=None. CSV spread chi co khi do std that va ghi ro nguon. Khong bo qua loi checkpoint de nap run spread cu.

Co san ban thuan config/train_student_v2_diff.yml (khong KD/gate) va ban KD khong gate config/train_student_kd_e1_v2_diff.yml. Khong doi hai YAML nay, loss hay lich train. Gate KD giu 30 epoch, LR [5,10,15,22,27], init Teacher va hai doi chung real/zero nhu muc 14. Lenh train/inspect gate va compare hien hanh trong README. Tat ca moc spread/KD da bao o cac muc truoc la lich su, khong phai ket qua gate. Chua co log train gate moi; gate van la gia thuyet cho kiem chung.

Da doc lai source/diff, kiem tra tham chieu nhanh cu va cau hinh bang van ban; khong chay/import/kiem cu phap Python hay train/test tai local theo AGENTS.md. Khong doi code DiffLifter, cong cu chan doan rieng cua lifter hoac repo tham khao.
## 17. Chan doan pose DiffLifter tren vai tram mau train/test (2026-10-24)

Nguoi dung dang train KD khong gate tren 4 bo va gate KD tren 3DPW-train, muon kiem tra loi joint dau vao tren vai tram mau. Da ra main/eval_difflifter_3dpw.py, diag_difflifter.py, inspect_difflifter.py, oracle_student_diff.py, eval_motionbert_lift.py va check_fusekin_data.py. Eval cu chua gioi han mau/ho tro 4 bo train; diag ep lift_only=True, phan train chi thong ke. Nguoi dung duyet mo rong script eval hien co, khong tao Python moi.

Da sua main/eval_difflifter_3dpw.py: parse_args, eval_mpjpe, main; them expand_sets/seed_worker/validate_student_input/target_label/fit_weights. --train_sets config lay train_list, --sets config lay test_list; --max_samples chon ngau nhien khong hoan lai toi da N moi bo, default 0 giu eval full nhu cu. Lenh de xuat 512 mau/bo, S1/K1, batch32, workers0, seed123. Do rieng dataset, khong dung ratio hay oversampling. --student_input bat key DIFFLIFT.eval_student_input=False moi trong config.py, khong doi YAML train.

Che do Student dung dung loader/target/2D/conf/camera theo YAML, giu lift_only=False va augmentation train; 4 bo can fusekin_unified=True. GT unified cua ca 4 bo lay tu SMPL NeuralAnnot va duoc ghi pseudo-GT; 3DPW theo GT SMPL, H36M test rieng la mocap/GT2D sach. Lay model tu diag.build_model, GT tu diag.get_gt. Student reset noise seed0 moi batch nhu ARTS, du doan ca batch truoc khi loc GT; che do cu giu noise stream. Khong chay image backbone/fusion/gate/KD/Teacher, khong can checkpoint mesh, nhung loader Student van doc anh/dung GT. Chua ho tro selector/pose augmentation/s2d noise trong che do Student.

In MPJPE17, MPJPE16 bo pelvis, per-joint, so chon/hop le/GT rong va nguon GT. Train Student in them metric theo fit_joint_trunc*is_valid_fit; tong loi/tong trong so tap con co the khac trung binh batch trong log train. Khong dung mask 2D lam mask GT; khong so voi MPJPE mesh14. --out_dir tuy chon luu log, summary, CSV mau va YAML, tu choi thu muc da co noi dung; default khong luu giu lenh cu. Dataset loi duoc ghi rieng va tra exit code1 sau khi do cac bo con lai.

Hai lenh trong README: config/train_student_kd_e1_v2_diff_jotr.yml -> 4 bo train + 3DPW test; config/train_student_kd_e1_v2_diff_spread_gate.yml -> 3DPW train + test. Dung lifter experiment/finetune_difflifter_r25_cam/best_epoch.bin; khong thay bang checkpoint Student. Tap con ngau nhien/augmentation/noise theo batch khac phep do toan tap va khac GT native cua lifter; khong quy chenh lech thanh loi tong quat hoa rieng.

Trang thai: da code, doc lai source/diff; CHUA chay/import/kiem cu phap Python hay train/test tai local. Nguoi dung da gui log train/test tap con, xem muc 18. Giu nguyen run/YAML dang train va thay doi san co cua nguoi dung trong train_student_kd_e1_v2_diff.yml.
## 18. Loi pose dau vao tren train Student full dataset - log nguoi dung gui (2026-10-24)

Nguoi dung gui bang TOM TAT tu script da mo rong, thu muc experiment/check_lifter_input_full_s123. Theo mach trao doi la lenh Student input, S1/K1, max_samples512, batch32, workers0, seed123; doan log khong kem header YAML/checkpoint/epoch hay summary.json nen chua xac minh doc lap cac tham so nay. Day la ket qua nguoi dung chay, khong phai eval local cua tro ly.

| Loader / dataset | Hop le / da chon | MPJPE17 | MPJPE16 bo pelvis |
|---|---:|---:|---:|
| train/Human36M | 512/512 | 68.580 | 72.867 |
| train/MuCo | 512/512 | 113.010 | 120.074 |
| train/MSCOCO | 512/512 | 92.476 | 98.255 |
| train/CrowdPose | 280/280 | 165.385 | 175.722 |
| eval/3dpw | 512/512 | 69.843 | 74.208 |

Don vi mm, H36M-17 root-relative, KHONG phai mesh14. MPJPE16 ~ MPJPE17*17/16 phu hop pelvis sai so0; khong la bang chung doc lap pipeline hoan toan dung. 3DPW512 khong so truc tiep voi moc full35515 72.18 mm (checkpoint baseline ep6) de ket luan model da doi; khac tap con/noise/augmentation/cohort. Train co augmentation va target Student khac native target lifter o H36M/MuCo; khong coi train > test la bang chung underfit thuan tuy.

Da doc source/YAML: finetune_difflifter_r25_cam.yaml da khai bao Human36M+MuCo+MSCOCO+CrowdPose, ratio0.25 moi bo, samples_per_epoch512000. Day la gia tri YAML local hien tai, khong xac nhan log/YAML that luc train checkpoint. main/finetune_difflifter.py ep DATASET.lift_only=True; H36M/MuCo hoc native/mocap GT. Student *_jotr.yml dung lift_only=False + fusekin_unified=True, target hoi quy tu mesh NeuralAnnot. MSCOCO/CrowdPose o hai nhanh deu dung pseudo-GT NeuralAnnot va view_rot/camera uoc luong theo cung nguyen tac, nen khong quy moi loi cao cua chung cho chenh lech loai GT.

CrowdPose n280/280: neu max_samples512 nhu lenh da gui thi loader chi con280 mau; can kiem tra dataset_n trong summary va log Load Halpe-26 giu/bo khong NeuralAnnot/bo khong Halpe. Code local CrowdPose.load_data va tools/extract_halpe26_coco.py da khong loc fit_err; file Halpe hoac source tren server co the cu/chi trich mot phan/khac split, day chi la gia thuyet. Khong co cache trong loader CrowdPose local hien tai. Chua doc duoc du lieu/NPZ/annotation tren server, khong khang dinh nguyen nhan. Neu chi280 mau ma ratio25% thi bi lap lai nhieu; train lai khong tu tang do phu du lieu.

Buoc tiep de xuat: dung script hien co voi YAML lifter r25_cam, --train_sets config --sets3dpw --max_samples512 --S1 --K1 de do loi tren native target ma lifter da hoc; khong bat --student_input. Day la phep do native rieng, khong la so ghep cap cung input/augmentation/noise voi bang Student. Truoc khi retrain can xac minh do phu CrowdPose va quy uoc target/camera/flip/rotation. Neu native target tot nhung target Student kem thi xet thi nghiem finetune dong bo target Student; neu native cung kem moi xet retrain/toi uu lifter. Chua sua source/YAML train hay khoi dong run moi, chua dung cac ket qua test nay de chon model cho danh gia cuoi.

## 19. Ket qua nguoi dung bao: KD gate real/zero va KD full dataset (2026-10-24)

Nguoi dung gui ba dong chon best theo ['3dpw']: KD_gate MPJPE77.47/MPVPE94.06; gate_zero 76.11/93.84; KD binh thuong full dataset 76.17/93.08 mm. Day la metric mesh14/MPVPE, khong phai MPJPE17 cua lifter. Chua co PA-MPJPE, epoch, seed train thuc, header TEST CUOI, duong dan checkpoint hay YAML sao luu server. Dong chon best theo trong source in metric epoch hien tai, nen ghi la MOC NGUOI DUNG BAO, chua xac minh la best.pth.tar cuoi run. Chua chay lai tai local.

| Ban nguoi dung bao | Du lieu/Teacher theo YAML local da chot | MPJPE14 | MPVPE |
|---|---|---:|---:|
| KD gate std that | 3dpw-train / teacher_gt_v2 | 77.47 | 94.06 |
| KD gate std dau vao=0 | 3dpw-train / teacher_gt_v2 | 76.11 | 93.84 |
| KD khong gate full dataset | Human36M+MuCo+MSCOCO+CrowdPose / teacher_gt_v2_jotr | 76.17 | 93.08 |

Don vi mm, thap hon tot hon. Da doi chieu lai YAML bang van ban: train_student_kd_e1_v2_diff_spread_gate.yml va _gate_zero.yml chi khac MODEL.difflift_spread_gate_zero. Cung kien truc gate, S1/K1, nhomS5/K20 van duoc tinh o ban zero, khoi tao Teacher, loss E1 va LR [5,10,15,22,27]. Theo ba so da bao, std that kem zero **+1.36 MPJPE/+0.22 MPVPE**. Ket qua run nay khong ung ho loi ich cua std trong gate hien tai, cung huong voi cac ablation inference spread cu. Chua suy y nghia thong ke hay ket luan moi cach khai thac diffusion deu vo ich tu mot cap run.

Zero van co gate hoc tu joint/anh; khong duoc goi zero la Student khong gate hoac suy gate da co loi. So voi KD cong embedding lich su E18 76.763/93.574, zero giam0.653 MPJPE nhung tang0.266 MPVPE; day khong phai doi chung tach rieng tac dung gate. Can KD khong gate cung du lieu/Teacher/khoi tao/loss/LR/seed neu muon ket luan gate.

KD full dataset kem zero0.06 MPJPE va tot hon0.76 MPVPE theo moc bao. Khac du lieu train, Teacher, lich LR ([15,22,27]) va cach lay mau (ratio25%, 32000/epoch), nen khong quy chenh lech cho gate/std/KD. Neu dung dung YAML *_jotr.yml thi khong train 3DPW, nhung van chon best tren 3DPW TEST; chua la giao thuc test doc lap/JOTR ket qua cuoi.

Buoc tiep uu tien de xuat: giu KD full lam moc giao thuc4bo, tam dung mo rong co che std-gate nay. Co san config/train_student_init_v2_diff_jotr.yml; da doi chieu tat ca key/values voi KD full, CHI khac MODEL.kd_lambda=0 thay1. student_use_kd=True/feat giu Teacher init, hard_w1 va privileged giong KD full, van tao projector/Teacher forward nen tach duoc tin hieu KD khoi khoi tao/hard scaling khi cung seed. Day la doi chung can hon viec chi so voi scratch co0.5hard/khong init Teacher. Chua co ket qua init-only full trong log moi, chua khoi dong run hay sua YAML/code model. Xac minh best checkpoint/epoch khi tong hop va giu cac mau test nhu ket qua phat trien.

## 20. Sua validation nguon2D cho 3DPW-Crowd (2026-10-24)

Nguoi dung bao loader in Load Higher-HRNet input roi loi ValueError: cfg.DATASET.pw3d_2d_source khong hop le: halpe26. Da doc PW3D.load_data: self.use_halpe26 = (source=='halpe26') and not cfg.crowd la dung chu y, nhung elif source!='openpose' phia sau lai tu choi halpe26 khi Crowd bo qua nhanh Halpe. Day la loi dieu kien validation, khong phai checkpoint/model hay thieu file Halpe; file HHRNet da doc qua buoc print trong log.

Sua cuc bo data_final/PW3D/dataset.py, PW3D.load_data: elif kiem tra source not in ('openpose','halpe26'). Crowd van dung HHRNet (va fallback OpenPose cu khi thieu annotation), split thuong van nap Halpe theo split/ann_id rieng, key khac van bao loi. Khong doi YAML/nguon detector cua cac run, khong them config hay doi model/train. Tai lieu truoc da ghi dung chinh sach Crowd/HHRNet nhung source con sot validation nay; nay moi sua source, chua coi la da eval thanh cong Crowd.

Da doc lai diff va cac nhanh dieu kien, khong chay/import/kiem cu phap Python hay test/train tai local theo AGENTS. Nguoi dung dong bo file loader len server va chay lai lenh eval dang dung. Co the kiem tra rieng lifter Crowd bang main/eval_difflifter_3dpw.py voi cfg finetune_difflifter_r25_cam.yaml, best_epoch.bin, --sets3dpw-crowd --S1 --K1 --max_samples512 --batch_size32 --workers0 --seed123; day la H36M17, khong phai mesh14.

## 21. Crowd dùng Halpe theo YAML (2026-10-24, sửa lại mục 20)

Người dùng xác nhận muốn Crowd cùng pipeline Halpe và gửi log HHRNet thiếu nhiều ann_id, fallback OpenPose. Sửa mục 20 chỉ bỏ ValueError, vẫn ép Crowd sang detector cũ; nhận định giữ HHRNet là đúng mục tiêu trước đó chưa đúng yêu cầu này.

Đã sửa PW3D.load_data trong data_final/PW3D/dataset.py: kiểm tra nguồn trước khi nạp detector, self.use_halpe26 chỉ phụ thuộc pw3d_2d_source, áp dụng cả Crowd. Crowd đọc 3DPW_validation_halpe26.json đúng ann_id của 3DPW_latest_validation.json, giữ lọc hai sequence Crowd, bỏ hoàn toàn nạp/tra HHRNet khi chọn Halpe. Không fallback OpenPose nếu thiếu Halpe. OC/PC và openpose giữ quy tắc cũ; không đổi YAML/checkpoint/model.

Chưa có kết quả Crowd Halpe sau sửa. Log PC người dùng báo n=2218, MPJPE=95.31/PA=61.91/MPVPE=120.91 mm, nhãn toan bo/ro ri; chưa gắn cho run cụ thể vì chưa có YAML/checkpoint header. Các kết quả Crowd HHRNet/OpenPose cũ không được đổi tên thành Halpe. Chỉ đọc lại source/diff, không chạy/import/kiểm cú pháp Python hay test tại local. Đồng bộ loader và chạy lại lệnh eval cũ; nếu thiếu file Halpe validation thì dùng tools/extract_halpe26_3dpw.py --data_root data_final/PW3D --splits validation trên server, môi trường pose2d.

## 22. Doi chung loss KD tren mot bo 3DPW (2026-10-24, da co ket qua; xem muc 26)

Nguoi dung chon train rieng 3dpw-train de rut ngan thoi gian, khong chay doi chung full luc nay. Nguoi dung xac nhan CrowdPose da full du lieu; lo ngai 280 mau o muc 18 la chan doan lich su, khong coi do phu CrowdPose la blocker hien tai. Chua co metric CrowdPose moi de thay bang cu.

Da tao config/train_student_init_v2_diff_gate_zero_3dpw.yml tu config/train_student_kd_e1_v2_diff_spread_gate_zero.yml: CHI khac MODEL.kd_lambda=0 thay1. Chon gate_zero de tan dung moc nguoi dung bao 76.11/93.84 mm, chi can them mot run. Giu student_use_kd=True, init_student_from_teacher=True, hard_w1/privileged, gate_zero, Teacher teacher_gt_v2, pose S1/K1 va lay std S5/K20, batch32, 30 epoch, LR [5,10,15,22,27]. Khi lambda0 van tinh Teacher/KD feature, loss KD khong dong gop gradient; khong tat student_use_kd vi se doi khoi tao va hard scaling. Khong doi YAML cu, source model hay config default.

Lenh: python main/train.py --cfg config/train_student_init_v2_diff_gate_zero_3dpw.yml --exp_name student_init_v2_diff_gate_zero_3dpw_s123 --seed 123 --gpu 0

Ket qua moi du kien o experiment/student_init_v2_diff_gate_zero_3dpw_s123/checkpoint/best.pth.tar; chua tao checkpoint/chua train. So ghep cap tren cung tap, gate_zero, Teacher/lifter checkpoint va seed (can khop seed run KD thuc), bao MPJPE14/PA-MPJPE/MPVPE. Chi tach tac dong loss KD tren gate_zero/3DPW; khong suy ket luan cho KD khong gate/full dataset hay so truc tiep voi 76.17/93.08. Neu gan nhau can lap seed truoc ket luan. Ca hai chon best 3DPW test, chi phuc vu phat trien. Da doc source Student_Trainer de xac nhan cong thuc loss va doi chieu YAML bang van ban, khong chay Python/train/test theo AGENTS.

## 22. Loi lifter tren train native + chan doan 2D Halpe (2026-10-24)

Nguoi dung gui eval lifter r25_cam best_epoch.bin, S1/K1, toan bo tap train, target native (YAML lifter, lift_only). MPJPE17 mm co xoay / khong xoay (config/eval_difflifter_r25_cam_norot.yaml, chi doi no_rot_aug=True): H36M 60.06/59.04, MuCo 98.38/96.79, MSCOCO 83.50/82.16, CrowdPose 74.08/72.99 (35,659 nguoi, khong phai 280), 3dpw-val 63.19. Tat xoay chi giam ~1 mm -> loi train cao khong do augmentation xoay. Loi train ~ val: lifter khong overfit. MuCo bat thuong (co tay ~185 mm, khuyu ~131 mm), hong/torso thap o moi bo -> khong giong loi don vi/anh xa khop toan cuc. Gia thuyet (chua kiem chung): Halpe bat nham nguoi tren anh ghep MuCo.

Da viet main/diag_halpe2d.py (chua chay): chi doc datalist Human36M/MuCo, so Halpe H36M-17 voi GT 2D pixel anh goc, chuan hoa theo max(w,h) tight bbox, dem mau nghi nham nguoi/doi L/R, ghep samples.csv cua eval lifter theo (dataset,index) de xem loi lifter theo nhom loi 2D. Khong doi YAML train/model.
Bo sung main/diag_halpe2d.py --target_samples (mac dinh 5000): chieu lai GT3D native va SMPL NeuralAnnot H36M17 so voi GT2D, chenh lech root-relative native vs H36M regressor x SMPL; H36M lam moc. Chua chay.
Ket qua nguoi dung gui (5000 mau/bo): chieu lai GT3D native -> GT2D MuCo 0.2 px, H36M 0.0 px (3D/camera nhat quan); SMPL NeuralAnnot -> GT2D MuCo 5.3 px, H36M 2.7 px (NeuralAnnot dung duoc). Native vs H36M regressor x SMPL root-relative: MuCo mean 25.8 mm (Nose 85.6, Torso 52.4, R/L_Knee 39.7/31.4, Neck 34.3; co tay ~13), H36M 18.9 mm. Lech dinh nghia khop MPI-INF-3DHP co that o dau/than/goi nhung KHONG giai thich loi co tay ~185 mm; Spearman(lech target, loi lifter) MuCo = -0.02. Ket luan tam: loi MuCo cao chu yeu do do kho noi tai (pose/goc nhin da dang, mo ho chieu sau chi) + ~5% mau Halpe nham nguoi/~2% doi L/R; khong phai loi camera/don vi. Chua train lai.

### Dinh chinh 2026-10-24 (nguoi dung xac nhan)
Checkpoint experiment/finetune_difflifter_r25_cam/best_epoch.bin dung cho cac eval lifter ngay 24/10 (3dpw 67.91, val 63.19, OC 52.92, PC 80.14, Crowd 77.55, train H36M/MuCo/MSCOCO/CrowdPose) la lifter CO train 3DPW, KHONG phai r25_cam sach trong log 21/10 (khong 3DPW; best ep6, 3dpw 72.18, val 84.58). OC/PC co sequence cua 3DPW train -> so OC/PC cua lifter nay ro ri. Cac so lifter ngay 24/10 o muc 22 thuoc checkpoint co 3DPW. Phan chan doan 2D Halpe va target 3D MuCo (khong phu thuoc lifter) van dung; cac cot ghep loi lifter thuoc checkpoint co 3DPW. Can xac minh Student student_kd_e1_v2_diff_jotr dung lifter nao; neu la ban co 3DPW thi ket qua 76.17/49.92/93.08 va OC 68.17 KHONG theo giao thuc JOTR, khong dat canh JOTR/DPMesh.
Bo sung: nguoi dung xac nhan Student dung lifter sach (log 21/10, khong 3DPW, best ep6). Da doc source: load_model_weights chi bo prefix pose_lifter., nen trong so diff_lifter.* luu trong best.pth.tar cua Student duoc nap lai khi eval -> eval Student khong phu thuoc file lifter hien tai tai duong dan difflift_checkpoint (can xac nhan dong "Load DiffLifter (...) epoch 6" trong log train Student). So sanh dung lifter sach (muc 5: 3dpw 72.18, OC 82.14, PC 93.87) voi mesh Student jotr (76.17, 68.17, 95.31): nhan dinh truoc "Student lam loi tang 8-17 mm so voi lifter" la SAI (dung nham lifter co 3DPW). Voi lifter sach: OC mesh tot hon lifter ~14 mm, PC/3DPW gan ngang -> tren PC/Crowd pose dau vao tu lifter la diem yeu chinh.
Chot cua nguoi dung (24/10, 12:45): coi nhu Student student_kd_e1_v2_diff_jotr dung lifter CO train 3DPW (trong so diff_lifter trong best.pth.tar cho ket qua trung tung chu so voi file lifter hien tai: 3dpw 67.905, OC 52.917, PC 80.142, Crowd 77.545). Chua xac minh best_val_mpjpe cua file; log Student in epoch 6 nen van con kha nang file la ban sach va dau vao 2D 3DPW da doi. Theo gia dinh nay: ket qua Student khong theo giao thuc JOTR; nhan dinh "Student lam loi tang 8-17 mm so voi pose lifter tren PC/Crowd" la dung tro lai (lifter 80.1/77.5 -> mesh 95.3/94.6).

## 23. Chan doan Student trong canh dong nguoi - da sua script, cho chay (2026-10-24)
main/content_ablation_test.py them 2 tuy chon mac dinh tat: --set (tap test, rong = test_list[0] nhu cu) va --out_dir (samples.csv tung mau x dieu kien + summary.txt). In them bang theo quintile loi pose dau vao: mesh that, mesh-pose, (anh TB batch - anh that), % mau anh rieng gay hai; ty le mau pose tot nhung mesh te hon pose >= 30 mm. Metric la MPJPE17 H36M root-rel tu mesh, khong phai dataset.evaluate 14 khop. Teacher GT (cfg.MODEL.TEACHER) chay cung de lam moc pose hoan hao. Chua chay.
Ket qua nguoi dung gui (toan tap, MPJPE17 tu mesh): Teacher GT PC 24.97 / Crowd 25.31 (anh TB chi +1.2-1.4) -> voi pose dung, fusion/head SMPL tot ca trong canh dong. Student PC 85.00 (pose vao 80.0), Crowd 85.00 (pose vao 77.3); anh TB - anh that = +10.3 / +9.9 mm, 91% mau te hon -> anh CO ICH, gia thuyet "anh nguoi ben canh lam hai" bi bac bo (chi 4-20% mau anh rieng gay hai, cao nhat o Q5). Theo quintile pose vao: mesh bam theo pose; Q5 (20% mau) pose 166/172 mm -> mesh 161/169 (anh chi sua ~4-5 mm); Q1 pose 37/26 mm -> mesh 54/53 (+17/+27 mm, trong khi Teacher pose 0 -> 25). Hai diem yeu: (1) ~20% mau loi pose tham hoa trong canh dong, Student khong tu bo qua pose sai; (2) Student khong khai thac het pose tot (san ~50 mm).
Them content_ablation_test.py --oracle_mask_thr (mac dinh 0 = tat; chi tap 3DPW, tu bat DATASET.pw3d_test_gt2d): khop Halpe hop le lech GT 2D (H36M regressor x mesh GT chieu len crop) > nguong (ty le canh crop) -> them 2 dieu kien Student: [oracle mask] (bo khop sai, mask=0) va [oracle fix] (thay bang GT 2D). In ty le khop/mau sai. Chi chan doan, dung GT. Chua chay.
Ket qua oracle (nguong 0.1 canh crop, MPJPE17 tu mesh, Student jotr hien tai, khong train lai): khop Halpe sai 6.4% (PC) / 7.1% (Crowd); mau >=1 khop sai 28.3/29.7%, >=3 khop sai 8.5/9.6%. Mesh: Halpe 85.00/85.00 -> [oracle mask] 80.81/80.40 (-4.2/-4.6) -> [oracle fix] 72.86/71.42 (-12.1/-13.6). Gan nhu toan bo thay doi nam o Q5 (PC 161 -> 142 -> 104; Crowd 169 -> 148 -> 104); Q1-Q4 khong doi. Ket luan: 2D sai chiem ~12-14 mm; mask hoan hao luc test chi lay lai ~1/3 vi Student/lifter chua duoc train de suy khop bi thieu; ngay ca 2D dung Q5 van ~104 mm. Log lan chay nay in "Load DiffLifter (...) epoch 15" -> file lifter hien tai la epoch 15, khac epoch 6 trong log train Student.

## 24. Lifter L2: mask GT + D512 + 40 epoch + chon theo PA - da code, cho chay (2026-10-24)
Nguoi dung chot gop 4 thay doi vao 1 run (khong tach ablation): config/finetune_difflifter_r25_cam_gtmask_d512_e40.yaml, thu muc moi experiment/finetune_difflifter_r25_cam_gtmask_d512_e40. (1) DATASET.halpe_gt_mask_thr=0.1: lib/utils/lift_pseudo3d.gt_mask_halpe, goi trong Human36M/MuCo._halpe_input, lift_pseudo3d_sample va nhanh unified cua MSCOCO/CrowdPose; GT 2D chup vao data['gt2d_px'] ngay sau deepcopy (truoc khi __getitem__ sua joint_img tai cho); chi tap train; COCO/CrowdPose chi kiem khop co GT cung ten. (2) embed_dim 512. (3) 40 epoch, lr_decay 0.99. (4) DIFFLIFT.select_metric='pa_mpjpe'. Log moi: loss x0 tung bo tren thanh tien do + dong log moi epoch; DIFFLIFT.train_eval_samples=2000 -> MPJPE/PA S=1 tren tap con co dinh moi bo train (giu aug train, seed co dinh, khoi phuc RNG). Ca 3 key moi mac dinh tat. Vi gop nhieu yeu to, khong quy duoc cai thien cho rieng mask/D512/epoch. Chua chay; chi doc lai diff, khong import/chay Python local.
Ket qua L2 nguoi dung gui (2026-10-24 toi): best theo PA 3dpw-val = epoch 11/40. Test best: 3dpw 73.11/49.24, H36M 55.45/42.86, OC 80.44/52.86, PC 95.94/60.28, Crowd 99.94/55.88 (MPJPE17/PA, mm). Tap con train ep11: H36M 57.67/43.68, MuCo 91.87/70.19, COCO 84.76/47.02, Crowd 69.51/37.65 (~ngang lifter cu); ep40: 43.61/34.00, 75.70/58.90, 79.00/43.44, 62.86/34.11 nhung 3dpw 77.32, val 94.91, Crowd 112.04 -> sau ep11 la overfit phan bo train (train giam, 3DPW tang). Tang suc chua/train lau khong cai thien tong quat hoa; so voi r25_cam sach ep6 (72.18/82.14/93.87): 3dpw +0.9, OC -1.7, PC +2.1, H36M -7. Chenh MPJPE-PA lon tren du lieu ngoai troi (3DPW 24, COCO train 36, val ep40 41, Crowd 44-52) -> loi chu yeu o huong toan than. Crowd = 2 sequence validation courtyard_hug_00, courtyard_dancing_00. Dung L2 ep11 lam lifter sach cho Student theo giao thuc JOTR.

## 25. Phep thu MotionBERT cung du lieu L2 - da code, cho chay (2026-10-24)
Muc dich: kiem tra kien truc/pretrain co phai nut that lifter. config/finetune_motionbert_jotr4_gtmask.yaml: 4 bo JOTR, Halpe, soft conf, ratio 0.25, 512000 mau/epoch, halpe_gt_mask_thr 0.1, chon best PA 3dpw-val, 20 epoch, MB_release.bin; khac DiffLifter o aug 2D (chi joint_drop 0.1) va loss. main/finetune_motionbert.py them (mac dinh tat): MOTIONBERT.dataset_ratio/samples_per_epoch/train_eval_samples/select_metric/epoch_eval_sets/full_eval_sets/full_eval_freq, MPJPE tung bo tren thanh tien do (TagDataset), MPJPE/PA tren thanh eval. Neu ~72-75 tren 3dpw -> tran do du lieu/mo ho; neu < ~68 -> kien truc/pretrain co tac dung. Chua chay.

## 26. Init-only gate_zero 3DPW: ket qua TEST CUOI (2026-10-24)

Nguoi dung gui header TEST CUOI: experiment/student_init_v2_diff_gate_zero_3dpw_s123/checkpoint/best.pth.tar, epoch12, thieu0/thua0 key. Test 3DPW Halpe-26, 35,515 mau, MPJPE14=77.65, PA-MPJPE=51.72, MPVPE=94.51 mm. Da co ket qua thuc te tren server; khong phai ket qua local. Ban nay van Teacher-init/gate_zero/hard_w1/privileged, chi loss KD lambda0 theo YAML da chuan bi o muc22.

| Cau hinh | MPJPE14 | PA-MPJPE | MPVPE | Muc xac nhan |
|---|---:|---:|---:|---|
| Init-only + gate_zero, lambda0 | 77.65 | 51.72 | 94.51 | TEST CUOI best epoch12, n35515, strict keys khop |
| KD + gate_zero, lambda1 | 76.11 | chua co | 93.84 | Moc nguoi dung bao truoc; chua co header TEST CUOI/epoch |
| Gain init-only - KD | +1.54 | chua tinh | +0.67 | So hai moc da bao |

Hai YAML local chi khac kd_lambda0/1 (da doi chieu lai bang van ban). Ket qua ung ho giu loss KD trong cau hinh gate_zero/3DPW, thay vi quy toan bo loi ich cho Teacher init. Mot cap run khong du ket luan y nghia thong ke, gate co loi, std huu ich hay KD full dataset co cung gain. Truoc khi quy chenh lech RIENG cho KD can xac minh seed, Teacher va trong so diff_lifter trong hai checkpoint that trung nhau: cac muc22-24 ghi duong dan r25_cam da co nhieu phien ban/epoch tren server. Cung chuoi difflift_checkpoint trong YAML khong dam bao cung trong so luc train. Thieu0/thua0 key chi xac nhan nap du kien truc, khong xac nhan lifter giong run KD cu. Chua co header lifter train init-only moi de giai quyet diem nay.

Buoc tiep: xac nhan best checkpoint KD gate_zero va lifter/Teacher/seed chung; sau do so theo mau/nhom loi input chung (neu pose khop) de xem KD sua ca nao. Lap them cap seed neu muon ket luan on dinh. Khong uu tien tang lambda/them spread chi tu hai mean nay. Ca hai chon best 3DPW test, van la giao thuc phat trien. Chi cap nhat tai lieu va doc/diff, khong sua source/YAML, khong chay Python/train/test local.

## 27. KD quan hệ thay MSE joint, giữ global KD (2026-10-24, đã code/chờ train)

Người dùng duyệt thử phương pháp KD khác. Một yếu tố của mục tiêu train đổi: thay MSE feature từng joint qua projector bằng MSE quan hệ cosine giữa các token joint trong cùng mẫu. Giữ KD feature global và mọi thành phần còn lại của run KD gate_zero/3dpw-train. Đây là biến thể lấy ý tưởng relational distillation áp dụng giữa token khớp; không phải quan hệ giữa các sample của RKD gốc, không cam kết buộc Student dùng ảnh hay cải thiện mesh.

Source: lib/core/config.py thêm MODEL.kd_joint_relation=False và kd_joint_relation_log_grad=False. lib/core/base.py: Student_Trainer.__init__ xác minh chế độ và in mục tiêu; _kd_joint_relation tính LN/L2 normalize FP32, R17x17, mask GT hai đầu/bỏ đường chéo, Teacher detach, loss mỗi mẫu rồi mean; train chỉ chọn loss joint mới khi cờ bật. Mẫu <2 khớp hợp lệ loss0, không NaN do chia0. Cấm noise weighting trong chế độ mới để không âm thầm đổi nghĩa trọng số. Giữ tạo projector joint dù loss mới không dùng, nhằm không đổi RNG/nhóm optimizer/resume checkpoint cũ. Không thay kiến trúc/gate/lifter/builder hoặc đường inference.

Log thêm kd_joint_relation/kd_relation_pairs trong W&B và dòng tổng kết epoch; J trên thanh tiến độ là loss quan hệ khi bật cờ, không còn MSE joint vector. Tùy chọn log_grad=True tính norm gradient J/G/hard đã nhân trọng số trên cùng tham số fusion ở batch đầu mỗi epoch qua autograd.grad; không đi vào optimizer .grad/không đổi loss hay tự điều chỉnh hệ số. Log gradient là norm, không đo độ đồng thuận hướng gradient hay chứng minh đường ảnh đã học tốt hơn. Chi phí tăng ba lượt grad tại batch đó; không phải mỗi batch.

YAML mới config/train_student_kd_relation_v2_diff_gate_zero_3dpw.yml: sao từ config/train_student_kd_e1_v2_diff_spread_gate_zero.yml, đối chiếu các key/giá trị sau bỏ comment chỉ thêm hai cờ True. Vẫn Teacher teacher_gt_v2, lifter r25_cam S1K1/std S5K20, gate_zero/initTeacher, kd_lambda=1/w_joint=1/w_global=1, hard_w1, privileged, batch32, 30epoch, LR[5,10,15,22,27], chọn best3DPW test. Không sửa YAML/run cũ, không đổi tỷ lệ loss tự động theo dữ liệu/test. Loss quan hệ khác thang đo MSE cũ; xem norm gradient để nhận biết tín hiệu quá yếu trước kết luận phương pháp không hiệu quả.

Lenh server:
python main/train.py --cfg config/train_student_kd_relation_v2_diff_gate_zero_3dpw.yml --exp_name student_kd_relation_v2_diff_gate_zero_3dpw_s123 --seed 123 --gpu 0

Dự kiến checkpoint experiment/student_kd_relation_v2_diff_gate_zero_3dpw_s123/checkpoint/best.pth.tar. Mốc so: KD gate_zero 76.11/93.84, init-only77.65/51.72/94.51; kết quả relation CHƯA CÓ. Phải cùng Teacher/lifter weights và seed thực, không chỉ YAML path, vì r25_cam trên server có nhiều phiên bản. Đánh giá mesh14 theo dataset.evaluate; phân nhóm theo lỗi input chung nếu đã xác nhận input khớp, không chỉ nhóm theo output scratch. Mới một seed và chọn best trên test nên chỉ kết luận phát triển. Không gán gain cho gate/std hay toàn bộ giao thức full dataset.

Xác minh local chỉ đọc source/diff, so YAML bằng PowerShell và kiểm tra key có khai báo trong config.py. Đã rà các nhánh cờ tắt, mask rỗng/một khớp, detach Teacher và đường gradient Student theo source; CHƯA chạy kiểm thử số, import, kiểm cú pháp Python hoặc train/test do AGENTS.md. Không coi bước rà văn bản là test đã pass trên GPU.

## 28. Visual Teacher relation trước khi train KD (2026-10-24, đã code/chờ chạy)

Người dùng yêu cầu nạp Teacher, chạy và xuất ảnh để xem ma trận quan hệ thực tế trước khi đánh giá phương pháp KD mới. Đã rà các script main: inspect_student_spread.py có strict load Teacher GT/GT input và hàm vẽ pose dùng được; content_ablation_test.py dùng cho ảnh mean/blind, chưa có heatmap token relation. Chọn mở rộng script đầu, không viết Teacher builder mới.

Đổi main/inspect_student_spread.py: parse_args/load_model/main hỗ trợ --mode teacher_relation; thêm inspect_teacher_relation. Chọn tối đa512 mẫu ngẫu nhiên toàn split, seed cố định, sample_ids tùy chọn; Teacher.eval/no_grad với GT3D sạch, xác minh input thực trùng GT, PE2D detector. Thu feat_joint_in trước và feat_joint sau fusion, dùng shared helper cosine(LN(token)) FP32. GT mesh đủ17 khớp: không mask theo detector; loại đường chéo khi thống kê/vẽ. Hình overview mean/std giữa mẫu trước-sau fusion, mỗi hình mẫu gồm crop/detector/GT-pred pose/front-side/ma trận trước-sau/phần lệch mean. Dùng chung thang [-1,1] cho cosine và chung biên màu phần lệch, không tự phóng đại từng mẫu. Matrix diagonal màu xám. Xuất NPZ ma trận/GT-pred pose/indices/names, CSV mẫu và136 cặp, summary/config/log. Không có annotation Student/KD hay chạy thêm Student/lifter.

Gom phép tính relation vào lib/utils/joint_relation.py::joint_relation_matrix, dùng lại trong Student_Trainer._kd_joint_relation để visualization và target KD không lệch công thức. Giữ nguyên LN/L2 normalize/FP32, detach Teacher/mask/chia số cặp/loss/gradient Student; không đổi mặc định hay YAML/run đang train.

Lenh:
python main/inspect_student_spread.py --mode teacher_relation --cfg config/train_teacher_k16_v2.yml --ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar --set 3dpw --relation_samples 512 --num_vis 12 --batch_size 32 --workers 0 --seed 123 --gpu 0

Đầu ra mặc định cạnh checkpoint trong teacher_relation_3dpw_<timestamp>, không ghi đè thư mục có dữ liệu. Cần đúng YAML Teacher checkpoint, kể cả img_probe; strict load không bỏ key để giả nạp được. Metric pose trên hình là MPJPE17 từ mesh, không phải mesh14. Before đã chứa joint encoder/PE2D, không phải pose 3D thuần. STD là cùng cặp thay đổi giữa mẫu, không phải khác biệt giữa cặp trong một mẫu. Constant-template MSE dùng ma trận trung bình toàn tập con, là thống kê mô tả/không phải kết quả test độc lập. Chưa có hình/CSV/log GPU nên chưa kết luận quan hệ có thay đổi đủ hay hữu ích cho KD; sự thay đổi cũng có thể do GT pose/identity token, không riêng nội dung ảnh.

Chỉ đọc lại source/diff/luồng args và phép biến đổi tensor bằng văn bản; CHƯA chạy model/import/kiểm cú pháp Python hay test/train local theo AGENTS. Người dùng chạy server rồi gửi overview/hình mẫu/summary để phân tích.

## 29. Heatmap Teacher / KD hien tai / KD relation (2026-10-24, da code/cho chay)

Nguoi dung duyet mo rong visual Teacher de xem ca Student KD hien tai va so voi Student sau train KD quan he. Da sua main/inspect_student_spread.py: parse_args them cac cap --student_cfg/--student_ckpt va --relation_cfg/--relation_ckpt (chi teacher_relation, yeu cau du cap; relation can Student hien tai). main giu defaults truoc update_config va route inspect_relation_compare neu co student_cfg; Teacher-only cu van chay. plot_joint_relation tach dung chung cho hai duong visual. Khong sua loss/config model/YAML/run train trong luot nay.

inspect_relation_compare dung lai builder load_model strict va snapshot config hien co. Teacher/Student la cac model rieng, moi forward phuc hoi dung cfg cua model. Cung dataset loader Student KD hien tai, mau chon ngau nhien khong hoan lai/seed/sample_ids, detector/crop/camera/GT chung. Teacher inputGT duoc xac minh, Student input la lifter cua checkpoint. Chi ho tro Student feat KD DiffLifter S1K1 root-relative, khong selector/hint; current YAML co kd_joint_relation=False, new YAML True. Khong tu lay checkpoint "gan nhat" hay tao model relation chua train. Khi hai Student cung co: kiem tra pose S1 maxabs <=1e-6m, std neu ca hai gate; sai thi dung ro loi. Config train khac ngoai co relation/log duoc luu/canh bao, parity khop khong dong nghia da tach moi confounder train.

Output overview.png: cot tung model, hang mean before/after fusion va std across samples before/after. Hinh moi mau: crop/GT va pose ba model, matrix before/after, delta Student AFTER - Teacher AFTER; cung thang [-1,1] cho cosine va chung thang delta tren tat ca model/mau. Chua co relation checkpoint thi hien Teacher+current, cot relation khong gia lap. Samples.csv ghi MPJPE17 tu mesh, loi input lifter, MSE raw relation vsTeacher tung stage; neu co new thi gain current-new cho relation MSE va MPJPE17. NPZ luu matrices/poses/sample_ids/names, pairs.csv mean/std136cap tung model, summaryJSON epoch/config differences/parity, YAML moi model va log.

Lenh Teacher+KD gate_zero hien tai:
python main/inspect_student_spread.py --mode teacher_relation --cfg config/train_teacher_k16_v2.yml --ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar --student_cfg config/train_student_kd_e1_v2_diff_spread_gate_zero.yml --student_ckpt experiment/student_kd_e1_v2_diff_spread_gate_zero/checkpoint/best.pth.tar --set 3dpw --relation_samples 512 --num_vis 12 --batch_size 32 --workers 0 --seed 123 --gpu 0
Sau khi train relation, them --relation_cfg config/train_student_kd_relation_v2_diff_gate_zero_3dpw.yml --relation_ckpt experiment/student_kd_relation_v2_diff_gate_zero_3dpw_s123/checkpoint/best.pth.tar. Dung ten run/checkpoint thuc neu khac vi du. Output moi mac dinh compare_relations_<split>_<timestamp> canh checkpoint current, khong ghi de.

MSE relation la raw cosine gap khong projector, khong dong nhat loss KD E1 current; nho hon chi cho thay gan Teacher o thong ke nay, can doc ca MPJPE17, khong suy ra mesh14. Truoc/sau fusion la feature stages, current/relation la hai run; chua co ket qua relation de ket luan. Tat ca mode la chan doan, khong train/chon checkpoint. Moi ra source/diff/luong args/ctx/shape theo van ban, CHUA chay model/import/kiem cu phap Python hay train/test local theo AGENTS.

## 26. Student KD jotr voi lifter L2 epoch cuoi - da tao YAML, cho chay (2026-10-24)
config/train_student_kd_e1_v2_diff_jotr_l2last.yml: giong train_student_kd_e1_v2_diff_jotr.yml, chi doi MODEL.difflift_checkpoint -> experiment/finetune_difflifter_r25_cam_gtmask_d512_e40/latest_epoch.bin (ep40, KHONG 3DPW). Theo yeu cau nguoi dung dung latest thay vi best ep11. Student khong bat halpe_gt_mask_thr. Van chon best theo 3DPW test nhu YAML goc. Lan dau tien Student va lifter deu khong thay 3DPW (theo YAML) -> so sanh duoc voi JOTR/DPMesh (kem luu y bbox GT/RTMPose, chon theo test).

## 30. Visual attention joint->image that truoc khi KD (2026-10-24, da code/cho chay)

Sau khi xem flow va thao luan paper2025-2026, nguoi dung duyet visual Teacher/Student hien tai: khac ro thi xet KD, giong thi doi huong. Chua duyet hay khoi dong run loss attention. Ma tran cosine17x17 truoc do khong la attention17x64. Da doc RGBJointCrossTransformerBlock.forward: attn_j=softmax(Qjoint* Kimage^T /sqrt(d)), feat_layers co san attn_joint_to_img, khong can hook/sua model.

Thay doi: main/inspect_student_spread.py parse_args/main/inspect_relation_compare them mode attention va CLI attn_block/attn_joints/alias max_samples. Dung lai Teacher/Student strict builder va default config snapshots, chung loader Student, khong nhan relation checkpoint trong mode nay. Chi branch chan doan lay attention; Teacher_relation va compare cu giu luong cu. Tao lib/utils/attention_diagnostics.py voi extract_joint_image_attention/save_attention_diagnostics. Lay mean heads cua block cuoi mac dinh, grid8x8 (bao loi neu khac), kiem tra phan bo huu han/khong am. Khong can head1Teacher trung y nghia head1Student; luu entropy trung binh tunghead va JS head-vs-ownmean de canh bao trung binh che khac biet.

Visual12 mau ngau nhien trong512, moi mau mac dinh6 khop co tay/khuyu/co chan. Moi hang4cot cropdetector, Teacher overlay, Student overlay, deltaStudent-Teacher; scale xac suat va delta dung chung tren tat ca mau/model, interpolation nearest theo o8x8. Overview la ma tran17x64 mean moi model, meanabs delta tren tungmau (khong tru hai mean roi layabs de tranh triet tieu), perjointTV/entropy, TV theo inputerror. Attention khong bat buoc peak tai detector (detector co the sai). Khong ep attention moi/khong chay StudentvoiGT3D.

So do: TV=0.5sum|p-q|, JS/log2 trong[0,1], KL(T||S) nats; entropy/log64 (1=uniform); Teacher-to-own-template TV (mean theo CUNG joint giua mau) de nhan biet map gan co dinh. Q1-Q4 theo StudentinputMPJPE16 (bo pelvis), cung cohort, khong chon nhom theo Studentoutput. CSV perjoint luu detector_valid/loi input va meshjoint17 cua ca hai, du17GTjoint. Summary bao TVmean/median/p90, JS, entropy, headdiversity, templateTV, epoch/indices/block/headcounts/protocol. Can >=2mau de doc templateTV. Khong dat nguong auto ket luan/khong tune model tren test.

Lenh server:
python main/inspect_student_spread.py --mode attention --cfg config/train_teacher_k16_v2.yml --ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar --student_cfg config/train_student_kd_e1_v2_diff_spread_gate_zero.yml --student_ckpt experiment/student_kd_e1_v2_diff_spread_gate_zero/checkpoint/best.pth.tar --set 3dpw --max_samples 512 --num_vis 12 --attn_block -1 --batch_size 32 --workers 0 --seed 123 --gpu 0
Doi dung YAML/checkpoint run that neu tenkhac. Mac dinh output compare_attention_3dpw_<timestamp> canh Studentckpt, thu muc moi, co the --out_dir. --attn_joints all ve17khop, --attn_block0/1 de xem cac block som; lenh copy day du trong README.

Trang thai: da code/ra source va diff, chua co hinh hay so lieu attention moi. Chua chay/import/kiem cu phap Python, test/train hay modelGPU local theo AGENTS. Khong sua YAML/run/train/loss/checkpoint hien co. Khac attention chi la co so de THU KD, chua chung minh Teacher map hop ly hon. Giong meanheads la it tin hieu cho muc tieu mean-head KD, khong ket luan moi cach KD deu voich.

### Ket qua attention nguoi dung bao (2026-10-25)

Nguoi dung gui log 512 mau (4 nhom, moi nhom 128). Chua nhan summary/YAML/epoch de xac minh dung cap checkpoint, split va block thuc te; khong tu gan log cho lenh vi du o tren. Day la ket qua server do nguoi dung bao, khong phai chay local.

- Mean-head TV mean=0.077004, median=0.068995, p90=0.115972; JS/log2 mean=0.007702.
- Entropy/log64 Teacher=0.892323, Student=0.888096; Teacher-vs-own-template TV=0.204324.
- Q1 input43.46mm: TV0.059992, JS0.004387; Q2 input59.15mm: TV0.070209, JS0.006066.
- Q3 input74.15mm: TV0.076148, JS0.007161; Q4 input115.67mm: TV0.101667, JS0.013193.

Doc log: mean-head attention kha gan nhau; Q4 TV cao hon Q1 khoang1.69 lan. Teacher map thay doi giua mau (template TV lon hon gap Teacher-Student trung binh), khong phai map hoan toan co dinh. Entropy cao cua mean heads khong chung minh tung head deu phang hay model bo qua anh. Khong co nguong TV/JS pho quat de ket luan KD se thanh cong/that bai; tuong quan nhom inputerror khong chung minh nhan qua. Attention khac khong dong nghia Teacher map dung hon.

Ha uu tien KD mean-head attention tren moi mau so voi de xuat truoc khi co log; chua them loss hay khoi dong run. Buoc tiep de quyet dinh: doc groups.csv/joints.csv da xuat de doi chieu Teacher/Student mesh MPJPE17 voi TV (khong so truc tiep mesh14), neu tiep tuc chan doan thi xem vector cap nhat tu anh A*V qua output projection va decoder/output. Cung attention chua dam bao cung value/gate/feature hay pose. Chua co ket qua KD attention de ket luan phuong phap khong hieu qua. Anh dinh kem chua doc duoc noi dung pixel trong phien nay; nhan dinh dua tren log va source cong thuc.Dinh chinh nhan dinh 25/10: "Teacher GT gan nhu khong dung anh" chi dung cho teacher_gt_v2_jotr (4 bo, khong 3DPW) do tren PC/Crowd (+1.2/+1.4 mm). Teacher GT v2 train 3DPW (muc 13) do tren 3DPW: anh TB +20.56 MPJPE / +28.26 MPVPE -> CO dung anh. Hai Teacher, hai tap khac nhau; khong suy chung.

## 27. Tran oracle tap ung vien pose (y tuong Teacher-Critic) - da code, cho chay (2026-10-25)
Y tuong: thay vi chinh KD, dung Teacher (biet pose nao khop anh) lam critic cham diem tap ung vien pose co cau truc, chon cung roi dua vao Student. Buoc 0 la do tran bang GT. content_ablation_test.py them --oracle_cand K (mac dinh 0 = tat) va --cand_steps (5): ung vien = K gia thuyet DiffLifter (seed 0), phan xa do sau toan than z->-z (+/- doi T-P), lat do sau doan tren/duoi tung chi (4 chi x 4 trang thai, giu x,y), va hop tat ca (moi ung vien + lat chi). Chon bang GT -> chi la tran; Student giu nguyen (pose chon co the lech phan bo luc train Student). Neu tran mesh giam ro so voi oracle K=20 cu -> dang xay critic; neu khong -> dung.

## 28. Script mo xe Teacher - main/analyze_teacher.py (2026-10-25, chua chay)
Muc tieu: tim kien thuc nao cua Teacher GT co the truyen truoc khi thiet ke lai KD. Eval-only, dung lai build_and_load cua content_ablation_test. A: san Teacher voi pose GT (MPJPE17/PA/MPVPE, goc xoay toan than Procrustes, loi tung khop). B: Teacher voi pose GT bi lam hong co kiem soat (Gauss 10/20/40/80 mm, lat do sau 1 chi, phan xa do sau toan than), anh that vs anh TB -> anh co giup sua pose sai khong. C: Student voi pose lifter / pose GT; Teacher voi pose lifter. D: activation patching Teacher -> Student (token khop sau bo ma hoa; token khop / token anh / ca hai sau fusion) -> % khoang cach lap duoc. E: attention khop->anh tung block, entropy va do lech tam so voi GT 2D (3DPW, pw3d_test_gt2d).

## 29. KD chon loc (y tuong nguoi dung) - da code, cho chay (2026-10-25)
MODEL.kd_sel (mac dinh False) + kd_sel_gap_q (0.7), kd_sel_teacher_joint_mm (50), kd_sel_teacher_sample_mm (0 = bo). Trong Student_Trainer (kd_mode feat, khong ho tro kd_joint_relation): mask no_grad = khop hop le & loi khop Teacher (mesh Teacher -> H36M-17 so GT) <= nguong & (tuy chon) MPJPE mau Teacher <= nguong & khoang cach KD khop <= phan vi q trong batch; global KD dung mask mau (Teacher dung + gap global <= phan vi q). Nhan vao jw_valid / sw (van chia theo n_valid nen do lon loss ~ ty le duoc chon). Log kd_sel_joint, kd_sel_sample, teacher_err_mm. YAML config/train_student_kd_e1_v2_sel.yml = train_student_kd_e1_v2.yml + 4 key tren. Chua chay.

## 30. Kiem chung quy uoc loss mesh/root va projection (2026-10-25, da co synthetic va64mau train/cho checkpoint)

Nguoi dung duyet viet phep kiem chung truoc khi coi hai khac biet quy uoc la loi. Them `main/check_loss_conventions.py`, khong sua model/loss train/YAML/run hien co. Script doc `cfg` hien co va cac tuy chon CLI rieng; khong them hanh vi/toggle vao pipeline train. Danh gia co chu dich hay cai thien sau retrain CHUA CO, khong goi ket qua doc source la loi da kiem chung.

- Synthetic: cung mot mesh SMPL neutral lam prediction va GT, co T-pose va pose/shape ngau nhien, camera kiem soat 3-6m. Goi chinh `FusionPose2Mesh.get_coord` qua context chi gom SMPL/regressor, khong dung backbone/lifter. So mesh tru pelvis SMPL voi cung mesh tru pelvis H36M; so projection SMPL-30 chon theo ten voi projection H36M-17, cung camera va epsilon. Co self-control cung root/cung regressor, doi chieu get_coord, FP64 tren cung mesh FP32, sai lech do epsilon 1e-4 rieng va doi chung chuan hoa tong hang regressor CHI de phan tich. Dung sai mac dinh 0.01mm/0.01px la sai so so hoc, khong la nguong y nghia thuc te hay thong ke.
- Mau train: toi da64 mau ngau nhien khong hoan lai MOI bo DATASET.train_list (hoac --sets), workers0/seed123, giu augmentation theo YAML va lift_only=False. Ghi pelvis SMPL/H36M cua mesh target da luu, phan du khi cung mesh do duoc tru pelvis SMPL nhu prediction; doi chieu mesh->jointGT. Do dinh nghia khop tren mesh that voi camera AO 3m: khong gia vo do la camera/GT2D dataset. Nhanh nay chi dung bo train co target mesh + H36M17 (3dpw-train hoac cac bo fusekin_unified). Geometry chi gom fit hop le; thong ke target/checkpoint ghi valid_fit_n, loss checkpoint van giu mask0 nhu trainer.
- Tuy chon --ckpt: nap Teacher GT thuan hoac Student khong hint bang strict=True, kiem metadata gate; dung wrapper forward eval that (Student gate van sinh std), GT2D va mesh target that tren cung batch. Replay get_coord theo pose/shape/camera checkpoint va dung neu khong khop wrapper trong dung sai; tinh loss hien tai va loss doi chung dong nhat root/regressor. Do gradient cuc bo theo SMPL axis-angle pose, beta va camera translation o batch dau moi bo, ghi norm va cosine cung loss truoc/sau thay quy uoc. Giu DETACH_POSE_FOR_PROJ: projection khong co gradient pose khi co bat. Doi chieu mean loss hien tai voi chinh CoordLoss/JOTRCoordLoss cua trainer; khong AWL/KD/hard_w, khong backward vao network/optimizer/update/save checkpoint. Gradient gan loss0 cua L1 nhay voi lam tron; phai doc cung FP64/dung sai. Khong ho tro refiner diffusion/Teacher mixing/hint, khong loc key de gia nap checkpoint cu.

Lenh 1, chi kiem hinh hoc, khong can checkpoint hay anh dataset (van can SMPL/regressors va moi truong repo):

```bash
python main/check_loss_conventions.py --cfg config/train_student_kd_e1_v2_diff_spread_gate_zero.yml --synthetic_only --gpu 0 --out_dir experiment/check_loss_synthetic_s123
```

Lenh 2, them64 mau train that, khong can checkpoint:

```bash
python main/check_loss_conventions.py --cfg config/train_student_kd_e1_v2_diff_spread_gate_zero.yml --samples 64 --batch_size 8 --seed 123 --gpu 0 --out_dir experiment/check_loss_train_s123
```

Lenh 3, them loss/gradient tu checkpoint gate_zero (doi dung ten run/checkpoint thuc neu khac):

```bash
python main/check_loss_conventions.py --cfg config/train_student_kd_e1_v2_diff_spread_gate_zero.yml --ckpt experiment/student_kd_e1_v2_diff_spread_gate_zero/checkpoint/best.pth.tar --samples 64 --batch_size 8 --seed 123 --gpu 0 --out_dir experiment/check_loss_ckpt_gate_zero_s123
```

Output thu muc moi, tu choi ghi de: `summary.json`, `samples.csv`, `joints.csv`, `gradients.csv`, `regressors.csv`, `config.yaml`, `log.txt`. Luu indices/seed/epoch va SHA256 config/checkpoint/regressor/source, lifter constructor neu co. `synthetic_checks` chi danh dau phan du vuot dung sai so hoc khi self-control/replay khop; khong auto ket luan bug hay loi ich thay loss. Cac loss L1 nay KHONG la MPJPE14/MPVPE, khong so truc tiep voi moc76.11/93.84. Projection sample that co camera ao va projection checkpoint voi targetGT that duoc gan stage rieng. Neu mean loss moi thap hon o cung checkpoint thi chi la muc tieu khac; muon xac nhan loi ich can cap train doi chung tung loss rieng, dung run moi va giao thuc chon checkpoint ro rang.

Trang thai luc viet: moi viet va doc lai source/diff/mask/don vi/luong CLI, CHUA chay script/import/kiem cu phap Python hay train/test local theo AGENTS.md. Chua doi quy uoc lifter/mesh hay sua loss dang train. Ket qua server nguoi dung cung cap cap nhat ben duoi.

### Synthetic nguoi dung bao (2026-10-25)

Nhan summary status=completed, --synthetic_only, seed123, n16, camera3-6m, YAML gate_zero/3dpw-train; khong checkpoint/dataset that. SHA256 script 108520576532e80c36be7b9a5a88fa7322e8128eaa4dcd7a0b487d370cde83cc trung file local khi doi chieu. Day la phep do server do nguoi dung gui, khong phai chay lai local.

| Phep do | Mean | Doi chung |
|---|---:|---|
| Lech pelvis SMPL/H36M tren cung mesh | 29.7163 mm | Chuan hoa tong hang van 29.5205 mm |
| L1 mesh hoan hao, hai root hien tai | 14.8811 mm/toa do | FP64 14.88109; loss_graph cung root H36M 0.001298 mm |
| Lech dinh nghia17khop tren cung mesh | 65.4000 mm | Khong la loi lifter/model |
| Lech projection hai regressor | 68.6395 px, Euclid/khop | FP64 68.63948; chuan hoa tong hang 68.63948 |
| Lech chi do epsilon1e-4 | 0.01404 px | Nho hon nhieu so voi lech regressor |
| L1 projection loss_graph hien tai | 10.90027 heatmap/toa do | Cung regressor H36M =0 |

Replay get_coord mesh/projection lech0, self-control cung root/cung regressor lech0, hai co phan du FP64 vuot dung sai deu True. Xac nhan phan du do quy uoc tren phep kiem chung synthetic; khong giai thich bang FP32/epsilon hay chi tong hang regressor. Khong suy ra y dinh tac gia, khong quy 14.88mm L1 nay thanh MPJPE14/MPVPE hay ky vong gain14.88mm khi sua. Projection68.64px dung focal cua model/camera ao3-6m va tat ca17khop, chua mask crop nhu trainer; KHONG la sai so crop thuc tren 3DPW. Chua nhan joints.csv de xac dinh khop nao chi phoi, chua nhan gradients.csv de dien giai gradient.

Buoc tiep sau synthetic luc do: chay lenh2 tren64mau train that, doc stored_target_joint_parity/root va joints.csv; sau do lenh3 voi dung YAML/checkpoint da train de do loss/gradient cuc bo. Ket qua64mau cap nhat ben duoi; chua duyet/sua loss hay khoi dong run moi trong lan doc log nay.

### 64 mau 3DPW-train nguoi dung bao (2026-10-25)

Summary moi status=completed, synthetic_only=False, checkpoint=null, out_dir=experiment/check_loss_tra, seed123, random64indices da luu, valid_fit64/64. YAML/source/regressor SHA256 nhu lan synthetic. Day la ket qua server nguoi dung gui, khong chay local.

| Phep do | Mean | Max |
|---|---:|---:|
| Norm pelvis SMPL tren mesh target da luu | 34.0180 mm | 55.2396 mm |
| Norm pelvis H36M tren cung mesh target | 0.3172 mm | 0.7095 mm |
| Mesh->H36M rootrel so voi orig_joint_cam | 0.2458 mm | 0.5499 mm |
| L1 cung hinh hoc sau tru pelvis SMPL so mesh target luu | 17.4371 mm/toa do | 30.9729 mm/toa do |
| Lech dinh nghia17khop, camera ao3m | 64.8381 mm | 69.6479 mm (mean17khop/mau) |
| Lech projection hai regressor, camera ao3m | 94.7126 px/khop | 111.3195 px (mean17khop/mau) |

Bang xac nhan truc tiep tren target that: mesh luu gan rootH36M=0, pelvisSMPL con lech~34mm; model lai tru pelvisSMPL. Phan du L1~17.44mm/toa do xuat hien khi hinh hoc da trung GT, la bat nhat quy uoc giua target va prediction trong mesh loss. Khong la loi chi co o synthetic. Mesh->jointGT gan khop (0.246mm), khong co bang chung lech don vi lon trong64mau nay. Sai lech nho van can phan biet voi che do tong hang regressor/lam tron, khong tu coi moi lech nho la0.

Nhanh camera ao: pelvis gap34.0361mm, chuan hoa tong hang34.0430mm; L1 perfect17.4474mm (FP6417.44743). Projection94.7126px khong doi sau FP64/chuan hoa tong hang; epsilon chi0.02136px. Xac nhan khac dinh nghia khop tren mesh that, NHUNG projection van o camera AO focalmodel3m/tatca17khop, khong phai sai so projection loss tren crop3DPW that. Khong so94.71 voi68.64 synthetic de ket luan train te hon, vi camera/pose khac. Chua co joints.csv de quy khop chi phoi.

Khong can lap lai synthetic/64mau de xac nhan root mismatch. Neu can dinh luong tac dong tai model da hoc thi chay che do checkpoint tren cung cohort; JSON hien tai chua co checkpoint stage/gradient cua model that. Loi ich retrain va nguyen nhan cac moc MPJPE14/MPVPE cu van chua duoc xac minh, khong tru17.44/34mm khoi metric. Chi cap nhat tai lieu, khong sua source/loss/YAML hay chay code local.

## 30. Phan tich truoc KD chon loc / co trong so - main/analyze_kd_selection.py (2026-10-25, chua chay)
Nguoi dung yeu cau phan tich truoc khi code trong so. Script eval-only: tung khop tung mau tren tap train cua Student (loader train YAML, co aug, --max_samples moi bo) va tap test: e_T (Teacher), e_S (Student KD), e_lift, d = 1-cos(LN token khop Student sau fusion, LN Teacher); mau: d_glob, MPJPE_T/S. Bao cao phan phoi, Spearman (d~e_S, d~e_lift, d~e_T, e_T~e_S), bang 4 o (Teacher tot<=50mm x d<=trung vi), thap phan vi theo d va e_T, goi y tau_T (p75 e_T) / tau_d (trung vi d), samples.csv. --init_ckpt (tuy chon) do loi ich KD theo nhom. Checkpoint nguoi dung co: Teacher teacher_gt_v2_jotr, Student student_kd_e1_v2_diff_jotr_l2last (ca hai khong 3DPW); chua co init-only cung lifter L2 nen chua do duoc loi ich KD. Code kd_sel (muc 29) giu tat, chua dung.
Ket qua nguoi dung gui (2000 mau/bo, Teacher teacher_gt_v2_jotr, Student kd_e1_v2_diff_jotr_l2last; MPJPE17 tu mesh): Teacher tot tren COCO/CrowdPose (p90 24/21 mm), H36M (p90 43), kem tren MuCo (TB 41, 28.8% khop >50 mm, Nose/Head ~70) ; test 3DPW/PC/Crowd TB 20.5/25.1/25.3, >50 mm 4-6%. Student ~ lifter: e_S vs e_lift H36M 35.3 vs 55.7 (sua duoc), MuCo 95.5/99.7, COCO 83.1/83.2, CrowdPose 65.0/66.8, 3DPW 67.4/68.0, PC 81.8/74.1, Crowd 84.6/79.4 -> ngoai H36M Student gan nhu chep pose lifter. d tuong quan manh voi e_S (0.50-0.73) va e_lift (0.48-0.72); thap phan vi d: e_S ~ e_lift o moi bin. O "Teacher tot & d lon" 44-50% khop, e_S 90-128 vs e_T 15-30 -> noi co kien thuc can truyen; o "Teacher kem & d nho" chi 0-1.7% (MuCo 7.4%). Ket luan: loc bo khop d lon se bo dung phan KD can; loc theo chat luong Teacher chi co tac dung chu yeu tren MuCo. Nut that: Student khong sua pose bang anh (tru H36M) du da co GT supervision.
Ket qua analyze_teacher (3DPW, 2400 mau, teacher_gt_v2_jotr + student l2last, MPJPE17 tu mesh): Teacher GT 20.47 (PA 13.78, rot 1.4 do); anh TB chi +0.9..+2.2 mm o MOI muc lam hong -> Teacher gan nhu khong dung anh de sua pose. Gauss 15/30/60/121 mm vao -> 23.0/28.8/44.3/79.8 (khu bot nhieu nho prior bo giai ma, khong nho anh); lat do sau 1 chi 33 -> 47 (te hon); phan xa toan than 279 -> 266. Teacher(pose lifter) 68.40 ~ Student(pose lifter) 68.64 -> Student khong hon Teacher dung pose lifter. Student(pose GT) 39.65 vs Teacher 20.47 -> bo giai ma Student kem ~19 mm khi pose tot. Patching: token khop sau bo ma hoa cua Teacher -4.1% (OOD), token khop sau fusion +43.1% (47.89), token anh sau fusion 0.1%. Attention Teacher/Student gan nhu trung (entropy ~0.94-0.98, tam lech ~0.15-0.20) -> KD attention vo nghia.
Phan F (800 mau 3DPW): coef1/coef2 Teacher 1.04/0.90, 0.98/0.87, 0.95/0.86; Student 1.03/0.84, 0.93/0.80, 0.87/0.75. ||coef2*joint_att||/||coef1*joint_tok|| Teacher(pose GT) 0.48/0.24/0.21, Student(pose lifter) 0.31/0.22/0.20. Ket hop anh TB chi +0.9 mm cho Teacher (Student +8.3): phan anh cua Teacher co do lon 20-48% nhung gan nhu khong mang thong tin rieng tung mau -> co dinh fusion 50/50 khong co ich; can tin hieu train buoc doc anh (Teacher-struct).

## 31. Can tren oracle chon phuong an DiffLifter - main/oracle_best_hypothesis.py (2026-10-25, chua chay)
Ket qua lan do truoc (content_ablation --oracle_cand 20, 3DPW 2400 mau, Student l2last, MPJPE17): pose/mesh mac dinh 69.2/68.6; oracle K=20 ca pose 55.9/60.0; lat mat-lung toan than 68.6/67.7; lat do sau tung chi 62.3/63.3; gop tat ca 49.6/55.1. Script moi chay nhieu tap: muc L0 mac dinh, L1 trung binh K, L2 oracle ca pose, L3 oracle theo nhom (4 chi + than-dau), L4 oracle theo khop, L5 nhom + lat chi, L6 L5 + phan xa toan than; bao pose MPJPE17/PA, loi tung khop, va mesh khi dua pose da chon vao Student / Teacher (tuy chon).
Ket qua oracle_best_hypothesis (Student l2last + Teacher jotr, K=20 S=5, 4000 mau/tap; pose MPJPE17 / mesh Teacher giai ma): 3DPW L0 68.3/67.5, L2 55.0/54.9, L3 46.4/47.0, L5 42.6/43.0, L6 40.9/40.9 (Student giai ma L5 52.0). PC L0 74.2/78.2 -> L5 49.3/54.5; Crowd L0 79.4/82.9 -> L5 53.2/58.7, L6 50.2/54.4. L1 (trung binh K) = L0. Chon theo nhom chi >> ca pose (~9 mm); lat chi them ~4 mm (co tay 3DPW 111 -> 59); lat mat-lung chi dang ke o Crowd. Teacher giai ma pose da chon tot hon Student ~9-10 mm o muc oracle; o L0 bang nhau -> Teacher dung duoc lam bo giai ma luc test (khong can GT). CANH BAO: OC L0 pose 45.9 mm thap bat thuong so voi log lifter L2 ep40 (OC 84.4) -> can kiem tra loader OC theo YAML Student. Huong de xuat: Critic chon theo tung chi + Teacher giai ma.

## 32. Loi quy uoc khop trong loss mesh / loss chieu 2D (2026-10-25)
Do bang main/check_root_proj_convention.py (3DPW test, 2000 mau, mesh GT):
- Loss mesh: GT mesh tru pelvis H36M (fusekin_sample.py, PW3D train), pred mesh tru pelvis SMPL (get_coord). Lech pelvis
  SMPL - H36M: TB 38.2 mm (median 36.0, p95 54.6), chu yeu truc y (+26 mm). San loss mesh L1 khi pred = GT hoan hao: 19.1 mm/toa do.
- Loss chieu 2D: GT 2D = chieu H36M-17 (J_h36m @ mesh); pred = SMPL-30 chon theo ten. Lech 3D cung ten TB 68.4 mm:
  Torso 170, R_Hip 152, L_Hip 139, Nose 90, Head_top 83 mm. Khong dao trai/phai (so cheo 242-294 mm); do rong hong
  H36M 292 mm vs SMPL 137 mm, hong SMPL thap hon 11-14 cm -> khac dinh nghia khop.
- Eval (dataset.evaluate) KHONG bi anh huong (tu tru cung goc 2 phia). Chi anh huong luc train (ca Teacher_Trainer va
  Student_Trainer dua mesh_loss + body_joint_proj vao AWL; cfg.LOSS.w_mesh khong tat duoc).
Sua THANG trong code (khong dung co config): loss mesh tru pelvis H36M tu mesh tuyet doi; loss chieu 2D dung H36M-17 tu mesh tuyet doi
smpl_mesh_cam_render_det (pose detach nhu cu). Kiem tra check_mesh_root_fix.py: lech 38.2 -> ~0 mm/dinh.
Checkpoint train TRUOC 2026-10-25 13:30 dung loss cu (eval khong doi). Code: lib/core/base.py (_mesh_for_loss, _proj_for_loss,
_h36m_proj_heatmap), lib/models/teacher_student.py (them output smpl_mesh_cam_render_det). Ban sao: *.bak_rootfix.

## 33. Dọn source + kiểm tra sửa loss gốc mesh (2026-10-25)
- Kiểm tra `main/check_mesh_root_fix.py` (3DPW 2000 mẫu, mesh GT làm pred hoàn hảo): trước sửa L1 19.12 mm/tọa độ, 38.23 mm/đỉnh (p95 54.6); sau sửa (trừ pelvis H36M trên mesh đã trừ gốc) 0.15 / 0.41 mm. Dư 0.4 mm do tổng hàng pelvis H36M = 0.9996 -> code cuối lấy pelvis H36M từ mesh tuyệt đối (khớp tuyệt đối, chờ người dùng chạy lại xác nhận ~0).
- Oracle `oracle_best_hypothesis.py` đo bằng `dataset.evaluate` (người dùng chạy, Student l2last ep27 + Teacher teacher_gt_v2_jotr, K=20): L0 khớp bảng test (3DPW 75.06/47.80/91.49; OC test 76.23/46.45/91.21). Teacher giải mã: 3DPW L0 74.99 -> L2 61.44 -> L3 52.24 -> L5 47.30 -> L6 44.86; REF pose GT 21.92/14.11/34.06. Student giải mã: L5 57.19, REF 42.79. OC test: Teacher L5 48.51, REF 25.94. Kết luận: phương án tốt có sẵn trong K giả thuyết; Teacher tận dụng pose tốt, Student thì không.
- Dọn source theo yêu cầu (chi tiết `01_PROJECT_CONTEXT.md` mục 20). Chưa chạy; cần smoke test trên server trước khi train Teacher lại.


## 34. Teacher GT sau sua loss (teacher_gt_v2_jotr_rootfix, 2026-10-25)
Cung YAML train_teacher_k16_v2_jotr.yml, chi khac loss mesh / chieu 2D. Best epoch 40 (cu: 46), ca hai chon tren 3DPW test.
MPJPE / PA / MPVPE (14 khop, dataset.evaluate), moi / cu:
- 3dpw: 22.43/14.55/32.64 vs 21.92/14.11/34.06
- OC chi test: 26.87/17.64/39.52 vs 25.94/17.77/41.56; OC toan bo: 23.95/17.45/38.06 vs 24.06/17.33/39.27
- PC chi test (281 mau): 21.11/18.93/40.29 vs 20.36/17.93/42.95; PC toan bo: 27.50/20.00/40.89 vs 26.98/18.83/40.89
- Crowd: 27.61/19.26/40.42 vs 27.12/18.07/39.83
Nhan xet: MPVPE giam 1.2-2.7 mm o 3DPW/OC/PC test (Crowd +0.6, PC toan bo 0); MPJPE tang 0.5-0.9, PA tang 0.4-1.2. Chenh lech nho, 1 run moi ben (AWL lech ~1 mm giua seed) -> sua loss KHONG doi dang ke Teacher khi dau vao la GT. Can do Teacher moi lam bo giai ma pose DiffLifter (oracle_best_hypothesis) truoc khi chon Teacher cho Selector.

## 35. Oracle voi Teacher rootfix + Student student_kd_full_3dpw (2026-10-25, nguoi dung chay)
Hypothesis DiffLifter lay tu trong so diff_lifter trong checkpoint Student (khac l2last o muc 33) -> KHONG so truc tiep voi Teacher cu.
MPJPE (dataset.evaluate), Student / Teacher rootfix giai ma:
- 3dpw: REF 38.91/22.43; L0 76.15/78.13; L2 66.61/65.27; L3 61.10/56.88; L5 56.75/50.95; L6 54.42/48.10
- OC test: L0 79.67/82.51; L3 61.71/58.50; L5 57.05/52.65
- PC test (281): L0 65.12/55.72; L5 54.56/40.31
- Crowd: L0 96.62/95.42; L3 77.95/72.27; L5 73.55/66.42
So voi muc 33 (Teacher cu + l2last): Student L2/L3/L5 gan nhu bang (66.6/61.1/56.8 vs 65.7/60.5/57.2) -> chat luong gia thuyet tuong duong; Teacher rootfix L0 78.13 vs 74.99, L5 50.95 vs 47.30 -> Teacher moi giai ma pose khong hoan hao kem hon ~3-4 mm (gia thuyet, chua A/B cung lifter). Can chay A/B: cung Student, doi Teacher cu.
A/B cung Student student_kd_full_3dpw (cung gia thuyet), doi Teacher cu teacher_gt_v2_jotr: Teacher cu / rootfix MPJPE
3dpw L0 77.82/78.13, L3 56.49/56.88, L5 50.59/50.95; OC L5 51.52/52.65; PC L5 39.15/40.31; Crowd L5 66.76/66.42.
-> Hai Teacher NGANG NHAU khi giai ma pose DiffLifter (chenh <= 1.1 mm). Gia thuyet "Teacher moi kem 3-4 mm" o tren SAI:
chenh lech den tu lifter trong checkpoint Student (lifter trong l2last cho Teacher L0 74.99 / L5 47.30, tot hon ~3 mm
du Student giai ma thi gan bang).

## 36. Selector (2026-10-25, da code, chua chay)
Muc tieu (lifter l2last, Teacher giai ma, 3DPW MPJPE14): L0 ~75 -> Oracle cung tap phuong an (L5) ~47-51. Bao % khoang cach
lay lai = (L0 - Selector) / (L0 - Oracle). Chi tiet thiet ke 01_PROJECT_CONTEXT muc 21. Ket qua: chua co.
Ket qua v1 (nguoi dung chay 3 epoch, 6.9 phut/epoch): train nhom L0/Selector/Oracle 77/92/45 mm, top1 5.5 -> 8%;
3dpw-val pose MPJPE17 L0 65.90 / Selector 73.98 -> 71.42 / Oracle 42.19 (lay lai -34% -> -23%). Selector chon TE HON
pose mac dinh: giu L0 la phuong an manh (gia thuyet S5 don le sai nhieu hon), mang khong biet phuong an nao la L0, CE mem
khong phat chon sai. Dung v1. v2 (config/train_selector_l2last_v2.yml): prior giu L0 (embedding + bias logit khoi tao 5),
loss sai so ky vong (cm, w 1) + CE mem w 0.1, log ti le giu L0. Chua chay.
Ket qua v2 (20 epoch, nguoi dung chay): train nhom L0/Selector/Oracle 76.4/66.3/45.3 mm (lay lai ~32% tren train), top1
12.7%, giu L0 54% (than-dau 39%). 3dpw-val pose MPJPE17 L0 65.90 / Selector 65.33 / Oracle 42.19 -> chi lay lai 2.4%;
tren val Selector giu L0 76-89% o chi (khong loi: chan 62/62, tay 82/83, 88/87), than-dau 61 -> 59 (giu L0 46%).
Ket luan: Selector hoc duoc tren phan bo train nhung KHONG chuyen sang 3DPW (khoang cach tong quat hoa lon); prior giu L0
lam no an toan (khong te hon L0) nhung gan nhu khong loi. Chua ro Selector dung anh hay chi dung thong ke pose cua tung bo.
Danh gia test v2 (last, epoch 20, Teacher rootfix; nguoi dung chay). LUU Y: log nap lifter
experiment/finetune_difflifter_r25_cam_gtmask_d512_e40_3dpw/latest_epoch.bin va KHONG co dong nap lifter_from_ckpt ->
KHONG phai lifter l2last; Teacher L0 78.13 trung oracle cua student_kd_full_3dpw (muc 35) -> cung lifter do.
Mesh qua Teacher MPJPE/PA/MPVPE, L0 -> Selector (Oracle): 3dpw 78.13/49.44/93.04 -> 77.60/49.58/92.71 (50.95), lay lai 2.0%;
OC test 82.51 -> 81.05 (52.65) 4.9%; PC test (281) 55.72 -> 57.53 (40.31) -11.8%; Crowd 95.42 -> 95.51 (66.42) -0.3%.
Ket luan: Selector anh SPIN dong bang gan nhu khong loi tren 3DPW (<= 1.5 mm), giu L0 75-90% o chi.

## 37. FuseKin-v3 (2026-10-26, da code, chua chay)
Thiet ke tu doc paper (3DCrowdNet, DPMesh, JOTR, ReFit, PandaPose, SAM 3D Body, ScoreHypo, BlockGCN): [A] backbone fine-tune +
heatmap 2D vao stride 4, [B] feature anh tai khop + [B'] bias hop, [C] phan bo do sau tung khop, [D] tinh chinh lap ReFit +
sai lech 2D, [E] loss do bat dinh + nhieu 2D. Chi tiet 01_PROJECT_CONTEXT muc 22. Moc so sanh: Student scratch v2 (nguoi dung
da co, cung giao thuc). Ket qua: chua co.

## 38. FuseKin-v3 scratch (full A-E) vo train tu epoch 6-7 -> loi [E] + AWL, da sua

- Run `student_v3_scratch_jotr` (bat du A, B, B', C, D, E). 3DPW theo epoch: 157.6 -> 127.1 -> 104.3 -> 94.2 -> 84.7 -> **83.34/53.55/100.21 (ep6)** -> 89.9 (ep7) -> ... -> 175.2/123.3/230.4 (ep11).
- Trieu chung: `smpl3d` am (-0.1 -> -0.86), total loss am toi -7e5; trong so AWL `smpl_joint_cam` (0.5/p^2) 0.6 -> 43 (ep6) -> 6.3e4 (ep7) -> 7.5e5 (ep10).
- Nguyen nhan: [E] dua NLL `|e|/sigma + log sigma` (am khi sigma < 1 m) vao AWL. AWL `0.5/p^2 * L + log(1+p^2)` gia dinh L >= 0; L < 0 -> giam p ve 0 de loss -> -vo cung -> trong so khuech dai grad hang tram nghin lan -> vo mo hinh.
- Sua (lib/core/base.py, `_smpl_hard_losses` + `_apply_awl`): AWL nhan L1 co trong so khop w = exp(-log_sigma) (detach, chuan hoa mean = 1, luon >= 0, cung thang do v2); log_sigma hoc rieng bang NLL tren loi detach (grad chi vao log_sigma), cong ngoai AWL (khoa `joint_sigma_nll`).
- Phai train lai tu dau (khong resume: tham so AWL trong checkpoint da hong). Checkpoint ep6 chi tham khao.
- Ra soat toan bo code v3 sau su co (2026-10-26): [A] heatmap (toa do luoi, zero-init, BN eval, lr x0.1 qua MultiStepLR/warmup), [B] grid_sample (align_corners=False khop chuan hoa x/W*2-1), [B'] attn_mask (B*h, J, J) dung thu tu PyTorch, [C] target/bin, [E] nhieu 2D: khong thay loi.
  Sua them [D]: feedback cua vong tinh chinh dung uoc luong truoc DA DETACH (truoc day comment noi detach nhung grad van chay qua phep chieu chia z va qua rot6d/shape/cam vao uoc luong truoc) + clamp diem chieu [-2, 2]. Khong phai nguyen nhan vo train (do la [E] + AWL) nhung la nguon bat on tiem an.

## 39. FuseKin-v3 scratch (full A-E, da sua [E]+AWL va [D] detach) - KET QUA (2026-10-27, nguoi dung chay)

- Lich lr_step [10, 16, 20], 22 epoch. best.pth.tar = epoch 15 (chon theo 3DPW test).
- 3DPW: **75.53 / 48.31 / 90.22** (MPJPE / PA / MPVPE).
  - vs JOTR 76.4 / 48.7 / 92.6: thang ca 3.
  - vs DPMesh 73.6 / 47.4 / 90.7: MPVPE thang (-0.48), MPJPE thua +1.93, PA thua +0.91.
  - Nguoi dung: tot hon ca Student KD v2.
- Chua co: OC / PC / Crowd (giao thuc JOTR -> --ocpc_mode all), so last.pth.tar, ablation tung module.

## 40. Ke hoach KD cho v3 (2026-10-27, da code, chua chay)

- Buoc 1: Teacher v3 `config/train_teacher_v3_jotr.yml` (--exp_name teacher_v3_jotr): Teacher v2 JOTR (GT 3D + che khop 30% + img_probe) + bo co v3_* giong het Student v3; lich [10, 16, 20] / 22 epoch. Dieu kien di tiep: tot hon Teacher v2 rootfix tren 3DPW.
- Buoc 2: Student v3 KD `config/train_student_v3_kd_jotr.yml` (--exp_name student_v3_kd_jotr): v3 scratch + cong thuc E1 cua v2 (init tu Teacher, KD feat joint + global, kd_hard_w 1.0, kd_lambda 1.0, w_proj 0). So voi v3 scratch 75.53 / 48.31 / 90.22.
- Code moi:
  - base.Student_Trainer: khi bat [A] (v3_bb_finetune / v3_bb_heatmap) va init_student_from_teacher -> copy CA backbone (+ heat_fuse) tu Teacher (truoc chi copy smpl_model -> smpl_model quen feature backbone da fine-tune nhung chay tren SPIN goc).
  - cfg.TEST.save_best_per_set (mac dinh False): moi epoch luu them checkpoint/best_<tap>.pth.tar (chi trong so) theo MPJPE tung tap trong test_list; lich su o test_log['per_set'] (giu qua resume). Bat trong yml Student KD voi test_list 4 tap (OC / PC toan bo = giao thuc JOTR).
- Buoc 3 (sau khi co so buoc 2): ablation init-only (kd_lambda 0); thu kd_joint_relation / KD do sau [C] neu con cach DPMesh.
- KET QUA Teacher v3 (teacher_v3_jotr, nguoi dung chay 2026-10-28): 3DPW **25.70 / 16.99 / 36.41** (input GT 3D, best chon theo 3DPW test).
  vs Teacher v2 rootfix 22.43 / 14.55 / 32.64 (50 epoch) -> v3 KEM hon 3.3 / 2.4 / 3.8 mm -> KHONG dat dieu kien da dat.
  Yeu to lan: lich 22 vs 50 epoch. Gia thuyet (chua kiem): [D] keo phep chieu ve 2D Halpe (nhieu hon GT 3D) va cac nhanh 2D ([A] heatmap, [B]) lam Teacher dua vao 2D detector thay vi GT 3D.
  Quyet dinh de xuat: van chay buoc 2 (Student KD), vi tieu chi that la Student so voi v3 scratch 75.53; chenh Teacher 3 mm nho so voi khoang Teacher-Student ~50 mm.
- Student v3 KD (student_v3_kd_jotr), log den epoch 13 (nguoi dung gui 2026-10-28; OC / PC toan bo):
  3DPW MPJPE theo epoch 77.98, 77.47, 78.55, 76.97, **75.98 (ep5)**, 77.43, 78.31, 77.31, 76.31, 77.05, 76.48, 76.02, 77.21 -> phang tu ep1 (init tu Teacher), dao dong +-1 mm.
  Best tung tap den ep13: 3DPW 75.98 (ep5) | OC 62.49 (ep11) | PC 90.48 (ep4) | Crowd 95.40 (ep2).
  So DPMesh (73.6 / 70.9 / 82.2 / 79.9): 3DPW +2.4, OC -8.4, PC +8.3, Crowd +15.5 -> yeu o nguoi-che-nguoi.
  So v3 scratch 3DPW 75.53: KD chua hon (chenh < dao dong giua epoch). Con 2 moc giam lr (16, 20).
  KD feat: J ~0.04, G ~0.024 nho tu dau (Student khoi tao tu Teacher); tong loss train van giam trong khi 3DPW phang.

## 41. Chan doan 2D tren PC / Crowd (2026-10-28, da code, chua chay)
- `main/diag_2d_crowd.py`: D1 loi 2D Halpe vs GT 2D (px crop 256, % khop xa > 40 px, % xa ma conf >= 0.5 = nghi nham nguoi, % bo khop);
  D2 Student voi 2D halpe / fixfar (chi thay khop xa bang GT) / gt (toan bo GT 2D) -> lift17 + mesh dataset.evaluate, tach mau xau / sach. In n tung tap (doi chieu dinh nghia OC).
- Doc ket qua: gt giam PC / Crowd ve ~ DPMesh -> 2D la nut that; fixfar lay lai phan lon -> loi chinh la khop sai tho (nham nguoi) -> giai phap A (nhieu "doi nguoi" luc train).
- Luu y da phat hien: 3dpw-crowd = 3DPW VALIDATION (courtyard_hug_00, courtyard_dancing_00) -> chon checkpoint theo 3dpw-val se ro ri Crowd; tap chon sach = val bo 2 seq nay.
- KET QUA diag_2d_crowd (v3 scratch best, nguoi dung chay 2026-10-28), mesh MPJPE14 halpe -> fixfar -> gt | lift17 halpe -> gt | % khop xa tu tin:
  3dpw n=35515: 75.53 -> 74.63 -> 84.15 | 71.69 -> 89.10 | 0.3% ; OC n=20243: 63.24 -> 62.97 -> 74.65 | 41.99 -> 76.97 | 0.3%
  PC n=2218: **93.04 -> 80.75** -> 86.81 | 74.02 -> 88.46 | 3.0% ; Crowd n=1923: **97.54 -> 83.55** -> 84.58 | 82.67 -> 88.82 | 3.5%
  Doc: (1) bien the gt KHONG hop le (GT 2D chieu tu mesh lech dinh nghia khop Halpe ma lifter hoc -> te hon ca Halpe o 3DPW / OC).
  (2) fixfar: chi sua ~3% khop sai tho -> PC -12.3, Crowd -14.0 mm (PC 80.75 < DPMesh 82.2; Crowd 83.55 vs 79.9) -> khoang cach PC / Crowd gan nhu toan bo do it khop sai tho (nham nguoi), khong phai nhieu nho.
  (3) Lifter tren OC chi 42 mm (3DPW 72) -> tap OC de ve pose voi 2D Halpe; giai thich mot phan OC 63.
  Them bien the dropfar (bo khop xa, mask 0) de do loi ich neu chi PHAT HIEN duoc khop sai.
- KET QUA Student v3 KD (student_v3_kd_jotr, 22 epoch, nguoi dung 2026-10-28): best (chon theo 3DPW test) 3DPW **74.62 / 47.64 / 89.80**.
  vs v3 scratch 75.53 / 48.31 / 90.22: -0.91 / -0.67 / -0.42 (1 seed; dao dong giua epoch ~ +-1 mm -> loi ich KD nho, chua chac).
  vs DPMesh 73.6 / 47.4 / 90.7: MPJPE +1.02, PA +0.24, MPVPE -0.90 (thang). vs InstanceHMR 73.2 / 44.3 / 80.3: con thua.
- KET QUA dropfar (v3 scratch, PC / Crowd): mesh MPJPE halpe / dropfar / fixfar / gt = PC 93.04 / 89.47 / 80.75 / 86.81 ; Crowd 97.54 / 93.64 / 83.55 / 84.58.
  Mau xau (>=1 khop xa tu tin) = 9.0% PC / 10.2% Crowd: MPJPE ~261 mm (dropfar ~222, fixfar ~113, gt ~83); mau sach 76.4 / 79.0 (~ muc 3DPW).
  Khop xa nhieu nhat: L_Wrist 13-15% (lech trai/phai ro: R_Wrist 5-6%).
  Doc: (1) khoang cach PC / Crowd = ~10% mau hong nang (~260 mm), con lai ngang 3DPW. (2) chi BO khop xa lay lai it (-3.6 / -3.9) -> can toa do dung hoac mo hinh chiu duoc 2D nham nguoi.
  (3) mesh MPJPE (93) >> lifter (74) tren PC, PA 58.6 -> mo hinh KHUECH DAI loi o mau xau (nghi xoay toan than; nghi [D] keo theo phan du 2D sai). Them vao script: halpe_init (mesh truoc [D]) + lift17 tach xau / sach de kiem.
- KET QUA halpe_init + lift tach xau / sach (v3 scratch): mau xau lift17 221 (PC) / 222 (Crowd) vs sach 59 / 67; mesh truoc [D] 263.8 / 263.1 vs sau 260.9 / 260.4 -> [D] KHONG khuech dai (gia thuyet cu SAI); loi sinh tu DiffLifter khi 2D nham nguoi, mesh chep lai.
- Them D3 vao diag_2d_crowd.py: crop that so voi crop "ly tuong" (cung quy tac, tinh tu GT 2D) -> IoU / lech tam / ti le; nhom mat >= 8 khop 2D, crop lech (IoU < 0.5), nham nguoi; "gop vao TB" (mm); --vis_k hinh toan anh mau te nhat (crop, Halpe, GT); --variants halpe de chay nhanh.
- Ghi chu crop: test 3DPW cat theo bbox tu khop Halpe (x1.2, vuong, x1.1; < 2 khop -> bbox RootNet) = giao thuc 3DCrowdNet / JOTR; train cat theo bbox GT -> lech train / test.
- KET QUA D3 (v3 scratch, --variants halpe): crop KHONG phai van de (IoU TB 0.84-0.88; nhom crop lech / mat >= 8 khop <= 0.5%, gop <= 0.1 mm).
  Nham nguoi gop vao TB: PC 16.64 mm (9.0% mau, MPJPE 260.9), Crowd 18.59 (10.2%, 260.4), 3DPW 1.17 (3.2%, 111.3), OC 0.89 (2.7%, 95.0).
  [D] o OC lam TE hon: truoc [D] 61.84 / 42.39 / 77.05 vs sau 63.24 / 43.20 / 78.27 (3DPW / PC / Crowd [D] giup 0.7 / 1.2 / 1.75).
  OC n = 20243 (doi chieu voi JOTR / DPMesh).

## 42. Giai phap A: DiffLifter voi nhieu "bat nham nguoi" (2026-10-28, da code, chua chay)
- Phat hien: lifter hien tai (gtmask_d512_e40) train voi DATASET.halpe_gt_mask_thr 0.1 -> moi khop Halpe lech GT > 0.1 canh crop bi MASK tren tap train -> lifter KHONG BAO GIO thay khop sai voi conf cao khi train (occ_corrupt co sai cho nhung HA conf ve gan nguong) -> test gap khop nham nguoi conf cao thi tin theo.
- Code: main/finetune_difflifter.py wrong_person_corrupt (cfg.DIFFLIFT.wp_aug_prob, wp_shift_min / max, wp_arm_prob; mac dinh tat): 1 doan dau chi (co tay | khuyu+co tay | co chan | goi+co chan, tay 70%) doi 40-120 px huong ngau nhien, GIU conf.
- YAML config/finetune_difflifter_d512_e40_wp.yaml = gtmask_d512_e40 + wp_aug_prob 0.2 (1 yeu to). save_dir experiment/finetune_difflifter_d512_e40_wp.
- diag_2d_crowd.py --lifter_ckpt: thay lifter trong Student de do lift17 (xau / sach) cua lifter moi.
- Lua chon tiep (chua lam): tat halpe_gt_mask_thr (de loi that cua Halpe o MuCo / CrowdPose vao train); them nhieu nham nguoi cho Student.
- KET QUA lifter wp (finetune_difflifter_d512_e40_wp, 40 epoch, final test = best_epoch theo PA 3dpw-val, nguoi dung 2026-10-29), MPJPE17 / PA:
  3dpw 68.33 / 45.03 | H36M 53.84 / 41.59 | OC 47.28 / 32.61 | PC 76.05 / 50.49 | Crowd 79.01 / 49.37.
  Cung giao thuc lifter cu gtmask_d512_e40 (best ep11, muc 482): 3dpw 73.11 / 49.24 | H36M 55.45 / 42.86 | OC 80.44 / 52.86 | PC 95.94 / 60.28 | Crowd 99.94 / 55.88
  -> -4.8 / -1.6 / -33.2 / -19.9 / -20.9 mm MPJPE. OC -33 KHONG hop ly voi 1 thay doi nham nguoi (OC chi 2.7% mau nham nguoi) va mau thuan diag_2d_crowd
  (lifter cu ep40 trong Student: OC lift17 42 vs eval script ep40 84.4) -> OC trong eval finetune_difflifter va trong diag do KHAC nhau; can do ca hai lifter bang CUNG diag_2d_crowd --lifter_ckpt truoc khi ket luan.
- SO SANH CUNG diag_2d_crowd (Student v3 scratch, thay lifter; lift17 toan tap | xau / sach):
  lifter cu (gtmask latest ep40, = lifter trong Student): 3DPW 71.69 | 100.15 / 70.76 ; OC 41.99 | 62.79 / 41.41 ; PC 74.02 | 220.99 / 59.45 ; Crowd 82.67 | 222.00 / 66.77
  lifter wp (best_epoch):                                 3DPW 68.30 |  99.89 / 67.26 ; OC 47.25 | 73.34 / 46.52 ; PC 76.04 | 222.47 / 61.53 ; Crowd 78.99 | 223.09 / 62.54
  (diag cua lifter wp trung final test cua finetune_difflifter: 68.33 / 47.28 / 76.05 / 79.01 -> 2 script do giong nhau; bang "lifter cu best ep11" o tren la checkpoint KHAC lifter trong Student.)
  KET LUAN: nhieu wp KHONG sua mau nham nguoi (PC / Crowd xau 221 -> 222, 222 -> 223). Thay doi con lai (3DPW -3.4, OC +5.3, Crowd sach -4.2) co the do khac epoch (best vs latest ep40), khong do wp.
  Giai thich: tu 2D don thuan, 1 doan chi bi doi (xuong van hop ly) KHONG phan biet duoc voi tu the that -> lifter khong the tu phat hien; o mau xau that nhieu khop cung lech (fixfar con 113 vs gt 83). Can bang chung tu ANH (Student) hoac 2D tot hon.

## 43. Student v3 scratch voi du lieu FULL (2026-10-29, da code, chua chay)
- Nguoi dung chot: bo huong lifter wp, giu lifter cu (gtmask latest ep40); thu luong du lieu.
- Code: cfg.TRAIN.mix_mode ('ratio' mac dinh = cu | 'full' = moi epoch duyet het hop 4 bo, moi mau 1 lan, shuffle | 'jotr' = nhom 3D (H36M, MuCo) / 2D (COCO, CrowdPose) 50/50 kieu 3DCrowdNet, trong nhom ti le theo kich thuoc). lib/core/base.get_dataloader in so mau tung bo + iter / epoch.
- YAML config/train_student_v3_scratch_jotr_full.yml = v3 scratch + mix_mode 'full'; end_epoch 12, lr_step [6, 9, 11] TAM (chinh theo do dai epoch thuc).
- So voi v3 scratch 75.53 / 48.31 / 90.22. Yeu to lan: ti le bo theo kich thuoc (khong con 25%), tong luot mau, lich lr.
- 2026-10-29: main/compare_2d_sources.py (CPU, khong model): so Halpe-26 / OpenPose (JOTR dung cho 3DPW / OC / PC) / Higher-HRNet (JOTR dung cho Crowd) voi GT 2D tren 12 khop chi; % khop xa, % bo, % mau Halpe xau ma nguon khac tot ("cuu duoc"); hinh so sanh. Muc dich: xem detector khac co bat dung mau kho khong.
- 2026-10-29 compare_2d_sources (hinh nguoi dung gui, 3DPW-Crowd): mau Halpe xau = Halpe bat CA NGUOI KHAC (12/12, 9/12 khop), OpenPose va Higher-HRNet bam DUNG nguoi. Nguyen nhan: tools/extract_halpe26_3dpw.py chay RTMPose top-down tren bbox GT RONG -> khi 2 nguoi chong nhau RTMPose bam nguoi ro hon.
- Sua (chua chay): extract_halpe26_3dpw.py --bbox_source gt | openpose | openpose_best (bbox tu khop OpenPose cua dung nguoi; best = chon trong {OP x1.25, OP x1.1, GT} ket qua khop OpenPose nhat); ra file 3DPW_{split}_halpe26_opbox.json (giu file cu); thu nghiem --seqs / --max_images / --dry_run / --vis_compare K (ve Halpe cu | moi | OpenPose cho nguoi Halpe cu lech OpenPose > 50%); in % nghi nham nguoi cu -> moi, sua duoc / lam hong.
  Dataset doc file moi qua cfg.DATASET.pw3d_halpe_suffix ('' mac dinh | '_opbox'), chi tap khong phai train. Ban sao script cu: tools/extract_halpe26_3dpw.py.bak_opbox.
