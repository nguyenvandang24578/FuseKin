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