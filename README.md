# FuseKin

Dự án nghiên cứu khôi phục mesh người từ một ảnh: kết hợp đặc trưng ảnh với pose 3D được nâng từ keypoint 2D. Hướng đang phát triển là **DiffLifter đa giả thuyết + Student v2**.

Cập nhật: **2026-10-23**, đối chiếu với source và cấu hình local. Trạng thái train trên server lấy từ thông tin người dùng cung cấp.

## Tài liệu

| File | Đọc khi cần |
|---|---|
| [01_PROJECT_CONTEXT.md](01_PROJECT_CONTEXT.md) | Kiến trúc, dữ liệu, quy ước, cấu hình và giới hạn hiện tại |
| [experiment.md](experiment.md) | Kết quả, trạng thái thí nghiệm và bước tiếp theo |
| [AGENTS.md](AGENTS.md) | Quy tắc làm việc cho AI/cộng tác viên |

Mỗi nội dung có một nơi chính để cập nhật. README của các repo tham khảo trong thư mục con được giữ riêng.

## Pipeline hiện tại

1. RTMPose cung cấp Halpe-26; adapter đổi về 17 khớp H36M.
2. DiffLifter nhận 2D, confidence và thông tin bbox/camera, sinh pose 3D hoặc nhiều giả thuyết.
3. ResNet-50 pretrained SPIN cung cấp đặc trưng ảnh; backbone và lifter được đóng băng khi train Student.
4. Student v2 kết hợp token ảnh và token khớp, dự đoán pose/shape/camera rồi dựng mesh SMPL.

Teacher/KD, MotionBERT và PMCE vẫn có trong source để đối chiếu. Chúng không phải hướng thí nghiệm chính của đợt hiện tại.

## Trạng thái hiện tại

- Augmentation: người dùng báo không hiệu quả. Spread: best trên 3DPW được báo là MPJPE 77.59 / MPVPE 94.17 mm; người dùng đã gửi log chẩn đoán 35,515 mẫu: real/off/shuffle gần nhau, chưa thấy lợi ích rõ của spread lúc inference (mục 9 của `experiment.md`).
- GCN + attention song song: đã đối chiếu best epoch 6 với attention baseline tại S=1/K=1; chỉ H36M giảm lỗi, 3DPW/OC/PC tăng. Giữ `r25_cam` làm mốc chính; bảng số ở mục 5 của `experiment.md`.
- Prior nhiễu có điều kiện từ SimCC/GMM: tạm hoãn. Code probe SimCC được giữ để nghiên cứu sau.
- Các YAML Student diffusion đang dùng (`diff`, `diff_aug`, `diff_sel` và hai bản gate KD) hiện chọn best trên **3DPW test**, dù một số comment còn ghi val. Lifter chọn best theo `3dpw-val`; chi tiết trong [bối cảnh dự án](01_PROJECT_CONTEXT.md#6-giao-thức-và-chọn-checkpoint).

## Cấu trúc chính

| Đường dẫn | Vai trò |
|---|---|
| `lib/models/diff_lifter.py` | Diffusion 2D→3D và denoiser |
| `lib/models/ARTS.py` | Wrapper backbone/lifter/Teacher/Student |
| `lib/models/teacher_student.py` | Fusion và các head SMPL |
| `lib/models/hyp_selector.py` | Chọn/gộp giả thuyết theo từng khớp bằng ảnh |
| `lib/core/config.py`, `lib/core/base.py` | Config, trainer và evaluator |
| `lib/utils/jotr_dataset.py`, `lib/utils/lift_pseudo3d.py` | Adapter H36M-17 và pseudo-GT cho lifter |
| `data_final/` | Dataset loader và dữ liệu theo đường dẫn cấu hình |
| `config/`, `main/`, `tools/` | YAML thí nghiệm, train/eval, trích keypoint/chẩn đoán |

## Lệnh chạy trên server

Chạy từ thư mục gốc repo, trong môi trường đã chuẩn bị dữ liệu, SMPL và checkpoint. Không đổi tên `data_final` thành `data` theo README ARTS cũ. Kiểm tra đường dẫn thực trong YAML/loader; dữ liệu lifter MSCOCO/CrowdPose cần cả Halpe và NeuralAnnot.

Train **lifter GCN + attention mới**, khởi tạo từ đầu:

```bash
python main/finetune_difflifter.py --cfg config/finetune_difflifter_r25_cam_gcn_attn.yaml --gpu 0
```

Đánh giá lifter tốt nhất:

```bash
python main/finetune_difflifter.py --cfg config/finetune_difflifter_r25_cam_gcn_attn.yaml --gpu 0 --eval_only --ckpt experiment/finetune_difflifter_r25_cam_gcn_attn/best_epoch.bin
```

Phân tích lỗi và trực quan hóa lifter:

```bash
python main/inspect_difflifter.py --cfg config/finetune_difflifter_r25_cam_gcn_attn.yaml --gpu 0 --ckpt experiment/finetune_difflifter_r25_cam_gcn_attn/best_epoch.bin
```

Ví dụ lệnh đã dùng cho bản Student augmentation (người dùng hiện báo không hiệu quả):

```bash
python main/train.py --cfg config/train_student_v2_diff_aug.yml --exp_name student_v2_diff_aug --gpu 0
```

Đánh giá mesh của run có tên như lệnh trên:

```bash
python main/test_ckpt.py --cfg config/train_student_v2_diff_aug.yml --ckpt experiment/student_v2_diff_aug/checkpoint/best.pth.tar --test_list 3dpw --gpu 0
```

`--exp_name` phải là tên mới khi tạo run mới; dùng đúng tên và `--resume_training` khi tiếp tục run đã có. Đường dẫn checkpoint thực phụ thuộc tên run. Không dùng số MPJPE 17 khớp của lifter để so trực tiếp với số 14 khớp của mesh.

## Nguồn gốc và ghi nhận

FuseKin phát triển từ ARTS, sử dụng/tham khảo MotionBERT, PMCE, D3DP và HyperDiff. Denoiser mới lấy ý tưởng hai nhánh song song từ [FMPose3D](https://github.com/AdaptiveMotorControlLab/FMPose3D); phần sinh pose vẫn là diffusion, chưa chuyển sang flow matching.

README này mô tả FuseKin hiện tại, không phải bản phát hành chính thức của ARTS. Khi công bố kết quả, ghi nhận các thành phần kế thừa và tuân thủ giấy phép tương ứng. Công trình ARTS gốc:

```bibtex
@inproceedings{tang2024arts,
  title={ARTS: Semi-Analytical Regressor using Disentangled Skeletal Representations for Human Mesh Recovery from Videos},
  author={Tang, Tao and Liu, Hong and You, Yingxuan and Wang, Ti and Li, Wenhao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={1514--1523},
  year={2024}
}
```

## Student DiffLifter thuần và KD

Hai cấu hình không dùng spread/gate:

- `config/train_student_v2_diff.yml`: Student scratch, không Teacher/KD.
- `config/train_student_kd_e1_v2_diff.yml`: KD feature E1, khởi tạo phần chung của Student từ Teacher.

Cả hai dùng pose DiffLifter đóng băng S1/K1 kết hợp đặc trưng ảnh. Nhánh cộng embedding spread và hai YAML `train_student_v2_diff_spread.yml`, `train_student_kd_e1_v2_diff_spread.yml` đã xóa theo yêu cầu người dùng. Lịch sử kết quả giữ trong `experiment.md`; checkpoint nhánh cũ cần source/YAML tương ứng của run cũ, loader hiện tại sẽ từ chối thay vì bỏ qua nhánh.

## Train gate + KD

Gate nhận token joint, thông tin ảnh vừa đọc và `log1p(std_xyz*10)`; `g=2*sigmoid(MLP)` trong (0,2), khởi tạo g=1. Cập nhật: `coef1*joint + coef2*(g*image_update)`. Teacher GT và loss KD E1 joint/global giữ nguyên.

Đồng bộ source mới lên server, chạy từ gốc repo. Đây là run mới từ khởi tạo Teacher như E1, không resume checkpoint spread cộng token cũ:

```bash
python main/train.py --cfg config/train_student_kd_e1_v2_diff_spread_gate.yml --exp_name student_kd_e1_v2_diff_spread_gate --seed 123 --gpu 0
```

Đối chứng cùng gate, chỉ đặt ba giá trị std ở đầu vào gate bằng 0:

```bash
python main/train.py --cfg config/train_student_kd_e1_v2_diff_spread_gate_zero.yml --exp_name student_kd_e1_v2_diff_spread_gate_zero --seed 123 --gpu 0
```

Hai YAML chỉ khác `MODEL.difflift_spread_gate_zero`; vẫn sinh cùng S5/K20 và dùng pose chính S1/K1. Cùng Teacher, khởi tạo, loss, lịch LR `[5,10,15,22,27]`, 30 epoch. Tên YAML giữ nguyên để dùng lệnh đã gửi. `difflift_spread` cùng steps/k hiện chỉ cấu hình std cho gate; bật spread mà không bật gate là lỗi, không còn nhánh cộng embedding.

Log có `g: mean+/-std` theo batch và `Spread gate: mean/std/min/max` theo epoch, kèm W&B. Checkpoint lưu `spread_gate_config`; loader kiểm tra real/zero/hidden khi resume/eval. Gate thay đổi chưa chứng minh std hữu ích: so real/zero cùng seed để tách lợi ích của std khỏi gate dựa vào joint/ảnh. Nếu chênh nhỏ cần thêm seed. Các run vẫn chọn best trên **3DPW test**, chỉ phục vụ phát triển.

## Kiểm tra gate trên server

Giữ file inspector hiện có `main/inspect_student_spread.py`; mặc định là `--mode gate`, `--mode spread` chỉ là alias cho gate để giữ lệnh đã gửi. Đã bỏ code replay nhánh cộng embedding.

```bash
python main/inspect_student_spread.py --mode gate --cfg config/train_student_kd_e1_v2_diff_spread_gate.yml --ckpt experiment/student_kd_e1_v2_diff_spread_gate/checkpoint/best.pth.tar --set 3dpw --gpu 0 --batch_size 32 --num_vis 4
```

`real`: gate đã học; `off`: ép mọi gate=1; `shuffle`: đổi std giữa các mẫu trong batch trước gate. Giữ ảnh/pose/PE2D; kiểm tra replay khớp wrapper. Real-off đo tác động gate, real-shuffle kiểm tra std đúng mẫu; shuffle cục bộ có thể yếu vì các frame gần nhau. Bản zero có thể khác off vì vẫn học từ joint/ảnh, nhưng không phụ thuộc std thật khi shuffle.

Mặc định đánh giá toàn split, lưu `inspect_gate_3dpw_<timestamp>/` cạnh checkpoint: CSV mẫu/khớp, nhóm spread, `summary.json`, log, biểu đồ và hình `vis/`. Có gate_mean/std/min/max từng mẫu và gate_mean từng khớp; không còn cột embedding spread. Metric mesh dùng evaluator 14 khớp; chẩn đoán lifter dùng H36M-17 bỏ pelvis. Hình nhìn trước/ngang không phải overlay camera.

Thêm `--max_batches 10 --num_vis 0` để kiểm tra trên tập con trước. Giữ cùng batch và một GPU vì wrapper reset seed DDIM theo batch. Không dùng nguyên trạng nhánh Student của `content_ablation_test.py` hoặc `kd_res_mode=v2` cho gate; chúng chưa truyền std và sẽ báo lỗi.

## Kiểm tra Teacher GT

Dùng YAML/checkpoint Teacher riêng; có thể đọc CSV lịch sử để xác minh đúng ảnh/ann_id mà không nạp lại Student cũ:

```bash
python main/inspect_student_spread.py --mode teacher \
  --cfg config/train_teacher_k16_v2.yml \
  --ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar \
  --set 3dpw --sample_ids 2749 8785 17500 \
  --gpu 0 --batch_size 32 --num_vis 4
```

Thêm `--spread_csv <samples.csv_da_luu>` khi cần đối chiếu; CSV cũ vẫn đọc được. Bỏ `--sample_ids` để đánh giá toàn split. Teacher nhận GT mesh -> H36M-17 -> trừ pelvis -> mét, không corruption khi eval; PE2D vẫn từ detector theo YAML. Đây là phép đo oracle. Xuất metric, CSV và hình Teacher/GT; checkpoint nạp strict.

## So scratch/KD trên cùng mẫu

`--mode compare` hỗ trợ Student DiffLifter thuần hoặc gate; `--cfg/--ckpt` là KD, `--baseline_*` là scratch không KD, `--teacher_*` là Teacher GT. Dùng đúng YAML/checkpoint đã train, ví dụ cho hai bản không gate:

```bash
python main/inspect_student_spread.py --mode compare \
  --cfg config/train_student_kd_e1_v2_diff.yml \
  --ckpt experiment/student_kd_e1_v2_diff/checkpoint/best.pth.tar \
  --baseline_cfg config/train_student_v2_diff.yml \
  --baseline_ckpt experiment/student_v2_diff/checkpoint/best.pth.tar \
  --teacher_cfg config/train_teacher_k16_v2.yml \
  --teacher_ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar \
  --set 3dpw --gpu 0 --batch_size 32 --workers 0 --num_vis 4
```

Chỉ dùng ví dụ khi các checkpoint tương ứng đã tồn tại; không coi YAML là kết quả đã train. Có thể thay KD bằng YAML/checkpoint gate, nhưng khi đó so hai run khác cả cơ chế fusion, không tách riêng hiệu quả KD/std.

Luôn kiểm tra pose S1 hai Student khớp trong 1e-6 mét. Chỉ so std khi cả hai bật gate; nếu không, summary ghi `std_compared=false`, `std_max_abs_m=null`, không giả vờ std=0 là bất định đo được. Khi có std, CSV ghi rõ `spread_source`; không có gate ở cả hai thì không thêm cột spread. Teacher input GT phải khớp.

Đầu ra `compare_scratch_kd_<split>_<timestamp>/`: log, summary, CSV mẫu/nhóm, YAML sao lưu, biểu đồ và hình GT/scratch/KD/Teacher. Gain = scratch - KD, dương là KD tốt hơn. Q1-Q4/top10% theo lỗi scratch có thiên lệch chọn nhóm; cần đối chiếu thêm nhóm theo lỗi S1 chung. `--sample_ids` chỉ thêm hình, không giới hạn tập eval; `--expected_epoch N` kiểm tra epoch checkpoint thật. Replay hình dùng nguyên batch gốc và kiểm tra metric lệch không quá 0.01 mm. Giới hạn khởi tạo/hard loss/lịch LR giữa các run vẫn cần xét khi diễn giải gain.

Đã đọc lại source/diff và kiểm tra cấu hình bằng văn bản. **Chưa chạy/import/kiểm cú pháp Python, train hoặc test tại local.**
