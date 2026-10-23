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
- Các YAML Student diffusion đang dùng (`diff`, `diff_aug`, `diff_sel`, `diff_spread`) hiện chọn best trên **3DPW test**, dù một số comment còn ghi val. Lifter chọn best theo `3dpw-val`; chi tiết trong [bối cảnh dự án](01_PROJECT_CONTEXT.md#6-giao-thức-và-chọn-checkpoint).

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

## Kiểm tra Student spread trên server

Dùng đúng YAML train và `checkpoint/best.pth.tar` của run spread. Ví dụ dưới đây dùng tên run `student_v2_diff_spread`; thay đường dẫn nếu run thực có tên khác. Không cần train lại.

```bash
python main/inspect_student_spread.py \
  --cfg config/train_student_v2_diff_spread.yml \
  --ckpt experiment/student_v2_diff_spread/checkpoint/best.pth.tar \
  --gpu 0 --set 3dpw --batch_size 32 --num_vis 4
```

Mặc định chạy toàn tập, tạo thư mục kết quả mới `inspect_spread_3dpw_<timestamp>` cạnh checkpoint. So real/off/shuffle trên cùng checkpoint, in MPJPE/PA-MPJPE/MPVPE mesh, tỷ lệ tốt/xấu và nhóm spread thấp/cao. Xuất CSV từng mẫu/khớp, `summary.json`, `inspect_log.txt`, `overview.png` và hình trong `vis/`. Hình có crop/keypoint và pose/mesh nhìn trước/ngang, không phải overlay camera.

Thêm `--max_batches 10` để kiểm tra nhanh trên tập con; thêm `--num_vis 0` nếu chỉ cần bảng/CSV. Dùng `--out_dir <thu_muc_moi>` nếu muốn chọn nơi lưu; script từ chối thư mục không rỗng. Giữ cùng batch size và một GPU để đối chiếu vì wrapper đặt lại seed DDIM theo batch. Đã có log chạy toàn bộ 35,515 mẫu do người dùng gửi; bảng số, cách diễn giải và giới hạn tại mục 9 của `experiment.md`.

## Kiểm tra Teacher GT trên các mẫu khó của spread

Cùng script trên hỗ trợ `--mode teacher`, mặc định vẫn là chế độ spread. Dùng YAML **Teacher đã train**, không dùng YAML Student KD. YAML KD hiện trỏ `MODEL.TEACHER` tới `experiment/teacher_gt_v2/checkpoint/best.pth.tar`; nếu checkpoint thực nằm nơi khác thì thay đường dẫn tương ứng và dùng đúng YAML của run đó.

Ví dụ kiểm tra ba mẫu đã xem (#2749 guitar, #8785 chạy, #17500 trong xe), đối chiếu đúng ảnh/ann_id với CSV spread người dùng đã tạo:

```bash
python main/inspect_student_spread.py --mode teacher \
  --cfg config/train_teacher_k16_v2.yml \
  --ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar \
  --set 3dpw --sample_ids 2749 8785 17500 \
  --spread_csv experiment/student_v2_diff_spread/checkpoint/inspect_spread_3dpw_20261023_165208_598609/samples.csv \
  --gpu 0 --batch_size 32 --num_vis 4
```

Teacher nhận **GT mesh → H36M-17 → trừ pelvis → mét**, không qua lifter. PE2D vẫn dùng detector/mask như YAML train; không thay bằng GT 2D. Input thực vào Teacher phải trùng GT (script kiểm tra), không áp dụng corruption lúc eval. Checkpoint nạp strict, không bỏ key thiếu/thừa để nạp nhầm Student/T-hint.

Đây là phép đo **ba mẫu**, không phải kết quả toàn tập. Để đo toàn 3DPW, bỏ `--sample_ids 2749 8785 17500`; có thể giữ CSV để đối chiếu từng mẫu. Mỗi mẫu chọn theo index dataset 0-based, không phải ann_id; có CSV sẽ xác minh sequence/frame và ann_id trước khi so. Nếu chỉ số trỏ sai ảnh do đổi dataset, script báo lỗi.

Đầu ra mới `inspect_teacher_gt_3dpw_<timestamp>` cạnh checkpoint gồm `inspect_log.txt`, `summary.json`, `samples.csv`, `joints.csv`, `vis/*_teacher_gt.png`. Hình có crop/keypoint detector, pose input GT và khớp hồi quy từ mesh Teacher, mesh Teacher/GT nhìn trước-ngang, lỗi từng khớp. Khi có CSV spread, in thêm metric Student cũ đã khớp mẫu (không chạy lại Student). So Teacher GT với Student là so khác checkpoint/đầu vào, không tự chứng minh KD có lợi. Đã nhận ảnh kết quả Teacher mẫu #17500; chưa có log tổng kết. Chi tiết ở mục 10 của `experiment.md`.

## Train Student spread + KD feature E1

Đã tạo `config/train_student_kd_e1_v2_diff_spread.yml`: giữ nguyên KD feature E1 và bật spread S=5/K=20; pose chính vẫn S=1/K=1. Đây là bản đối chứng KD hiện có, chưa có trọng số ưu tiên ca khó.

```bash
python main/train.py --cfg config/train_student_kd_e1_v2_diff_spread.yml --exp_name student_kd_e1_v2_diff_spread --seed 123 --gpu 0
```

Chạy từ gốc repo trên server, với lifter `experiment/finetune_difflifter_r25_cam/best_epoch.bin` và Teacher `experiment/teacher_gt_v2/checkpoint/best.pth.tar` đúng kiến trúc. Run mới khởi tạo phần `smpl_model` từ Teacher như E1, không tiếp tục checkpoint Student spread cũ. Kết quả sẽ nằm trong `experiment/student_kd_e1_v2_diff_spread/`; không tạo trước thư mục kết quả vì trainer quản lý run mới. YAML vẫn chọn best trên 3DPW test để giữ giao thức đối chứng hiện có; đây là thí nghiệm phát triển. Mới chuẩn bị cấu hình, chưa chạy xác nhận.
