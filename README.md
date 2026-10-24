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

- Gate/KD mới (mốc người dùng báo): gate std thật **77.47/94.06**, gate_zero **76.11/93.84**, KD full dataset **76.17/93.08 mm** (MPJPE14/MPVPE). Std thật chưa cho lợi ích trong cặp gate này; full dataset khác dữ liệu/Teacher nên không phải ablation gate. Chưa xác minh epoch/best checkpoint; chi tiết và đối chứng init-only full ở mục 19 của `experiment.md`.
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

## Kiểm tra lỗi pose DiffLifter đầu vào trên train/test

Đã mở rộng `main/eval_difflifter_3dpw.py`; không cần checkpoint Student/Teacher. Dùng checkpoint lifter `best_epoch.bin` và đúng YAML Student để giữ nguồn 2D, confidence, augmentation, camera và GT đang dùng cho run. Không chạy ResNet/fusion/gate/KD/mesh head; loader Student vẫn đọc ảnh/dựng GT như khi train để giữ đúng xử lý dữ liệu.

Kiểm tra bản KD full dataset: tối đa **512 mẫu ngẫu nhiên không hoàn lại cho mỗi bộ** Human36M, MuCo, MSCOCO, CrowdPose và 3DPW test:

```bash
python main/eval_difflifter_3dpw.py \
  --cfg config/train_student_kd_e1_v2_diff_jotr.yml \
  --ckpt experiment/finetune_difflifter_r25_cam/best_epoch.bin \
  --train_sets config --sets 3dpw --student_input \
  --S 1 --K 1 --max_samples 512 --batch_size 32 --workers 0 --seed 123 --gpu 0 \
  --out_dir experiment/check_lifter_input_full_s123
```

Kiểm tra bản gate đang train trên 3DPW-train: cùng script, đổi YAML và thư mục kết quả:

```bash
python main/eval_difflifter_3dpw.py \
  --cfg config/train_student_kd_e1_v2_diff_spread_gate.yml \
  --ckpt experiment/finetune_difflifter_r25_cam/best_epoch.bin \
  --train_sets config --sets 3dpw --student_input \
  --S 1 --K 1 --max_samples 512 --batch_size 32 --workers 0 --seed 123 --gpu 0 \
  --out_dir experiment/check_lifter_input_3dpw_s123
```

Các cờ:

- `--train_sets config`: lấy các bộ trong `DATASET.train_list`, đo riêng từng bộ, không dùng ratio/oversampling. Có thể chỉ định `Human36M,MuCo` hoặc `3dpw-train`.
- `--sets 3dpw`: đánh giá 3DPW test. `--sets config` dùng `DATASET.test_list`. Vẫn hỗ trợ các split cũ `3dpw-train-noaug,3dpw-val,3dpw`.
- `--max_samples 512`: chọn ngẫu nhiên toàn dataset, giữ seed/indices trong summary; nếu ít hơn 512 thì lấy hết. 0 là toàn bộ như lệnh cũ. Số hợp lệ có thể ít hơn số chọn vì bỏ GT rỗng.
- `--student_input`: bật `DIFFLIFT.eval_student_input` (key mới mặc định False), yêu cầu YAML Student DiffLifter seqlen1. Loader giữ `lift_only=False`, không đổi sang target của nhánh lifter. S/K mặc định từ MODEL; có thể ghi đè bằng `--S/--K` nhưng sẽ ghi chú nếu khác YAML. Pose chính dùng noise seed=0 reset mỗi batch như ARTS, còn `--seed` điều khiển chọn mẫu/augmentation. Không hỗ trợ selector, pose augmentation hoặc s2d noise trong chế độ này.
- `--workers 0`: xử lý tuần tự để dễ tái lập augmentation; bỏ cờ thì dùng YAML. Giữ cùng seed/workers/batch khi đối chiếu. Tập con có vị trí mẫu trong batch khác full eval nên không so số tuyệt đối với log toàn tập như cùng một phép đo.
- `--out_dir`: tùy chọn lưu `eval_log.txt`, `summary.json`, `samples.csv` và YAML; từ chối thư mục đã có nội dung. Bỏ cờ để chỉ in như trước.

In **MPJPE17** có pelvis, **MPJPE16** bỏ pelvis, lỗi từng khớp, số chọn/hợp lệ/bỏ và nguồn GT. Train ở chế độ Student in thêm MPJPE theo mask GT `fit_joint_trunc * is_valid_fit` như log lift_err của trainer (tổng lỗi chia tổng trọng số toàn tập con; có thể khác trung bình các batch trong log train). Không dùng confidence 2D để loại khớp khỏi metric 3D. Đây không phải MPJPE mesh 14 khớp.

Train giữ augmentation của dataset/YAML, test dùng loader eval; 3DPW-train-noaug vẫn có thể dùng khi muốn so tập train tắt augmentation. Với `fusekin_unified=True`, 4 bộ train đều dùng target hồi quy từ mesh NeuralAnnot, ghi rõ **pseudo-GT**; không coi chênh lệch với 3DPW GT là thuần lỗi tổng quát hóa. Human36M test riêng dùng mocap/GT 2D sạch và được ghi nhãn riêng. Script dùng chung `diag_difflifter.build_model/get_gt/make_cond`, không sửa kiến trúc/checkpoint lifter hay các run đang train. Dataset lỗi được ghi riêng, chạy tiếp bộ còn lại và cuối cùng trả exit code 1.

Trạng thái: đã sửa và rà source/diff, **chưa chạy/import/kiểm cú pháp Python hoặc đánh giá trên GPU tại local**.
## Thử KD quan hệ joint trên 3DPW

Bản mới giữ gate_zero, Teacher init và global KD như run E1 gate_zero; chỉ thay loss KD joint sau fusion bằng MSE ma trận quan hệ cosine(LN(token)) 17x17. Mask cặp dùng GT hợp lệ hai đầu, bỏ đường chéo. Hai cờ mới mặc định tắt, YAML/run cũ giữ công thức cũ.

```bash
python main/train.py \
  --cfg config/train_student_kd_relation_v2_diff_gate_zero_3dpw.yml \
  --exp_name student_kd_relation_v2_diff_gate_zero_3dpw_s123 \
  --seed 123 --gpu 0
```

Giữ đúng phiên bản trọng số Teacher/lifter và seed của run KD gate_zero làm đối chứng; đường dẫn lifter giống nhau chưa bảo đảm cùng file lúc train. Không thay checkpoint lifter bằng L2 mới nếu muốn so riêng tác động loss với run cũ. Mốc KD gate_zero người dùng báo 76.11/93.84 mm; init-only TEST CUỐI 77.65/51.72/94.51 mm. Bản relation chưa có kết quả.

Log khởi động cần có `KD joint = relation`. `J` là loss quan hệ, `G` vẫn là global feature KD. Cuối epoch có `KD joint RELATION` và số cặp hợp lệ/mẫu (tối đa 272 với 17 khớp, tính hai chiều). `KD relation grad@fusion` ở batch đầu mỗi epoch báo norm gradient J/G/hard đã nhân trọng số trên cùng tham số fusion; chỉ để chẩn đoán, không tự cân bằng loss. W&B lưu `kd_joint_relation`, `kd_relation_pairs`, `kd_rel_grad_joint/global/hard`. Log gradient thêm chi phí backward tại batch đó; có thể tắt bằng `MODEL.kd_joint_relation_log_grad=False` trong cấu hình run mới trước khi chạy.

Checkpoint mới nằm ở `experiment/student_kd_relation_v2_diff_gate_zero_3dpw_s123/checkpoint/best.pth.tar`. Không đổi kiến trúc model hoặc inference; test dùng đúng YAML của run:

```bash
python main/test_ckpt.py \
  --cfg config/train_student_kd_relation_v2_diff_gate_zero_3dpw.yml \
  --ckpt experiment/student_kd_relation_v2_diff_gate_zero_3dpw_s123/checkpoint/best.pth.tar \
  --test_list 3dpw --gpu 0
```

Chọn best trên 3DPW test như đối chứng chỉ là giao thức phát triển. Đã rà source/diff/YAML bằng văn bản; **chưa chạy/import/kiểm cú pháp Python hay train/test local**. Chi tiết mục 27 của experiment.md.

## Xem ma trận quan hệ Teacher trước khi train KD mới

Đã mở rộng script hiện có, không dựng thêm Teacher builder. Mode `teacher_relation` yêu cầu YAML/checkpoint **Teacher GT thuần** và nạp strict; ảnh + GT3D sạch, PE2D detector theo YAML. Dùng chung `utils.joint_relation.joint_relation_matrix` với loss KD để tính cosine(LN(token)) đúng công thức, trước/sau fusion.

```bash
python main/inspect_student_spread.py \
  --mode teacher_relation \
  --cfg config/train_teacher_k16_v2.yml \
  --ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar \
  --set 3dpw --relation_samples 512 --num_vis 12 \
  --batch_size 32 --workers 0 --seed 123 --gpu 0
```

Phải dùng YAML kiến trúc của đúng Teacher checkpoint (gồm img_probe); không dùng YAML Student/KD. Mặc định tạo thư mục mới `teacher_relation_3dpw_<timestamp>` cạnh checkpoint và in đường dẫn. Có thể đặt `--out_dir <thu_muc_moi>`; thư mục có nội dung bị từ chối. `--relation_samples 0` chạy toàn tập; `--sample_ids 10 200 1000` chọn index dataset cụ thể. Thống kê lấy tối đa 512 mẫu ngẫu nhiên không hoàn lại, không lấy 512 frame đầu; seed cố định. `--num_vis` là tổng số hình mẫu trong mode này. Có thể đổi `--set` sang `3dpw-pc` hoặc `3dpw-crowd`, giữ đúng file Halpe của split.

Đầu ra:

- `overview.png`: quan hệ trung bình và độ lệch chuẩn **giữa các mẫu**, trước/sau fusion. Đường chéo tô xám, không tính thống kê.
- `vis/*_teacher_relation.png`: ảnh crop + detector 2D, GT/Teacher pose nhìn trước/ngang, heatmap trước/sau fusion và phần lệch so với ma trận trung bình. Màu cosine cố định [-1,1]; thang màu phần lệch dùng chung giữa các ảnh.
- `relations.npz`: toàn bộ ma trận, GT/Teacher pose mét, sample_ids và joint_names.
- `pairs.csv`: mean/std của từng cặp trong 136 cặp khác nhau; `samples.csv`, `summary.json`, `config.yml`, `inspect_log.txt` để đối chiếu mẫu/checkpoint.

Đọc `STD across samples` để biết **cùng một cặp khớp có thay đổi giữa các mẫu không**. `constant-template MSE` là lỗi dùng duy nhất ma trận trung bình để mô tả các mẫu, không phải KD loss Student. Ma trận biến thiên chưa chứng minh tín hiệu có ích cho KD hay đến từ ảnh; GT pose/định danh token cũng ảnh hưởng. Không có ngưỡng tự kết luận tốt/xấu. Test có GT mesh đầy đủ 17 khớp; không dùng detector mask để xóa cặp quan hệ. Pose trên hình dùng MPJPE17 từ mesh, không phải metric mesh14; đây không phải attention hay khoảng cách 3D. Mới đọc/rà source/diff, **chưa chạy Python/test/model local** theo AGENTS.

## So heatmap Teacher / Student KD hien tai / Student KD quan he

Van dung `--mode teacher_relation`. `--cfg/--ckpt` la Teacher; them cap `--student_cfg/--student_ckpt` de ve cung Student KD hien tai. Vi du dung gate_zero lam doi chung 3DPW:

```bash
python main/inspect_student_spread.py \
  --mode teacher_relation \
  --cfg config/train_teacher_k16_v2.yml \
  --ckpt experiment/teacher_gt_v2/checkpoint/best.pth.tar \
  --student_cfg config/train_student_kd_e1_v2_diff_spread_gate_zero.yml \
  --student_ckpt experiment/student_kd_e1_v2_diff_spread_gate_zero/checkpoint/best.pth.tar \
  --set 3dpw --relation_samples 512 --num_vis 12 \
  --batch_size 32 --workers 0 --seed 123 --gpu 0
```

Neu ten run that khac vi du, doi `--student_ckpt` sang best.pth.tar cua run do va dung dung YAML train. Co the dung KD khong gate, mien la Student KD feat DiffLifter S1K1 (khong selector/hint). Teacher phai dung checkpoint/YAML Teacher can doi chieu; khong tu suy Teacher moi nhat tu thu muc.

Sau khi **DA TRAIN** Student KD quan he, them hai tham so vao lenh tren:

```bash
  --relation_cfg config/train_student_kd_relation_v2_diff_gate_zero_3dpw.yml \
  --relation_ckpt experiment/student_kd_relation_v2_diff_gate_zero_3dpw_s123/checkpoint/best.pth.tar
```

Chua co checkpoint relation thi bo ca hai co nay, van ve Teacher + Student hien tai. Khong thay checkpoint cu bang YAML bat co de gia lap ket qua train relation. Mac dinh luu thu muc `compare_relations_<split>_<timestamp>` canh checkpoint Student hien tai, khong ghi de; co the dat --out_dir moi.

- `overview.png`: moi cot la mot checkpoint, cac hang mean/std quan he truoc/sau **fusion**. Cung phep LN/cosine va thang mau.
- `vis/*_compare_relations.png`: chung crop/pose; cac cot Teacher, KD hien tai, KD quan he neu co; ma tran truoc/sau fusion va hieu Student AFTER - Teacher AFTER. Thang hieu dung chung cho cac model/mau; duong cheo bo.
- `samples.csv`: MPJPE17 tu mesh, loi pose lifter, MSE relation tho so Teacher truoc/sau fusion; neu co relation checkpoint them gain KD hien tai - KD moi cho ca relation MSE va MPJPE17. Duong la tot hon.
- `relations.npz`, `pairs.csv`, `summary.json`, YAML rieng tung model va log: luu indices/names/ma tran/pose/epoch/cau hinh, input_parity va khac biet cau hinh train.

Tat ca cung loader Student hien tai; moi model nap strict va forward trong snapshot YAML RIENG tu default, tranh ro key giua Teacher/Student. Teacher dung GT3D, Student dung lifter trong checkpoint. Neu co hai Student, pose S1 phai khop <=1e-6 met; std cung duoc so neu ca hai gate. Khac pose/std thi dung ro loi, khong so nhu ablation cung input. Config train khac ngoai hai co relation/log se duoc luu va canh bao; input parity khop khong du chung minh chi khac loss.

**Before/after fusion** la vi tri token; **KD hien tai/KD quan he** la hai run train. Relation MSE o day do cosine token tho, khong qua projector va khong phai full loss KD E1. Gan Teacher ve quan he chua chac giam loi mesh; MPJPE tren hinh la **H36M17**, khong phai MPJPE14 evaluator. Kiem tra bang van ban/source/diff; chua chay Python/model/test local theo AGENTS.
