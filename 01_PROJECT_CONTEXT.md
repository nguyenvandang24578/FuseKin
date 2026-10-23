# FuseKin — bối cảnh kỹ thuật hiện tại

Cập nhật **2026-10-23**. Nội dung kỹ thuật đối chiếu với code/YAML local; đây không phải xác nhận trạng thái môi trường hoặc file trên server. Kết quả và tiến độ nằm trong [experiment.md](experiment.md), lệnh thường dùng trong [README.md](README.md).

## 1. Phạm vi và hướng đang làm

- Bài toán hiện tại: **image-based**, một ảnh, `seqlen: 1`.
- Hai phần được train riêng: lifter 2D→3D; Student kết hợp ảnh + pose để dự đoán mesh.
- Lifter đang dùng trong các YAML Student diffusion: `r25_cam`, đóng băng, checkpoint `experiment/finetune_difflifter_r25_cam/best_epoch.bin`.
- Augmentation: người dùng báo không hiệu quả, chưa có số chi tiết. Spread: người dùng báo best trên 3DPW 77.59 / 94.17 mm; chẩn đoán 35,515 mẫu đã có log: real/off/shuffle gần nhau, chưa thấy lợi ích rõ lúc inference. GCN + attention chưa cải thiện tổng thể; giữ lifter `r25_cam`. Xem mục 5 và 9 của `experiment.md`.
- Selector đã có trong code nhưng chưa chứng minh được lợi ích trên mesh. KD/hint hạ ưu tiên; khởi tạo nhiễu SimCC/GMM tạm hoãn.

## 2. Student v2 và tích hợp DiffLifter

| Thành phần | Cách xử lý hiện tại |
|---|---|
| Ảnh | Crop 256×256 → ResNet-50 pretrained SPIN → đặc trưng layer4; backbone đóng băng |
| Pose | Halpe 2D → lifter đóng băng → H36M-17 3D, mét, root-relative |
| Token ảnh/khớp | Chiếu về embedding 512; PE2D Fourier dùng chung cho vị trí 2D của khớp và tâm ô ảnh |
| Fusion | 3 block: self-attention khớp, cross-attention hai chiều ảnh↔khớp, FFN |
| Ngữ cảnh xương | HyperGCN tạo `ctx` cho 17 khớp |
| Pose SMPL | 24 query từ mean pose và đặc trưng xương cha/con → cross-attention `ctx` → delta rotation 6D theo khớp → góc local; hai khớp tay cuối đặt 0 |
| Shape/camera | Shape token đọc `ctx` + token ảnh; camera từ đặc trưng toàn cục |

Các cờ kiến trúc v2 (`joint_pe2d`, `fusion_joint_self_attn`, `pose_head_mode='per_joint'`, `pose_query_bone`, `shape_attn_mode='tokens'`) đã được bật trong các YAML hiện tại. Không nhầm chúng với cờ tùy chọn mới của denoiser lifter.

**Các cách dùng giả thuyết trong `ARTS.forward_student`:**

- Bình thường: `MODEL.pose_lifter: 'diffusion'`, `difflift_steps: 1`, `difflift_k: 1`.
- Augmentation: `difflift_aug_prob: 0.5`, `difflift_aug_steps: 5`. Khi train, mỗi mẫu có xác suất 0.5 nhận một pose sinh ngẫu nhiên S=5, K=1 thay pose chính; khi eval dùng S=1, K=1. Không có selector trong YAML aug.
- Selector: `difflift_select: True`, YAML dùng S=5, K=20. `HypSelector` đọc đặc trưng ảnh tại 2D từng khớp và hình dạng các giả thuyết, softmax theo K, gộp có trọng số từng khớp. Đây không phải chọn cứng cả pose.
- Gate: `difflift_spread=True` + `difflift_spread_gate=True` sinh nhóm S5/K20 riêng, truyền std xyz (mét) để điều chỉnh cập nhật ảnh; pose chính S1/K1. Nhánh cộng embedding spread đã xóa; không bật gate thì Student dùng pose DiffLifter thuần.

Các nhánh trên là thí nghiệm riêng; không bật cộng dồn nếu chưa có kế hoạch. Selector hiện nhận gradient từ đường mesh và loss phụ pose đã chọn; chưa có thay đổi chặn gradient từ FuseKin vào selector.

## 3. DiffLifter hiện tại

Code chính: `lib/models/diff_lifter.py`; train độc lập bằng `main/finetune_difflifter.py`.

| Mục | Giá trị/cách làm trong baseline `r25_cam` |
|---|---|
| Điều kiện | `(B,17,6)` = `[x, y, conf, cam_bbox(3)]` |
| Input denoiser | Nối `[x,y,conf]` với noisy 3D, chiếu thành token; cộng joint embedding, timestep embedding và camera embedding |
| Tiêm điều kiện | 2D được nối tại input một lần; không adaLN hay self-conditioning |
| Backbone | SpatialMixSTE, 8 block, D=256, 8 attention heads; spatial-only |
| Diffusion | Cosine T=1000; Gaussian noise; dự đoán `x0` |
| Scale | GT mét nhân 3 trước khi thêm nhiễu; clamp và chia scale trước denoiser |
| Output/loss | Output trừ pelvis, đơn vị mét; loss MPJPE là L2 norm, không bình phương |
| Sampling | DDIM, eta=1; S là số bước, K là số giả thuyết; mặc định gộp mean |
| Output eval | `mean_pose (B,17,3)`, `spread (B,17)`, `all_preds (B,K,17,3)` |
| Spread | Norm của std x/y/z giữa K giả thuyết; K=1 trả 0, không phải xác suất đã hiệu chuẩn |

Trong wrapper FuseKin, inference thường dùng generator seed 0; train selector/augmentation dùng nhiễu ngẫu nhiên ở các nhánh tương ứng. Không diễn giải S=1 là một lifter deterministic riêng được train trước.

### Denoiser GCN + attention mới

`DIFFLIFT.gcn_attn` mặc định **False**. YAML `finetune_difflifter_r25_cam_gcn_attn.yaml` bật True:

1. `LN(x)` đi vào hai nhánh **độc lập**: GCN skeleton và global self-attention.
2. Mỗi nhánh trả D kênh; concatenate thành 2D kênh rồi Linear về D.
3. Residual → LN → MLP → residual, giữ cách pre-LN/FFN của baseline.

`SkeletonGCN`: liên kết xương H36M-17 hai chiều + self-loop, chuẩn hóa đối xứng `D^(-1/2) A D^(-1/2)`, Linear và ReLU. Adjacency cố định, là buffer không persistent và không tham gia EMA.

Đây là bản áp dụng ý tưởng hai nhánh của FMPose3D, không phải sao chép toàn bộ model. Vẫn dự đoán `x0` bằng diffusion/DDIM. D=256, depth=8, dữ liệu, loss và lịch train giữ như `r25_cam`; số tham số tăng do thêm GCN/fusion. Ngày 23/10 người dùng đã gửi cặp kết quả hai best checkpoint epoch 6, S=1/K=1: GCN + attention chưa cải thiện tổng thể; bảng đối chứng ở mục 5 của `experiment.md`.

Các cờ cũ `cond_mode`, `conf_attn`, `skel_attn`, `mask_token`, `self_cond`, `cam_cond`, `aggregate` không còn là API của model/config hiện tại. JPMA chỉ còn là phép phân tích trong `main/diag_difflifter.py`. `config/diag_ocpc_clean.yaml` vẫn chứa key cũ, không dùng làm lệnh mẫu với config hiện tại.

## 4. Dữ liệu và hệ tọa độ

**H36M-17**, đúng thứ tự:

```text
Pelvis, R_Hip, R_Knee, R_Ankle, L_Hip, L_Knee, L_Ankle,
Torso, Neck, Nose, Head_top, L_Shoulder, L_Elbow, L_Wrist,
R_Shoulder, R_Elbow, R_Wrist
```

Cha lấy từ `H36M_PARENTS` trong `diff_lifter.py`. Không đổi định nghĩa khớp theo tên gần giống.

- `Human36M17Dataset` chuẩn hóa 2D theo `output_hm_shape`: `x/W*2-1`, `y/H*2-1`. Không trừ pelvis 2D. Mask=0 → tọa độ 2D đặt 0.
- Halpe-26 đổi qua `halpe26_to_human36m` trong `lib/utils/h36m_adapter.py`. Mask xét score > 0.3 và nằm trong crop; `lift_soft_conf=True` dùng score cắt [0,1] cho khớp hợp lệ.
- Lifter hiện luôn nhận `cam_bbox`: `[(cx-px)/f, (cy-py)/f, max(w,h)/f]`, có xử lý flip/rotation augmentation. Không cần cờ `lift_cam_bbox`/`cam_cond` cũ.
- 3D lifter: mét, root-relative tại pelvis H36M. 3DPW test dùng H36M regressor từ GT mesh; H36M test dùng mocap `orig_joint_cam`.
- Đơn vị/root của target mesh là chuyện riêng của dataset/evaluator; không áp quy ước lifter cho mọi tensor tên `joint_cam`/`smpl_mesh_cam` một cách máy móc.

| Tập dùng train lifter `r25_cam` | Nguồn 3D | Nguồn 2D/camera |
|---|---|---|
| Human36M | Mocap | Halpe-26, camera dataset |
| MuCo | 3D dataset | Halpe-26, camera dataset |
| MSCOCO | Pseudo-SMPL NeuralAnnot → mesh → H36M-17 | Halpe-26; camera ước lượng và `view_rot` |
| CrowdPose | Pseudo-SMPL NeuralAnnot → mesh → H36M-17 | Halpe-26; camera ước lượng và `view_rot` |

`lib/utils/lift_pseudo3d.py` chỉ giữ người có cả NeuralAnnot và Halpe; không coi pseudo-GT là mocap thật. Nhánh `lift_only` không đọc ảnh RGB, nhưng có dummy `img` để giữ giao diện.

Các file keypoint chính:

| Tập | File dưới `data_final/` | Tool |
|---|---|---|
| Human36M | `Human36M/Human36M_{train,test}_halpe26.npz` | `tools/extract_halpe26_h36m.py` |
| MuCo | `MuCo/MuCo_train_halpe26.npz` | `tools/extract_halpe26_muco.py` |
| MSCOCO/CrowdPose | `<name>/<name>_train_halpe26.npz` | `tools/extract_halpe26_coco.py`, chọn dataset `mscoco` hoặc `crowdpose` |
| 3DPW | `PW3D/3DPW_{train,validation,test,oc,pc}_halpe26.json` | `tools/extract_halpe26_3dpw.py` |

OC/PC phải dùng file Halpe và ann_id của chính split đó. `pw3d_ocpc_test_only=True` lọc sequence thuộc test khi cần phân tích tập sạch. 3DPW-Crowd dùng Higher-HRNet; code hiện loại nhánh Halpe cho Crowd, không còn lỗi cũ áp Halpe vào Crowd. CrowdPose hiện không lọc fit bằng ngưỡng `fit_err > 5` cũ.

Các ghi chú cũ về mesh target/mask/collate khi train **mesh đa dataset** chưa được kiểm chứng lại đầy đủ. Không suy từ việc lifter trộn 4 tập chạy được rằng pipeline mesh trộn 4 tập đã đúng; cần kiểm đơn vị, root, khớp regressor, `is_3D` và `is_valid_fit` trước khi mở rộng.

## 5. Config, checkpoint và môi trường

- Key phải được khai báo trong `lib/core/config.py` trước khi YAML ghi đè; key cũ không tồn tại sẽ gây `ValueError`.
- Lifter: `difflift_cfg` lưu kiến trúc, gồm `gcn_attn`; `model_state` là EMA khi bật EMA; `raw_model_state` phục vụ resume.
- Checkpoint thiếu `gcn_attn` được hiểu là False. `diag.build_model` và `ARTS._build_diff_lifter` đọc cờ từ checkpoint. `finetune_difflifter.load_ckpt` chặn nạp sai loại backbone.
- `main/inspect_difflifter.py` dùng chung `build_model` với diag. Đánh giá lifter bằng **`best_epoch.bin`**; mesh bằng **`checkpoint/best.pth.tar`**.
- `main/train.py --exp_name NAME` tạo `experiment/NAME`; tên mặc định có timestamp đến giây theo UTC+7. Thư mục tên tường minh đã tồn tại và không rỗng sẽ bị từ chối nếu không resume, không bị tự xóa như tài liệu cũ ghi.
- Môi trường `arts` dùng cho model; `pose2d` dùng RTMPose/ONNX. Người dùng đã xác nhận probe chạy với rtmlib 0.0.16 và onnxruntime-gpu 1.19.2. Không mặc định các phiên bản/hardware server cũ vẫn đúng.
- `tools/probe_halpe26_simcc.py` lưu raw SimCC và phép biến đổi crop→ảnh; sigma/entropy từ normalization là chẩn đoán, chưa hiệu chuẩn thành sai số thực. Không đưa probe vào pipeline train hiện tại.

## 6. Giao thức và chọn checkpoint

| Phép đo/run | Giao thức đang có trong source/YAML |
|---|---|
| Lifter `r25_cam` và bản GCN + attention | Train H36M + MuCo + MSCOCO + CrowdPose; không `3dpw-train`; best theo **`3dpw-val`**, có log test trong train |
| Student `train_student_v2_diff.yml`, `_diff_aug.yml`, `_diff_sel.yml` và hai bản gate KD | Train **`3dpw-train`**; `test_list=['3dpw']`, `TEST.select_dataset='3dpw'` -> chọn best trên **test** |
| Mesh best | `main/train.py` xét cải thiện MPJPE **hoặc** MPVPE trên tập được chọn; không phải chỉ một metric |

Comment “mỗi epoch chỉ val” trong các YAML Student trên đang không khớp giá trị thật. Đợt cập nhật tài liệu không thay YAML hoặc thay run đang train. Các kết quả chọn theo test là kết quả phát triển, cần giao thức tách validation trước khi dùng làm bảng đánh giá cuối.

Mục tiêu so với JOTR là **toàn pipeline không dùng dữ liệu 3DPW để train**, không chỉ riêng lifter. Vì Student hiện vẫn train trên 3DPW, chưa được gọi toàn pipeline hiện tại là giao thức JOTR. OC/PC có sequence từ nhiều split; Crowd liên quan validation, cần xem cả tập train lẫn tập chọn checkpoint khi báo kết quả. Halpe trích bằng bbox GT cũng phải được công bố trong thiết lập so sánh.

| Thước đo | Dùng cho |
|---|---|
| H36M-17 root-relative, MPJPE mm | So lifter với lifter; oracle pose input |
| `dataset.evaluate`, 14 khớp, MPJPE/PA-MPJPE/MPVPE mm | So mesh/FuseKin |

Không so trực tiếp hai loại MPJPE. Oracle dùng GT để chọn là phép đo tiềm năng, không phải kết quả triển khai không có GT.

## 7. Chẩn đoán gate và Teacher GT

Giữ script `main/inspect_student_spread.py`; mặc định `--mode gate`, `--mode spread` là alias gate. Không còn builder/replay cho Student cộng embedding spread. `capture_real` lấy ảnh/K/std/gate từ forward thật; `off` ép g=1, `shuffle` đổi std giữa mẫu trong batch. Replay kiểm tra mesh khớp wrapper; metric mesh dùng `dataset.evaluate` 14 khớp, chẩn đoán pose dùng H36M-17 bỏ pelvis. Xuất CSV, nhóm spread, tương quan từng khớp và hình; thống kê gate theo mẫu/khớp. Shuffle trong batch có thể yếu do frame gần nhau; gate khác 1 chưa chứng minh std hữu ích.

`--mode teacher` dùng YAML/checkpoint Teacher GT thuần, GT mesh -> H36M-17 -> root-relative -> mét; detector PE2D giữ nguyên. Có thể chọn sample index và dùng `--spread_csv` đọc CSV cũ để xác minh ảnh/ann_id, không chạy lại checkpoint Student cũ. Subset không liên tiếp dùng evaluator theo đúng index gốc. Teacher dùng GT là oracle. Lịch sử kết quả ở mục 9-10 và 13 của `experiment.md`; lệnh hiện tại ở README.

## 8. Student DiffLifter thuần và KD E1

`config/train_student_v2_diff.yml`: scratch, `student_use_kd=False`, pose S1/K1 từ lifter đóng băng và đặc trưng ảnh. Không sinh K/std cho fusion khi gate tắt.

`config/train_student_kd_e1_v2_diff.yml`: cùng pose S1/K1, Teacher GT đóng băng, Student khởi tạo phần chung của smpl_model từ Teacher. KD MSE qua Linear(LN(Student)) tới LN(Teacher) trên `feat_joint` (B,17,512) và `feat_global` (B,1024) sau cả 3 block fusion; global ghép mean token ảnh và mean token joint. Trọng số joint/global=1, trước fusion=0, hard_w=lambda=1, noise weighting=False, InfoNCE=0. Có hard loss GT và loss phụ privileged 3D. Không KD output mesh/SMPL hoặc attention map trong E1.

KD hiện yêu cầu đặc trưng cuối giống Teacher, chưa trực tiếp ép đường ảnh sửa các khớp lifter sai. Không đổi loss KD trong lần dọn. Kết quả KD cộng embedding cũ E18 và so paired được giữ như lịch sử tại mục 11-12,15 của `experiment.md`, không gán cho hai cấu hình thuần hoặc gate chưa có kết quả.

## 9. Đánh giá ghép cặp scratch/KD/Teacher

`main/inspect_student_spread.py --mode compare` dùng YAML riêng từ cùng default; `compare_config` khôi phục cfg riêng cho mỗi builder/forward. Hỗ trợ Student DiffLifter thuần/gate và Teacher GT, nạp strict. Luôn kiểm tra hai pose S1 khớp và Teacher input đúng GT; chỉ so raw std khi cả hai bật gate và dùng cùng S/K tính std. Nếu không so std, summary ghi std_compared=False và std_max_abs_m=None; CSV chỉ có spread_mm khi có nguồn std thật, ghi rõ spread_source.

Metric mesh theo evaluator. Nhóm Q1-Q4/top10% chọn theo lỗi scratch, nhóm missing/Teacher-better, gain scratch-KD và hình GT/scratch/KD/Teacher giữ nguyên. Chia nhóm theo lỗi scratch có thiên lệch chọn nhóm; không suy ra KD tự nhận biết ca khó. Hình replay nguyên batch DDIM gốc và kiểm tra metric/identity. Khởi tạo, hệ số hard/privileged, lịch LR và tập chọn best phải được xét khi so run. Không coi so gate KD với scratch thuần là ablation tách riêng std hoặc loss KD.

## 10. Gate cho Student KD

`MODEL.difflift_spread_gate=False` mặc định giữ Student thường, không tạo tham số gate. Bật gate chỉ ở Student; Teacher xây từ cùng YAML vẫn không có gate. `difflift_spread` và steps/k chỉ cấu hình lấy std cho gate; spread=True mà gate=False báo lỗi để chặn dùng nhầm YAML nhánh cũ. Giữ tên key và hai YAML gate đã gửi để đồng bộ server không đổi lệnh train.

`ARTS.forward_student` dùng pose S1/K1, std xyz (mét) từ S5/K20, truyền `joint_spread` qua FusionPose2Mesh/RGBJointFusion/RGBJointCrossTransformer tới 3 block. Raw std có trong `joint_spread_3d`. Không còn `spread_proj`, cờ runtime `diff_spread` hay tham số `extra_tok`; đường hint_z riêng giữ nguyên.

`JointImageSpreadGate`: MLP(1027 -> hidden64 -> 1), đầu vào `[LN(joint),LN(image_update),log1p(std_xyz*10)]`. Gate scalar riêng cho từng khớp/mẫu/block: g=2*sigmoid, Linear cuối zero-init nên g=1. Nhân g vào image_update sau out_joint trước residual/FFN, giữ coef1/coef2. Đây không phải confidence đã hiệu chuẩn hay tỷ lệ trộn chuẩn hóa. `difflift_spread_gate_zero=True` chỉ che std ở đầu vào gate, vẫn sinh K để giữ tính toán/RNG. Không ép spread cao luôn dùng ảnh nhiều hơn.

Student_Trainer giữ nguyên KD E1, log gate mean/std/min/max và kiểm tra key khi khởi tạo từ Teacher. main/train.py lưu metadata spread_gate_config={zero_std,hidden}; loader xác minh real/zero/hidden. Checkpoint nhánh cộng embedding cũ bị từ chối rõ ràng thay vì lọc mất spread rồi đánh giá sai; muốn tái hiện lịch sử phải dùng source/YAML của run cũ. Checkpoint Teacher, Student thuần và gate giữ kiến trúc tương ứng.

Hai YAML `train_student_kd_e1_v2_diff_spread_gate.yml`, `_gate_zero.yml` giữ nguyên: pose S1/K1, std S5/K20, KD E1, 30 epoch, LR [5,10,15,22,27], seed lệnh 123. Giữ mặc định tắt cho các key gate, không đổi lifter/constructor/save/load hay loss diffusion. Gate chưa hỗ trợ kd_res_mode=v2 và đường Student của content_ablation_test.py vì các đường đó chưa truyền std; không âm thầm bỏ gate.

## 11. Trạng thái dọn code

Người dùng chốt **chỉ bỏ nhánh cộng embedding spread, giữ gate + KD để train**. Đã xóa hai YAML spread cũ, gỡ nhánh trong ARTS/FusionPose2Mesh và inspector, giữ YAML gate real/zero. Thống kê spread của chính DiffLifter trong các công cụ chẩn đoán không bị xóa vì không phải đầu vào cộng token Student. Giữ kết quả/checkpoint đã train và lịch sử trong experiment.md; gate vẫn chờ kết quả người dùng chạy trên server. Chỉ kiểm tra văn bản/source/diff tại local, không import hay chạy Python.

## 12. Đo pose DiffLifter đầu vào trên tập con train/test

`main/eval_difflifter_3dpw.py` đã mở rộng theo yêu cầu: `--train_sets config` đo riêng các bộ DATASET.train_list, `--sets config` hoặc tên split cho eval, `--max_samples N` chọn ngẫu nhiên không hoàn lại tối đa N mẫu mỗi bộ. Không dùng ratio/oversampling của loader trộn, không lấy N frame đầu. Seed cố định và selected_indices được lưu khi có out_dir. Các tùy chọn mới mặc định tắt/rỗng; max_samples=0 và bộ split mặc định cũ vẫn eval toàn tập.

`DIFFLIFT.eval_student_input=False` là key mới chỉ phục vụ script eval; `--student_input` bật chế độ dùng YAML Student, yêu cầu pose_lifter=diffusion, seqlen1, lift_only=False và H36M-17. Đo 4 bộ train cần fusekin_unified=True; giữ nguồn Halpe/conf/camera/augmentation và GT orig_joint_cam như Student_Trainer (test có mesh thì dùng H36M regressor). Trong unified, target cả 4 bộ train là mesh NeuralAnnot, ghi nhãn pseudo-GT, khác nhánh lifter H36M/MuCo dùng mocap/native GT. Không chạy backbone ảnh/fusion/gate/Teacher hay KD; vẫn cần loader Student đọc ảnh/dựng GT. Chưa hỗ trợ selector, pose augmentation, s2d noise trong chế độ này.

Dùng chung diag_difflifter.build_model/get_gt/make_cond. Chế độ cũ giữ noise stream theo seed; chế độ Student lấy S/K từ MODEL nếu CLI không ghi đè và reset DDIM seed=0 mỗi batch, dự đoán cả batch trước khi loại GT rỗng, như ARTS.lift_diffusion. max_samples/batch/workers thay đổi có thể đổi vị trí nhiễu/augmentation; không đồng nhất với con số toàn split hoặc log train.

Metric raw: MPJPE17 có pelvis, MPJPE16 bỏ pelvis và từng khớp, root-relative mét -> mm, bỏ mẫu GT rỗng. Metric train phụ dùng mask GT fit_joint_trunc * is_valid_fit như trainer; không dùng mask detector để bỏ khớp 3D. Không phải metric mesh 14 khớp. Mỗi bộ ghi rõ GT, số chọn/số hợp lệ; không báo 0 nếu toàn GT rỗng.

--out_dir tùy chọn lưu log/JSON/CSV/YAML vào thư mục mới; CSV ghi index/ảnh/missing2D và lỗi từng mẫu, summary có per-joint/indices/config/noise và dataset lỗi. Không ghi đè kết quả cũ. Đã code/chờ người dùng chạy trên server, không chạy/import Python tại local; lệnh README, trạng thái experiment.md mục 17. Không đổi YAML train hay kiến trúc/save/load DiffLifter.