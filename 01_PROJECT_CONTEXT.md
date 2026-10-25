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

OC/PC phải dùng file Halpe và ann_id của chính split đó. `pw3d_ocpc_test_only=True` lọc sequence thuộc test khi cần phân tích tập sạch. 3DPW-Crowd tuân theo pw3d_2d_source: halpe26 dùng 3DPW_validation_halpe26.json và ann_id validation, lọc hai sequence Crowd như cũ; openpose giữ nhánh Higher-HRNet/fallback OpenPose lịch sử. Không dùng Halpe test cho Crowd. CrowdPose hiện không lọc fit bằng ngưỡng `fit_err > 5` cũ.

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

Người dùng chốt **chỉ bỏ nhánh cộng embedding spread, giữ gate + KD để train**. Đã xóa hai YAML spread cũ, gỡ nhánh trong ARTS/FusionPose2Mesh và inspector, giữ YAML gate real/zero. Thống kê spread của chính DiffLifter trong các công cụ chẩn đoán không bị xóa vì không phải đầu vào cộng token Student. Giữ kết quả/checkpoint đã train và lịch sử trong experiment.md; Người dùng đã báo gate std thật 77.47/94.06, gate_zero 76.11/93.84 và KD full 76.17/93.08 mm; xem mục 13 bên dưới và experiment.md mục 19, chưa xác minh epoch/best checkpoint. Chỉ kiểm tra văn bản/source/diff tại local, không import hay chạy Python.

## 12. Đo pose DiffLifter đầu vào trên tập con train/test

`main/eval_difflifter_3dpw.py` đã mở rộng theo yêu cầu: `--train_sets config` đo riêng các bộ DATASET.train_list, `--sets config` hoặc tên split cho eval, `--max_samples N` chọn ngẫu nhiên không hoàn lại tối đa N mẫu mỗi bộ. Không dùng ratio/oversampling của loader trộn, không lấy N frame đầu. Seed cố định và selected_indices được lưu khi có out_dir. Các tùy chọn mới mặc định tắt/rỗng; max_samples=0 và bộ split mặc định cũ vẫn eval toàn tập.

`DIFFLIFT.eval_student_input=False` là key mới chỉ phục vụ script eval; `--student_input` bật chế độ dùng YAML Student, yêu cầu pose_lifter=diffusion, seqlen1, lift_only=False và H36M-17. Đo 4 bộ train cần fusekin_unified=True; giữ nguồn Halpe/conf/camera/augmentation và GT orig_joint_cam như Student_Trainer (test có mesh thì dùng H36M regressor). Trong unified, target cả 4 bộ train là mesh NeuralAnnot, ghi nhãn pseudo-GT, khác nhánh lifter H36M/MuCo dùng mocap/native GT. Không chạy backbone ảnh/fusion/gate/Teacher hay KD; vẫn cần loader Student đọc ảnh/dựng GT. Chưa hỗ trợ selector, pose augmentation, s2d noise trong chế độ này.

Dùng chung diag_difflifter.build_model/get_gt/make_cond. Chế độ cũ giữ noise stream theo seed; chế độ Student lấy S/K từ MODEL nếu CLI không ghi đè và reset DDIM seed=0 mỗi batch, dự đoán cả batch trước khi loại GT rỗng, như ARTS.lift_diffusion. max_samples/batch/workers thay đổi có thể đổi vị trí nhiễu/augmentation; không đồng nhất với con số toàn split hoặc log train.

Metric raw: MPJPE17 có pelvis, MPJPE16 bỏ pelvis và từng khớp, root-relative mét -> mm, bỏ mẫu GT rỗng. Metric train phụ dùng mask GT fit_joint_trunc * is_valid_fit như trainer; không dùng mask detector để bỏ khớp 3D. Không phải metric mesh 14 khớp. Mỗi bộ ghi rõ GT, số chọn/số hợp lệ; không báo 0 nếu toàn GT rỗng.

--out_dir tùy chọn lưu log/JSON/CSV/YAML vào thư mục mới; CSV ghi index/ảnh/missing2D và lỗi từng mẫu, summary có per-joint/indices/config/noise và dataset lỗi. Không ghi đè kết quả cũ. Người dùng đã gửi log tập con train/test: H36M/MuCo/MSCOCO/CrowdPose/3DPW MPJPE17=68.580/113.010/92.476/165.385/69.843 mm; CrowdPose chỉ chọn được 280 mẫu. Cần đối chiếu native target lifter và độ phủ annotation/Halpe trước khi retrain; số liệu và giới hạn ở experiment.md mục 18. Không chạy/import Python tại local; lệnh README. Không đổi YAML train hay kiến trúc/save/load DiffLifter.
## 13. Trạng thái kết quả gate/KD mới nhất

Người dùng báo MPJPE14/MPVPE: gate std thật **77.47/94.06**, cùng gate std=0 **76.11/93.84**, KD không gate full dataset **76.17/93.08** mm. Hai YAML gate chỉ khác cờ zero; std thật kém +1.36/+0.22 mm trong cặp run này, chưa có bằng chứng giữ std. Zero vẫn là gate học từ joint/ảnh, không chứng minh gate tốt hơn không gate. Full dataset dùng Teacher/dữ liệu/lịch LR khác nên không phải ablation cho gate. Số liệu là mốc người dùng báo, chưa có epoch/header TEST CUỐI xác nhận best.pth.tar hay nhiều seed; đều chọn theo 3DPW test để phát triển. Chi tiết ở experiment.md mục 19.

Có sẵn `config/train_student_init_v2_diff_jotr.yml`, đã so toàn bộ giá trị với KD full: chỉ khác kd_lambda=0. Giữ init Teacher/hard_w1/privileged/lịch LR để đo đóng góp loss KD; student_use_kd=True nhằm giữ đường feat và Teacher init, không có gradient loss KD khi lambda=0. Chưa có kết quả đối chứng này trong log mới, không coi cấu hình có sẵn là run đã chạy. Không sửa code/YAML train trong lượt phân tích kết quả.

## 14. Nguồn 2D của 3DPW-Crowd (sửa 2026-10-24)

Sửa validation trước đó chỉ hết ValueError nhưng vẫn ép Crowd dùng HHRNet, chưa đúng pipeline Halpe theo yêu cầu người dùng. PW3D.load_data nay chọn self.use_halpe26 theo pw3d_2d_source cho mọi split: Crowd dùng 3DPW_validation_halpe26.json (cùng ann_id với 3DPW_latest_validation.json) và giữ lọc courtyard_hug_00/courtyard_dancing_00. Khi chọn Halpe không mở file HHRNet, không fallback OpenPose; thiếu file/ann_id Halpe báo lỗi rõ. OC/PC vẫn dùng file riêng; openpose giữ hành vi HHRNet cũ trên Crowd. __getitem__ đã ưu tiên halpe_h36m nên không cần đổi.

Không đổi YAML/default key hay model/train. Kết quả Crowd từ nhánh HHRNet/OpenPose trước đây thuộc nguồn detector khác; cần đánh giá lại khi chuyển Halpe. Chỉ rà source/diff, chưa chạy Python/test; xem experiment.md mục 21.

## 15. Doi chung KD mot bo 3DPW (2026-10-24)

Theo yeu cau nguoi dung, uu tien doi chung mot bo thay full dataset. config/train_student_init_v2_diff_gate_zero_3dpw.yml sao chep ban KD gate_zero 3DPW, chi doi kd_lambda=0. Giu gate_zero/Teacher init/hard_w1/privileged, lich LR va luong sinh std/forward de so loss KD voi moc 76.11/93.84. Khong phai scratch hay ban khong gate. Nguoi dung da gui TEST CUOI run student_init_v2_diff_gate_zero_3dpw_s123, best epoch12, n35515: MPJPE14/PA-MPJPE/MPVPE=77.65/51.72/94.51 mm. Moc KD gate_zero cu 76.11/93.84 tot hon1.54/0.67 mm; can xac minh lifter/Teacher weights va seed cung nhau vi r25_cam tren server da thay phien ban, cung YAML path chua du. Lenh o experiment.md muc22, ket qua va gioi han muc26. Khong doi kien truc/default/YAML cu. Nguoi dung xac nhan CrowdPose da full du lieu; khong tiep tuc coi nghi van do phu o muc 12 la blocker hien tai.

## 15. Key moi 2026-10-24 (mac dinh tat)
- `DATASET.halpe_gt_mask_thr` (0): tap train 4 bo, khop Halpe lech GT 2D > nguong (ty le canh crop) -> mask=0; ham `lib/utils/lift_pseudo3d.gt_mask_halpe`, GT tu `data['gt2d_px']`. Ap dung ca lifter lan Student unified.
- `DIFFLIFT.train_eval_samples` (0): MPJPE/PA S=1 cuoi moi epoch tren tap con co dinh moi bo train (co aug train).
- `DIFFLIFT.select_metric` ('mpjpe' | 'pa_mpjpe'): chi so chon best tren val_set.
- `main/finetune_difflifter.py` log loss x0 tung bo train (TagDataset gan meta['ds_id']).

## 16. KD quan hệ token joint (2026-10-24, đã code/chờ chạy)

Hai key MODEL mới mặc định False: kd_joint_relation, kd_joint_relation_log_grad. Student_Trainer._kd_joint_relation lấy feat_joint sau fusion (B,17,512), LN không affine rồi L2 normalize từng token FP32, tạo R=ZZ^T (B,17,17). Loss MSE quan hệ Student/Teacher detach, mask cặp từ valid_joint=fit_joint_trunc*is_valid_fit, bỏ đường chéo, chuẩn hóa theo tổng trọng số cặp mỗi mẫu rồi trung bình batch. Đếm cả hai chiều; 17 khớp hợp lệ có 272 cặp. Dưới hai khớp hợp lệ trả loss0. Đây là quan hệ feature, không phải attention/khoảng cách 3D.

Bật cờ chỉ thay kd_joint_ps; global/proj KD, hard/privileged và loss tổng giữ nguyên. Chỉ hỗ trợ student_use_kd=True, kd_mode=feat, kd_w_joint>0, kd_noise_weight=False (kết hợp noise weighting bị từ chối rõ). Joint projector cũ vẫn được tạo/lưu để giữ RNG và resume; loss quan hệ đi trực tiếp vào token, không qua projector joint. Không đổi kiến trúc/model state_dict hay checkpoint lifter/Teacher/gate. Cờ False giữ công thức cũ. Log gradient tùy chọn dùng autograd.grad/retain_graph trên cùng tham số fusion ở batch đầu mỗi epoch, không tích lũy .grad/không step hay tự cân bằng loss; báo norm J/G/hard đã nhân trọng số. Có thêm chi phí backward ở batch đó. Không đo grad hard/global theo feature gather DataParallel vì các output gather là nhánh riêng.

YAML mới config/train_student_kd_relation_v2_diff_gate_zero_3dpw.yml sao chép KD gate_zero, chỉ thêm hai cờ trên=True; dữ liệu/Teacher/lifter/init/30 epoch/LR/global KD/hệ số đều giữ. Chưa tự hiệu chuẩn hệ số loss mới. Run mới student_kd_relation_v2_diff_gate_zero_3dpw_s123; bắt buộc giữ đúng phiên bản trọng số lifter/Teacher như mốc KD gate_zero nếu muốn so tác động loss. Chọn best 3DPW test như run cũ chỉ phục vụ phát triển. Đã đối chiếu key/YAML/source/diff bằng văn bản, chưa chạy/import/kiểm cú pháp Python hay train/test local. Chi tiết experiment.md mục27, lệnh README.

## 17. Visual quan hệ Teacher GT (2026-10-24, đã code/chờ chạy)

main/inspect_student_spread.py thêm mode teacher_relation, --relation_samples (512 mặc định, chỉ mode mới dùng; 0 toàn tập), hỗ trợ sample_ids, num_vis là tổng hình mẫu. Dùng lại load_model strict Teacher GT, không nhân bản builder; giữ mode mặc định gate và luồng cũ. Thêm lựa chọn split Crowd theo loader đã sửa. Helper lib/utils/joint_relation.py::joint_relation_matrix được dùng chung bởi Student_Trainer._kd_joint_relation và inspector: cùng LN/L2 normalize FP32/R=ZZ^T như code KD đã duyệt, chỉ gom phép tính vào một nơi, không đổi mục tiêu/hệ số/mask KD.

Inspector Teacher nhận ảnh+GT mesh hồi quy H36M17 root-relative mét và PE2D detector như YAML; không lifter/Student/train/corrupt GT. Ma trận trước fusion từ feat_joint_in (đã joint encoding/PE2D), sau fusion từ feat_joint. Thống kê bỏ đường chéo, mọi khớp GT mesh hợp lệ; detector mask chỉ áp PE2D. Chọn ngẫu nhiên không hoàn lại tối đa N mẫu, giữ indices/seed. Xuất overview mean/std trước-sau, hình từng mẫu crop/GT-pred pose/heatmap/phần lệch mean, relations.npz, pairs.csv, samples.csv, summary/config/log. Constant-template MSE là phương sai quần thể mẫu của các phần tử ngoài đường chéo, không phải loss Student. Một mẫu có std0 được cảnh báo không đủ kết luận. Không tự gắn quan hệ feature với quan hệ giải phẫu hay attention, không suy ảnh hữu ích/KD hiệu quả chỉ từ hình. Không đổi YAML/train/checkpoint, không thêm key mô hình mới. Chưa chạy model/test/import Python local theo AGENTS; lệnh README, trạng thái experiment.md mục28.

## 18. Visual relation Teacher va hai Student (2026-10-24, da code/cho chay)

Mode teacher_relation trong main/inspect_student_spread.py mo rong them cap --student_cfg/--student_ckpt (KD feat hien tai, khong relation) va cap --relation_cfg/--relation_ckpt tuy chon (Student da train kd_joint_relation=True). Khong co Student thi giu Teacher-only nhu truoc; co Student thi inspect_relation_compare dung load_model strict/read_compare_config/compare_config hien co, snapshot rieng tung YAML tu default. Teacher GT + Student KD hien tai ve duoc ngay; Student relation chi ve khi da co checkpoint that.

Dung chung loader/crop/PE2D/camera cua Student hien tai; Teacher nhan GT H36M17 root-relative met, Student nhan DiffLifter S1K1 cua checkpoint, khong feed GT vao Student. Hai Student can pose khop <=1e-6m; std khop neu ca hai gate. Ghi khac biet config train ngoai co relation/log de tranh quy gain rieng cho loss khi khac dataset/Teacher/hard/LR. Khong so sanh scratch trong mode nay. Shader heatmap chung plot_joint_relation, shared matrix helper giu nguyen; khong doi kien truc/loss/YAML train.

Hinh cot Teacher/KD current/KD relation (neu co), hang truoc/sau fusion, hieu Student-Teacher; crop va pose chung, thang cosine va thang hieu chung, diagonal xam. Overview mean/std tung checkpoint, CSV NPZ JSON luu ca relation MSE tho vsTeacher va MPJPE17/input error/gain current-new/indices/epoch/parity. Relation MSE khong qua projector, khong phai full loss E1; gan Teacher ve relation chua chac mesh tot. Phan biet vi tri truoc/sau fusion voi run truoc/sau doi KD. Default model flag/luong Teacher-only va compare scratch cu giu, chi them CLI diagnostic mac dinh rong. Chua chay Python/train/test local, lenh README va experiment.md muc29.

## 19. Attention joint->image Teacher/Student (2026-10-24, da code/cho chay)

Nguoi dung chon xem attention hai checkpoint truoc, chi xet KD neu khac biet ro; giong nhau thi uu tien huong khac. main/inspect_student_spread.py them --mode attention, attn_block=-1 (cuoi), attn_joints (6joint ngoai bien de ve, all=17), alias --max_samples cua relation_samples. Moi CLI mac dinh khong anh huong mode cu; khong them MODEL key/khong thay train. Mode nay bat buoc Teacher+Student KD hien tai va tu choi relation checkpoint; dung lai strict loader/config snapshot/luong chung inspect_relation_compare. Teacher nhan GT3D, Student nhan lifter, cung crop/detector/camera. Khong can hai pose GT/lifter giong nhau: do la khac biet co y giua Teacher va Student.

lib/utils/attention_diagnostics.py: extract_joint_image_attention lay softmax Qjoint-Kimage that tu feat_layers, kiem tra (B,H,17,64), normalize/mean head FP32->CPU; save_attention_diagnostics xuat overlay8x8 cung scale, hieu Student-Teacher, mean maps/std nhan biet qua entropy/template-TV. TV/JS[0,1], KL Teacher||Student nats, entropy/log64, do da dang heads, perjoint/input errors/MPJPE17 tu mesh; Q1-Q4 theo input error16 bo pelvis. Template-TV dung mean Teacher cua cungjoint qua mau, can >=2mau. NPZ luu mean maps/pose/indices/grid centers. Khong mask attention theo detector/conf, chi luu detector_valid de doc nhom; tat ca17GTmesh hop le. Khong coi attention la confidence pose hay ep peak tai joint.

Output compare_attention_<split>_<timestamp> canh checkpoint Student: attention_overview.png, vis_attention/, CSV mau/khop/nhom, attention.npz, summary/YAML/log. Khong auto gan nhan hieu qua hay train theo threshold TV/JS. Khac mean maps chua chung minh Teacher tot hon; mean-head giong nhau khong dong nghia moi head giong nhau. Chua doi loss/model/checkpoint/YAML train, chua chay/import/kiem cu phap Python local theo AGENTS. Lenh README, trang thai experiment.md muc30.

## 16. Key moi 2026-10-25 (mac dinh tat)
- LOSS.mesh_root_h36m: loss mesh tru pelvis H36M cho pred (GT da tru pelvis H36M). Truoc: lech ~38 mm.
- LOSS.proj_h36m: loss chieu 2D dung H36M-17 tu mesh (giong GT 2D) thay vi SMPL-30 theo ten (lech TB 68 mm).
- Model tra them 'smpl_mesh_cam_render_det' (mesh tuyet doi, pose detach neu DETACH_POSE_FOR_PROJ).
