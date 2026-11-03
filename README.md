# FuseKin

Dự án nghiên cứu khôi phục mesh người từ một ảnh. Mô hình kết hợp đặc trưng ảnh với pose 3D được nâng từ keypoint 2D. Phiên bản chốt hiện tại là **FuseKin-v3** (DiffLifter + Teacher/Student v3).

Cập nhật: **2026-11-03**. Nội dung đã đối chiếu với source và YAML local. Trạng thái train trên server lấy từ thông tin người dùng cung cấp.

## Tài liệu

| File | Đọc khi cần |
|---|---|
| [01_PROJECT_CONTEXT.md](01_PROJECT_CONTEXT.md) | Kiến trúc v3, dữ liệu, quy ước, cấu hình và giới hạn hiện tại |
| [experiment.md](experiment.md) | Kết quả, trạng thái thí nghiệm và bước tiếp theo |
| [AGENTS.md](AGENTS.md) | Quy tắc làm việc cho AI/cộng tác viên |

Mỗi nội dung có một nơi chính để cập nhật. README của các repo tham khảo trong thư mục con được giữ riêng.

## Pipeline FuseKin-v3

1. **Keypoint 2D.** Test dùng OpenPose cho 3DPW/OC/PC và Higher-HRNet cho Crowd (giống JOTR/DPMesh). Train dùng 2D GT + nhiễu PoseFix kiểu JOTR. Mọi nguồn đều đổi về 17 khớp H36M.
2. **DiffLifter.** Nhận 2D, mask và camera bbox, sinh pose 3D H36M-17 (S=1, K=1). Lifter được đóng băng khi train Student.
3. **Đặc trưng ảnh.** ResNet-50 (pretrained SPIN) được fine-tune với lr nhỏ, có ghép heatmap 2D vào đầu mạng [A].
4. **Fusion.** Student v3 ghép token ảnh với token khớp, gồm:
   - đặc trưng ảnh tại vị trí khớp [B] và hop bias trong attention [B'];
   - phân bố độ sâu từng khớp [C];
   - tinh chỉnh lặp kiểu ReFit [D];
   - loss có độ bất định và nhiễu 2D khi train [E].
5. **Teacher v3.** Cùng kiến trúc nhưng nhận GT 3D. Student được khởi tạo từ Teacher và học KD trên đặc trưng token khớp và token toàn cục.

## Trạng thái hiện tại

- **Giao thức chốt (2026-11-03):** test bằng OpenPose/Higher-HRNet như JOTR/DPMesh.
- **Student v3 KD** (`student_v3_kd_jotr_pf05lift3dpw_t2`), MPJPE mesh 14 khớp: 3DPW **78.62** | OC **73.12** | PC **88.82** | Crowd **83.02** mm. Chưa vượt DPMesh (73.6 / 70.9 / 82.2 / 79.9).
- **Bản v3 dùng 2D Halpe** (`student_v3_kd_jotr`): 3DPW 74.62 / 47.64 / 89.80 (MPJPE / PA / MPVPE). Chỉ để tham khảo.
- **Nút thắt:** pose đầu vào từ lifter. Student gần như chỉ ngang lifter, và KD không lấp được khoảng cách ở token đầu vào. Chi tiết ở mục 5–7 của `experiment.md`.
- **Giới hạn giao thức:**
  - Lifter `jotr_pf05_3dpw` có train trên 3DPW-train, nên toàn pipeline chưa phải giao thức JOTR.
  - Student chọn best theo 3DPW **test**.
  - OC/PC không rò rỉ phải báo trên sequence test (`pw3d_ocpc_test_only`).

## Cấu trúc chính

| Đường dẫn | Vai trò |
|---|---|
| `lib/models/diff_lifter.py` | Diffusion 2D→3D và denoiser |
| `lib/models/ARTS.py` | Wrapper backbone/lifter/Teacher/Student, heatmap 2D [A] |
| `lib/models/teacher_student.py` | Fusion, head SMPL và các module v3 [B]–[E] |
| `lib/core/config.py`, `lib/core/base.py` | Config (cờ `MODEL.v3_*`), trainer, KD và evaluator |
| `lib/utils/jotr_dataset.py`, `lib/utils/posefix.py`, `lib/utils/fusekin_sample.py` | Dữ liệu H36M-17, nhiễu PoseFix kiểu JOTR, mẫu train thống nhất 4 bộ |
| `data_final/` | Dataset loader |
| `config/`, `main/`, `tools/` | YAML thí nghiệm, train/eval/chẩn đoán, trích keypoint |

## Lệnh chạy trên server

Chạy từ thư mục gốc repo, trong môi trường đã có dữ liệu, SMPL và checkpoint. Không đổi tên `data_final` thành `data` theo README ARTS cũ.

**Bước 1.** Train lifter, rồi đánh giá OC/PC chỉ trên sequence test (không rò rỉ):

```bash
python main/finetune_difflifter.py --cfg config/finetune_difflifter_d512_e15_jotr_pf05_3dpw.yaml --gpu 0
python main/eval_difflifter_3dpw.py --cfg config/eval_lifter_jotr_pf05_3dpw_ocpctest.yaml \
  --ckpt experiment/finetune_difflifter_d512_e15_jotr_pf05_3dpw/best_epoch.bin --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd
```

**Bước 2.** Train Teacher v3, rồi Student v3 KD. Student cần Teacher đã train xong. Mỗi epoch đánh giá 4 tập và lưu thêm `best_<tap>.pth.tar`.

```bash
python main/train.py --cfg ./config/train_teacher_v3_jotr_pf05.yml --exp_name teacher_v3_jotr_pf05 --gpu 0
python main/train.py --cfg ./config/train_student_v3_kd_jotr_pf05lift3dpw.yml --exp_name student_v3_kd_jotr_pf05lift3dpw_t2 --gpu 0
```

**Bước 3.** Đánh giá mesh một checkpoint trên 3DPW và OC/PC (cả chỉ-test lẫn toàn bộ):

```bash
python main/test_student_3dpw_sets.py --cfg ./config/train_student_v3_kd_jotr_pf05lift3dpw.yml \
  --ckpt experiment/student_v3_kd_jotr_pf05lift3dpw_t2/checkpoint/best.pth.tar --gpu 0
```

**Bản v3 với 2D Halpe (tham khảo):**
- Teacher: `config/train_teacher_v3_jotr.yml`
- Student scratch: `config/train_student_v3_scratch_jotr.yml` (biến thể `_full`, `_gt2d`)
- Student KD: `config/train_student_v3_kd_jotr.yml`

Dùng tên `--exp_name` mới khi tạo run mới. Khi tiếp tục một run đã có, dùng đúng tên cũ kèm `--resume_training`. Không so MPJPE 17 khớp của lifter trực tiếp với MPJPE 14 khớp của mesh.

**Refiner 2D (đang thử, đã code, chưa chạy):** ảnh + 2D detector → 2D đã sửa, đánh giá bằng lifter đóng băng.

```bash
python main/train_refiner2d.py --cfg config/train_refiner2d_jotr.yaml --exp_name refiner2d_jotr --gpu 0
python main/eval_refiner2d.py --cfg config/train_refiner2d_jotr.yaml \
  --ckpt experiment/refiner2d_jotr/checkpoint/last.pth.tar \
  --lifter_ckpt experiment/finetune_difflifter_d512_e15_jotr_pf05_3dpw/best_epoch.bin \
  --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --gpu 0
```

## Chẩn đoán cho v3

| Script | Mục đích |
|---|---|
| `main/diag_student_vs_lifter.py` | Lỗi tay chân của lifter so với Student theo nhóm 2D tốt/xa/thiếu; thử ảnh xám; oracle fixfar |
| `main/diag_kd_features.py` | Độ giống đặc trưng Student–Teacher (cos/L2) và tương quan với MPJPE |
| `main/diag_2d_crowd.py` | Lỗi 2D trên PC/Crowd, các biến thể halpe/dropfar/fixfar/gt, kiểm tra crop |
| `main/compare_2d_sources.py` | So Halpe/OpenPose/Higher-HRNet với GT 2D |
| `tools/check_jotr_equivalence.py` | Kiểm dữ liệu 2D khớp 100% với JOTR trước khi train |

```bash
python main/diag_student_vs_lifter.py --cfg ./config/train_student_v3_kd_jotr_pf05lift3dpw.yml \
  --ckpt_dir ./experiment/student_v3_kd_jotr_pf05lift3dpw_t2/checkpoint --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd
python main/diag_kd_features.py --cfg ./config/train_student_v3_kd_jotr_pf05lift3dpw.yml \
  --teacher ./experiment/teacher_v3_jotr_pf05/checkpoint/best.pth.tar \
  --students kd_t2=./experiment/student_v3_kd_jotr_pf05lift3dpw_t2/checkpoint/best_3dpw.pth.tar \
  --sets 3dpw,3dpw-crowd --max_samples 3000
python main/diag_2d_crowd.py --cfg ./config/train_student_v3_scratch_jotr.yml \
  --ckpt ./experiment/student_v3_scratch_jotr/checkpoint/best.pth.tar --gpu 0
```

Một số script chỉ chạy được với kiến trúc trước v3 (gọi trực tiếp `smpl_model` hoặc cần checkpoint cũ): `oracle_best_hypothesis.py`, `analyze_teacher.py`, `inspect_student_spread.py`, `train_selector.py`, `eval_selector.py`. Không dùng chúng cho checkpoint v3 khi chưa cập nhật.

## Nguồn gốc và ghi nhận

FuseKin phát triển từ ARTS. Phần lifter tham khảo D3DP và HyperDiff. Thiết kế v3 lấy ý tưởng từ 3DCrowdNet, JOTR, DPMesh, ReFit, PandaPose, SAM 3D Body, ScoreHypo và BlockGCN. Dữ liệu 2D khi train đồng bộ với JOTR (PoseFix).

README này mô tả FuseKin hiện tại, không phải bản phát hành chính thức của ARTS. Khi công bố kết quả, cần ghi nhận các thành phần kế thừa và tuân thủ giấy phép tương ứng. Công trình ARTS gốc:

```bibtex
@inproceedings{tang2024arts,
  title={ARTS: Semi-Analytical Regressor using Disentangled Skeletal Representations for Human Mesh Recovery from Videos},
  author={Tang, Tao and Liu, Hong and You, Yingxuan and Wang, Ti and Li, Wenhao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={1514--1523},
  year={2024}
}
```
