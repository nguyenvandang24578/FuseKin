# AGENTS.md — quy tắc làm việc trong FuseKin

Áp dụng cho toàn repo, gồm DiffLifter và mô hình mesh. Cập nhật **2026-10-22**.

## Tài liệu cần đọc

- [README.md](README.md): tổng quan, cấu trúc, lệnh chạy.
- [01_PROJECT_CONTEXT.md](01_PROJECT_CONTEXT.md): kiến trúc và quy ước hiện tại.
- [experiment.md](experiment.md): kết quả và tiến độ. Đọc phần liên quan trước khi đề xuất thí nghiệm hoặc phân tích log.

Không tạo thêm file kế hoạch MD cho từng ý tưởng nhỏ. Cập nhật đúng một trong các file trên; source/YAML thực tế là căn cứ khi tài liệu mâu thuẫn với code.

## Cách làm việc bắt buộc

1. Trả lời **tiếng Việt, ngắn gọn, dễ hiểu**. Comment mới trong code viết tiếng Việt **không dấu**.
2. **Không chạy code dự án, không cài thư viện, không train/test**, kể cả import model hoặc kiểm cú pháp bằng Python. Chỉ đọc/sửa file; người dùng chạy trên server GPU. Sau khi sửa, đọc lại code/diff và đưa lệnh chạy chính xác.
3. **KHÔNG viết/sửa code khi chưa có xác nhận rõ ràng của người dùng** (áp dụng cho MỌI yêu cầu, kể cả việc nhỏ hay khi người dùng chỉ đang bàn ý tưởng). Trước tiên nêu kế hoạch ngắn (file/hàm sẽ đổi, YAML mới, cách chạy) rồi chờ người dùng đồng ý; chỉ sửa sau khi được duyệt, và chỉ trong phạm vi đã duyệt. Khi đã được duyệt thì hoàn tất phạm vi đó, không hỏi lại cho từng bước. (Quy tắc người dùng đặt 2026-10-30.)
4. Đọc file thật trước khi sửa; không đoán tên hàm, key, checkpoint hoặc đường dẫn. Nêu rõ file/hàm đã đổi. Không ghi đè thay đổi sẵn có của người dùng.
5. Tính năng mới phải có key trong `lib/core/config.py`, mặc định **tắt** để giữ hành vi cũ. YAML chỉ dùng key đã khai báo. Giữ tương thích đường xử lý/checkpoint đang được hỗ trợ.
6. Mỗi thí nghiệm chỉ đổi **một yếu tố**. Tạo YAML và thư mục kết quả mới; không sửa YAML cũ hoặc thay run đang train trên server.
7. Đổi kiến trúc lifter phải đồng bộ constructor, `finetune_difflifter.save_ckpt/load_ckpt`, `diag_difflifter.build_model` và `ARTS._build_diff_lifter`. Inspector dùng chung builder với diag. Không bỏ qua key thiếu/thừa để giả vờ nạp được backbone khác.
8. Khi đọc log, so với mốc **cùng giao thức** trong `experiment.md`, nêu kết luận và bước tiếp. Nếu lập luận trước sai thì nói thẳng. Không coi code đã viết là thí nghiệm đã chạy.
9. Phân biệt kết quả người dùng báo, kết quả lịch sử, giả thuyết và kết quả mới kiểm chứng. Không suy ra ý nghĩa thống kê chỉ từ chênh lệch nhỏ của một run.
10. Không chỉnh repo tham khảo hoặc README bên trong `ARTS/`, `D3DP/`, `HyperDiff/`, `MotionBERT/` khi chưa thuộc phạm vi được yêu cầu.

## Quy ước không được đổi tùy tiện

- Joint lifter: **H36M-17**, thứ tự và cha trong `H36M_PARENTS`; GT/output **mét**, root-relative tại pelvis.
- Điều kiện DiffLifter: `(B,17,6)` = `[x, y, conf, cam_bbox(3)]`. 2D chuẩn hóa theo crop `output_hm_shape` về [-1,1], **không trừ root 2D**. Khớp mask=0 có tọa độ 0.
- Camera bbox tính trong `lib/utils/cam_bbox.py`; khi train/infer phải cùng quy ước flip, rotation và camera ước lượng.
- Phân biệt **2D confidence liên tục** cho lifter và **mask hợp lệ** cho PE2D của fusion. Kiểm tra `DATASET.lift_soft_conf` khớp checkpoint.
- 3DPW-OC/PC phải dùng file Halpe/ann_id đúng split. Không dùng file test để tra annotation của OC/PC.
- Không trộn đơn vị/root của target mesh với quy ước lifter; pseudo-GT NeuralAnnot không phải GT mocap.
- Loss diffusion hiện là MPJPE L2 norm trên `x0`, không bình phương; giữ nguyên trừ khi được duyệt đổi riêng.

## Đánh giá và checkpoint

| Kết quả | Quy ước |
|---|---|
| Lifter / oracle pose input | H36M-17 root-relative, MPJPE mm |
| Mesh qua `dataset.evaluate` | 14 khớp, MPJPE/PA-MPJPE/MPVPE mm |
| Checkpoint lifter để đánh giá | **`best_epoch.bin`**, không lấy latest làm mốc |
| Checkpoint mesh để đánh giá | **`checkpoint/best.pth.tar`**, đúng YAML của run |

Không so trực tiếp hai loại MPJPE. Luôn đọc giá trị YAML thật để biết tập train và tập chọn best; không tin comment “val” nếu key đang là `3dpw`. Không chọn/điều chỉnh model theo test cho giao thức đánh giá cuối. Khi phân tích thí nghiệm hiện có chọn theo test, ghi rõ giới hạn đó, không tự đổi run đã được người dùng cho chạy.

Mục tiêu giao thức JOTR đòi hỏi xét **toàn pipeline**: lifter không train 3DPW chưa đủ nếu Student vẫn train 3DPW. OC/PC/Crowd cần kiểm tra giao nhau với cả tập train lẫn tập chọn checkpoint. Oracle dùng GT chỉ đo tiềm năng.

## Duy trì tài liệu

Sau thay đổi đáng kể, cập nhật kiến trúc trong `01_PROJECT_CONTEXT.md` và trạng thái trong `experiment.md`. Chỉ chuyển “đã code/chờ chạy” sang “đã chạy/có kết quả” khi nhận được bằng chứng từ người dùng. Không nhân bản lịch sử thí nghiệm vào AGENTS.
