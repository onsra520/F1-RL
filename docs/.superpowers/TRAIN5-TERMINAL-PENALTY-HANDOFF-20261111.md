# TRAIN-5 terminal penalty: bàn giao ngày 11/11/2026

## Đúng nhánh và thứ tự đọc

- Repo: `onsra520/F1-RL`. Nhánh đang triển khai và nhận bàn giao: **`fix/physics-v4-handoff-audit`**. Checkout nhánh này và lấy commit mới nhất của nhánh; không mặc định dùng `main`.
- Nhánh `handoff/physics-v4-khanh-bang-20261106` là **nguồn handoff lịch sử**, không phải nhánh chứa thay đổi terminal-penalty mới. Không sửa hoặc force-push nhánh đó khi làm tiếp.
- Tài liệu này là entry point mới nhất cho tiến độ terminal penalty. `HANDOFF-TRAIN5-IMPLEMENTATION-STATUS-20261107.md`, `TRAIN5-V4-READINESS.md`, `TRAIN5-V4-REVIEW-PACKET-20261109.md` và `handoff/README.md` là tài liệu lịch sử/bối cảnh, không tự xác nhận trạng thái batch 11/11.
- Commit mã triển khai ban đầu: `0deb29cac09c88fe6e6e54cc6b481fe7d68b7df8`. Commit chứa tài liệu này nằm sau commit đó. Xem `git log` và CI cho **đúng HEAD**. Việc đổi tài liệu được Git theo dõi cũng làm proposal mới có source commit/hash mới; không dùng packet cũ.

## Phạm vi và kết quả đã có

Đây là **mô phỏng nội bộ trên năm TRAIN track**, không phải chứng nhận xe thật, độ chính xác sim-to-real, kết luận khoa học đã duyệt hay đánh giá HELD-OUT. Các gate và quyền HELD-OUT/scientific vẫn tách riêng.

Batch v4 trước đã hoàn tất sáu lượt (ba from-scratch, ba khởi tạo từ checkpoint đóng băng), mỗi lượt 200.704 bước và tổng 3.600 episode đánh giá TRAIN. Kết quả development **FAIL** hai ngưỡng: slalom hoàn thành 363/720 = 50,4% (yêu cầu từng track >= 80%); 315/3.600 = 8,75% va chạm vật lý được chứng nhận (giới hạn <= 1%). Nominal finish 90,11% >= 90%; không có episode physics invalid/uncertified. Bốn track khác có finish trên 99%. Đây là thống kê mô tả cho tập TRAIN đã dùng, không phải bằng chứng HELD-OUT.

Trong `handoff/penalty-20261111/` có báo cáo `prior-run/development-result.json`, journal đánh giá, sáu model kết quả v4 và bốn Step-3 witness. `SHA256SUMS` là hash SHA-256 theo byte gốc; `.gitattributes` giữ byte khi checkout. Các model `prior-run/models/` là **kết quả v4 cũ**, không phải kết quả penalty và không thay thế ba checkpoint gốc `models/C-seed0`, `models/B-seed1`, `models/A-seed2`. Kiểm tra hash trước khi nạp; chỉ nạp các tệp model từ nguồn tin cậy. Các packet đã ký, private key và passphrase không nằm trong gói này.

Với các ca tái hiện, crash terminal có collision reward 0 vì reward cũ chỉ tính prefix trước va chạm. Một wrapper thử nghiệm trừ thêm **5** đúng một lần cho terminal physical crash hợp lệ; giữ nguyên các kết thúc khác. Đây là thay đổi reward, không thay đổi geometry, physics, observation, action, track sampling hoặc HELD-OUT. Test replay một ca thật cho reward bước cuối 0,2541578465 -> -4,7458421535; không phải bằng chứng policy mới tốt hơn. Bảy rangefinder hiện chỉ dò vật cản, không đo mép đường; kiểm tra observation trước đó chỉ là tính nhất quán với encoder, không phải đối chiếu độc lập toàn phần.

## Batch penalty ngày 11/11: đã ký nhưng bị gián đoạn

Một packet owner đã được ký cho commit `0deb29c`, output trên máy KHANH BANG và proposal cũ. Lượt đầu `baseline / train5_from_scratch / replicate 0` ghi `started` lúc 00:58, sau đó Windows vào sleep lúc 00:58:49 và thức 07:43:22 (UTC+7). Không còn process, không có exit status hoặc traceback và không có model hoàn tất. Sleep trùng thời điểm gián đoạn; **nguyên nhân tiến trình biến mất chưa được chứng minh**. Ba checkpoint gốc đã được so SHA-256 và không đổi. Không coi lượt này là thành công, không xóa claim/journal, không khẳng định seed chưa từng được tiêu thụ.

Đường dẫn máy gốc để người sở hữu tìm log, không phải đường dẫn bắt buộc trên máy khác:

- `D:\handoffphysics-v4-khanh-bang-20261106\TRAIN5-penalty-paired-20261111\trial-journal.jsonl`
- `D:\handoffphysics-v4-khanh-bang-20261106\TRAIN5-penalty-execution-20261111.log` (rỗng), file `.status` không có.
- Packet và output cũ đã gắn commit, output path và nonce một lần. **Không copy packet cũ để chạy lại**, không xóa claim rồi chạy lại nguyên lệnh. Chữ ký cũ không ủy quyền cho HEAD, path hay máy mới.

## Thiết kế thử nghiệm và việc còn phải làm trước khi chạy

Proposal dự kiến so sánh cặp baseline/penalty trên **12 lượt**: hai cohort x ba replicate x hai nhánh reward. Trong mỗi cặp, seed PPO, sampling, reset và checkpoint khởi tạo giống nhau; track sampling giữ uniform. Ngân sách 200.704 bước thực tế/lượt, 8 vector env, 256 rollout steps, final-only; tổng 2.408.448 bước. Sau đó dự kiến **7.200 episode TRAIN** (hai nhánh x 3.600, dùng cặp environment/policy seed), stochastic CPU. Dùng kết quả theo từng policy/cell; không coi episode của cùng policy là các policy độc lập hay khẳng định ý nghĩa thống kê với ba replicate.

**Chưa hoàn thành so sánh penalty–baseline và chưa có model penalty thật.** Trước khi ký/chạy lại, người tiếp nhận phải:

1. Quyết định và ghi lại cách xử lý seed của lượt đã bắt đầu nhưng không hoàn tất. Hiện `scripts/wp7_penalty_trial_protocol.py` dùng domain seed cố định, vì vậy chạy `prepare` lại trên source hiện tại sẽ đề xuất lại cùng seed. Không giả định seed đó sạch; nếu đổi seed/domain phải cập nhật protocol, kiểm tra trùng với lịch sử có thể truy cập, review và chạy test/CI lại. Không tự áp dụng ngoại lệ 18 seed lịch sử cho seed mới.
2. Giữ bản cũ bất biến, chọn **output mới ngoài checkout**, bảo đảm dung lượng và máy duy trì nguồn/không sleep suốt training và đánh giá. Runner hiện không resume; chỉ lưu final model sau mỗi lượt, không có intermediate checkpoint. Một lần gián đoạn có thể mất lượt đang chạy. Kiểm tra thời hạn packet 168 giờ đủ cho kế hoạch thực tế; hết hạn thì guard dừng.
3. Kiểm tra lại implementation, các nhánh fail-closed và thời gian/thiết lập runtime trên máy thực thi. Không bỏ qua guard; xác nhận CI Windows/Linux theo HEAD và chạy lại local tests có witness. GitHub CI không có evidence ngoài máy, vì vậy một số kiểm thử trial phụ thuộc witness/journal được skip ở CI; CI xanh không chứng minh các witness đó đã được kiểm tra.
4. Đăng ký/xác thực public key của người ký theo quy trình project. Registry hiện tại thuộc owner `onsra520`; người vận hành khác **không tự động có quyền ký** và không được xin/chia sẻ private key hoặc passphrase của owner. Nếu thay key hoặc source phải cập nhật và kiểm thử bindings, commit/CI, tạo plan mới rồi owner hoặc key được ủy quyền tự ký. Thời hạn và các disclosure được hiển thị trong terminal.
5. So sánh baseline–penalty bằng tập seed đánh giá mới đã khóa trong proposal; kiểm tra cả năm track và va chạm, không chọn lọc kết quả. Không thử tăng tỷ trọng slalom cho tới khi so sánh này xong và có thiết kế riêng. HELD-OUT cần quyết định riêng.

Các giới hạn không mất đi: 15 log GitHub lịch sử và replay C2 lịch sử chưa khôi phục; global non-use của seed mới chưa được chứng minh; review độc lập chưa được xác lập. Đánh giá và training ở đây chỉ là nội bộ.

## Chuẩn bị trên máy người tiếp nhận

Từ repo root, tạo môi trường Python 3.11 trở lên và cài trực tiếp các phụ thuộc bằng `python -m pip install "gymnasium>=1.3,<2" "numpy>=2,<3" "cryptography>=50.0.2,<51" "pytest>=8,<10" "stable-baselines3>=2.9,<2.10" "PyYAML>=6,<7" bcrypt` (bcrypt cần để đọc khóa OpenSSH có passphrase). Chạy các lệnh Python từ repo root; hiện không dùng `pip install -e .` vì `pyproject.toml` tham chiếu README.md ở root nhưng file đó không có trong checkout. Bảo đảm `git status --short` sạch. Kiểm tra file trong `handoff/penalty-20261111/SHA256SUMS` theo byte, đặc biệt bốn witness và ba checkpoint gốc theo `configs/research/wp7-train5-paired-v2-registration.json` / `wp7_train5_v4_checkpoint.py`.

Implementation hiện tìm hai tệp seed evidence **ngoài checkout tại thư mục cha**. Sao chép byte nguyên vẹn, không sửa nội dung:

- `handoff/penalty-20261111/seed-evidence/slalom-fresh-audit.json` -> `<repo-parent>/slalom-fresh-audit.json`.
- `handoff/penalty-20261111/seed-evidence/evaluation-journal.jsonl` -> `<repo-parent>/TRAIN5-v4-internal-20261109/evaluation-journal.jsonl`.

Bốn witness nằm trong `handoff/penalty-20261111/step3-witnesses/`; truyền đường dẫn thư mục đó bằng `--evidence`. `prepare` kiểm tra witness và ba checkpoint. Giữ output, plan và packet ngoài checkout, mỗi path mới và riêng biệt. Các lệnh dưới đây là **mẫu sau khi đã giải quyết seed bị gián đoạn và có commit/CI mới**; chúng không tự cấp quyền:

```text
python -B -m scripts.wp7_penalty_trial_cli prepare --output <NEW_OUTPUT> --plan <NEW_PLAN.json> --evidence <STEP3_WITNESSES_DIR>
python -B -m scripts.wp7_penalty_trial_cli sign --plan <NEW_PLAN.json> --private-key <OWNER_PRIVATE_KEY_PATH> --packet <NEW_PACKET.json>
python -B -m scripts.wp7_penalty_trial_cli execute --packet <NEW_PACKET.json> --evidence <STEP3_WITNESSES_DIR>
```

`sign` hiển thị scope, yêu cầu người giữ key gõ `DONG Y TRAIN5 PENALTY` và nhập passphrase tại terminal; không gửi passphrase qua chat. `execute` kiểm tra chữ ký và tạo output độc quyền, sau đó lần lượt 12 lượt rồi đánh giá 7.200 episode. Dùng log process và `trial-journal.jsonl` để theo dõi; `training-result.json` chỉ có sau 12 lượt, `paired-evaluation-journal.jsonl` và `paired-evaluation-result.json` thuộc bước sau. Kiểm tra model hash, số bước và ba checkpoint gốc; không tuyên bố thành công chỉ vì process bắt đầu. Khi có kết quả, báo cáo cả điểm không đạt và những giới hạn ở trên.
