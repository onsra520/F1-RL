# PLAN - Chạy lại TRAIN-5 terminal penalty (bản cho Claude Code / Codex)

- Ngày lập: 11/11/2026. Người điều phối và quyết định: Trần Anh Kiệt.
- Repo local: `D:\REL301m\F1-RL` (remote `onsra520/F1-RL`).
- Nhánh gốc: `fix/physics-v4-handoff-audit` @ `21a37dd` (CI Windows + Linux đã đạt, run 42936763142).
- Nhánh làm việc mới: `kiet/physics-v4-penalty-rerun` (tách từ `21a37dd`).
- Tài liệu canonical phải đọc trước: `docs/.superpowers/TRAIN5-TERMINAL-PENALTY-HANDOFF-20261111.md`.
- Bản plan này được commit tại `docs/.superpowers/PLAN-TRAIN5-PENALTY-RERUN-KIET-20261111.md` trên nhánh `kiet/physics-v4-penalty-rerun` để owner đọc và duyệt Gate G2. Commit xong trước khi chạy `prepare`.

## 0. Quy tắc cứng

1. Không sửa, không force-push `fix/physics-v4-handoff-audit`, `handoff/physics-v4-khanh-bang-20261106`, `main`. Chỉ commit trên nhánh `kiet/...`.
2. Không HELD-OUT. Không đổi physics, geometry, observation, action, track sampling, mức penalty (5), ngân sách (200.704 bước/lượt, 8 env).
3. Không xóa, không sửa code guard/chữ ký/capability (`wp7_penalty_trial_authorization.py`, `wp7_penalty_trial_capability.py`, các kiểm tra trong `cli.py`, `runner.py`). Xem Gate G2.
4. Không dùng lại packet, claim, journal cũ của Khánh Băng. Không sửa 3 checkpoint gốc `models/C-seed0`, `B-seed1`, `A-seed2`.
5. Trong lúc `execute` đang chạy: KHÔNG commit, checkout, pull, sửa file trong checkout. Mỗi lượt đều dựng lại proposal từ source; checkout bẩn hoặc HEAD đổi là lượt kế tiếp chết và batch không resume được.
6. Output, plan, packet, log đều nằm ngoài checkout, đường dẫn mới chưa tồn tại.
7. Không bịa kết quả. Mọi con số báo cáo phải lấy từ file output; nêu cả điểm không đạt.
8. Commit message tiếng Anh. File tiếng Việt giữ UTF-8, dùng dấu "-" ASCII.
9. Gặp tình huống ngoài plan (test fail không rõ nguyên nhân, seed va chạm, pin hash fail): DỪNG và báo Kiệt, không tự mở rộng phạm vi.

## 1. Phase A - Chuẩn bị (không cần ai duyệt)

A1. Kiểm tra trạng thái: `git fetch origin`; xác nhận `origin/fix/physics-v4-handoff-audit` vẫn là `21a37dd`. Nếu đã có commit mới: dừng, báo Kiệt (đồng đội cập nhật thường xuyên).
A2. `git switch -c kiet/physics-v4-penalty-rerun 21a37dd`.
A3. Môi trường: tạo venv NGOÀI checkout (ví dụ `D:\REL301m\.venv-f1rl`, Python 3.12.10 sẵn có), cài đúng lệnh trong tài liệu bàn giao:
    `python -m pip install "gymnasium>=1.3,<2" "numpy>=2,<3" "cryptography>=50.0.2,<51" "pytest>=8,<10" "stable-baselines3>=2.9,<2.10" "PyYAML>=6,<7" bcrypt`
    Không `pip install -e .`. Luôn chạy `python -B` từ repo root.
A4. Kiểm tra hash: mọi file trong `docs/.superpowers/handoff/penalty-20261111/SHA256SUMS` và `handoff/evidence/SHA256SUMS` theo byte. Fail thì dừng.
A5. Sao chép seed evidence (byte nguyên vẹn, rồi so SHA-256 với bản gốc):
    - `...\penalty-20261111\seed-evidence\slalom-fresh-audit.json` -> `D:\REL301m\slalom-fresh-audit.json`
    - `...\penalty-20261111\seed-evidence\evaluation-journal.jsonl` -> `D:\REL301m\TRAIN5-v4-internal-20261109\evaluation-journal.jsonl`
A6. Baseline test TRƯỚC khi sửa: chạy đúng danh sách test trong `.github/workflows/*.yml`, cộng 4 file `test_wp7_penalty_trial_*.py` (giờ không còn bị skip vì đã có seed evidence và witness). Ghi lại số pass/skip/fail. Có fail ở baseline: dừng, báo.
A7. `git status --short` phải rỗng sau A3-A6 (venv, evidence copy đều ngoài checkout).

## 2. Phase B - Sửa seed (quyết định của Kiệt: domain mới cho cả 12 lượt)

File: `scripts/wp7_penalty_trial_protocol.py`.

B1. Đổi `DOMAIN` thành `'F1RL-TERMINAL-PENALTY-TRIAL-20261111-v2'`.
B2. Giữ hằng `RETIRED_DOMAINS = ('F1RL-TERMINAL-PENALTY-TRIAL-20261111-v1',)`. Cho `_seed` nhận domain tường minh. Trong `build_proposal`, trước khi cấp seed mới, tính lại TOÀN BỘ seed của v1 bằng đúng công thức cũ và đưa vào `reserved`:
    - train: 2 cohort x 3 replicate x 3 role (`ppo`, `sampling`, `perturbation`) cộng dải `ppo .. ppo+7`;
    - eval: `env` và `policy` cho 6 cặp x 5 track x 2 condition x 60 episode.
B3. Không chọn seed bằng tay. Nếu v2 va chạm (code đã raise `ValueError`): dừng, báo Kiệt; phương án dự phòng là v3 theo cùng cách.
B4. Thêm field mô tả vào proposal, ví dụ `'retired_seed_domains': [...]`, để packet ghi rõ quyết định. Lưu ý: field mới làm đổi hash proposal - đúng ý, packet cũ không được dùng lại.
B5. Test (thêm vào `tests/research/test_wp7_penalty_trial_protocol.py`, file này đã nằm trong `EXTRA_SOURCES`):
    - seed train + eval của v2 rời hoàn toàn với tập seed v1;
    - cụ thể seed của lượt bị gián đoạn (`train5_from_scratch`, replicate 0, 3 role + 8 rank ppo, domain v1) không xuất hiện trong proposal;
    - các assert cũ (12 lượt, 2.408.448 bước, 3.600 hàng, 7.200 episode) vẫn đúng.
B6. CI: workflow chỉ chạy trên 2 nhánh cũ và có bước chặn tên nhánh. Thêm `kiet/physics-v4-penalty-rerun` vào `on.push.branches` và vào `case` của bước "Enforce candidate branch". Không đổi gì khác trong workflow.
    Rủi ro cần kiểm tra: `scripts/wp7_train5_handoff_preflight.py` ghim 44 blob và các manifest review ghim hash source. Nếu workflow hoặc file protocol nằm trong danh sách ghim thì preflight sẽ fail - khi đó DỪNG và báo, không tự sửa pin.
B7. Tài liệu: thêm mục "Cập nhật 11/11 - Kiệt tiếp nhận" vào `TRAIN5-TERMINAL-PENALTY-HANDOFF-20261111.md`, nêu rõ:
    - nhánh `kiet/physics-v4-penalty-rerun`, tách từ `21a37dd`;
    - quyết định seed: bỏ toàn bộ domain v1 (coi như đã tiêu thụ, kể cả phần chưa chạy), dùng v2, lý do là giữ ghép cặp baseline/penalty sạch; lượt `baseline/train5_from_scratch/0` của v1 đã `started` và không hoàn tất;
    - giới hạn giữ nguyên: non-use toàn cục của seed mới chưa được chứng minh, chưa có review độc lập, chưa HELD-OUT;
    - cách ủy quyền thực tế đã dùng (điền sau Gate G2).
B8. Chạy lại toàn bộ test như A6. Commit (1 commit code+test+workflow, 1 commit docs), push nhánh `kiet/...`, chờ CI xanh trên đúng HEAD. Ghi lại SHA.

## 3. Gate G2 - Ủy quyền chạy (DỪNG ở đây, chờ Kiệt xác nhận)

Lệnh `sign` chỉ nhận private key khớp `configs/research/wp7-train5-internal-owner-key-v1.json` (key của owner Onsra). Guard này do owner đặt; tài liệu bàn giao ghi "Không bỏ qua guard". Agent không được gỡ hoặc vô hiệu hóa nó. Hai đường hợp lệ, đều chỉ cần owner trả lời một lần:

- G2-A (ưu tiên, không đổi code): Kiệt chạy `prepare` tạo `plan.json`, gửi cho Onsra. Onsra checkout đúng commit, đặt 2 file seed evidence ở thư mục cha, chạy `sign`, gửi lại `packet.json`. Điều kiện: chuỗi `output_root` trong plan phải resolve giống nhau trên hai máy (cùng ổ đĩa, đường dẫn tuyệt đối Windows) và chưa tồn tại trên máy Onsra. Packet hết hạn sau 168 giờ.
- G2-B (nếu Onsra không ký kịp): Onsra xác nhận bằng văn bản (chat nhóm hoặc comment trên commit) cho phép ghi public key Ed25519 của Kiệt vào file owner-key TRÊN NHÁNH `kiet/...`. Khi đó: Kiệt tự tạo cặp khóa có passphrase, agent chỉ thay `owner_public_key_hex`, chạy lại test + CI, ghi rõ trong tài liệu bàn giao ai duyệt và khi nào. Agent không bao giờ đọc, nhận hay lưu private key và passphrase.
  Rủi ro G2-B: các manifest review v4 ghim hash file owner-key; đổi file có thể làm fail test/preflight. Fail thì dừng, báo.

Không có xác nhận của owner thì không sang Phase C. Phase A, B vẫn làm trọn để không mất thời gian.

## 4. Phase C - Chạy

C1. Nguồn điện (Kiệt tự làm, đây là thay đổi cấu hình hệ thống): cắm sạc suốt quá trình; tắt sleep/hibernate khi cắm điện (`powercfg /change standby-timeout-ac 0`, `powercfg /change hibernate-timeout-ac 0`), đặt hành vi gập máy là "Do nothing". Hiện máy đang sleep sau 5 giờ (AC) và 30 phút (pin) - đúng nguyên nhân nghi ngờ của lần gián đoạn trước. Tạm hoãn Windows Update restart.
C2. Đường dẫn (ví dụ, tất cả phải chưa tồn tại):
    - output: `D:\REL301m\F1RL-runs\penalty-v2-20261111`
    - plan/packet/log: `D:\REL301m\F1RL-runs\penalty-v2-20261111-plan.json`, `...-packet.json`, `...-execution.log`
    - evidence: `D:\REL301m\F1-RL\docs\.superpowers\handoff\penalty-20261111\step3-witnesses`
    Kiểm tra dung lượng trống ổ D.
C3. `prepare` -> (G2) `sign` -> `execute`, đúng 3 lệnh mẫu trong tài liệu bàn giao. `execute` chạy trong một terminal riêng, redirect stdout+stderr ra log, ghi exit code ra file `.status`.
C4. Go/no-go thời gian: chưa ai đo thời gian một lượt. Sau khi lượt 1 ghi `completed` trong `trial-journal.jsonl`, lấy thời gian lượt 1 x 12 + ước lượng eval. Nếu vượt hạn 168 giờ của packet hoặc vượt hạn Kiệt đặt: báo ngay, không tự đổi ngân sách.
C5. Theo dõi bằng cách ĐỌC `trial-journal.jsonl` và log. Không đụng vào checkout (quy tắc 5).
C6. Nếu gián đoạn: không xóa claim, không chạy lại nguyên lệnh. Ghi nhận hiện trạng, báo Kiệt. Chạy lại đồng nghĩa domain seed mới (v3) + packet mới.

## 5. Phase D - Kiểm tra và báo cáo

D1. Điều kiện hoàn tất: 12 dòng `completed`, mỗi lượt đúng 200.704 bước; có `training-result.json`; `paired-evaluation-journal.jsonl` đủ 7.200 dòng `completed`, 0 dòng `failed`; có `paired-evaluation-result.json`; SHA-256 của 12 model khớp journal; 3 checkpoint gốc không đổi; exit code 0.
D2. Báo cáo theo từng intervention x cohort x replicate x track x condition (không gộp chọn lọc): tỉ lệ finish, tỉ lệ `certified_physical_crash`, số episode invalid/uncertified. Đối chiếu ngưỡng cũ: mỗi track finish >= 80%, crash <= 1%, nominal finish >= 90%. Mốc so sánh: batch v4 cũ có slalom 363/720 và crash 315/3.600.
D3. Diễn giải đúng mức: 3 replicate, thống kê mô tả trên tập TRAIN, không tuyên bố ý nghĩa thống kê, không phải bằng chứng HELD-OUT.
D4. Sau khi process đã kết thúc hẳn: copy kết quả vào `docs/.superpowers/handoff/penalty-v2-<ngày>/` kèm `SHA256SUMS` và `.gitattributes` giữ byte (theo mẫu thư mục `penalty-20261111`), cập nhật tài liệu bàn giao, commit, push, chờ CI. Không commit packet, private key, passphrase.
D5. Báo Kiệt: kết quả, bằng chứng (đường dẫn file + SHA), giới hạn, bước tiếp theo.

## 6. Rủi ro còn lại (đã tự rà)

- Thời gian chạy chưa biết; có C4 làm điểm quyết định sớm.
- Runner không resume, không checkpoint trung gian: một lần mất điện là mất lượt đang chạy và phải đổi seed. Plan không thêm tính năng resume vì đó là thay đổi runner ngoài phạm vi.
- Pin hash (preflight 44 blob, manifest review) có thể chặn B6 hoặc G2-B; đã có điểm dừng.
- Nhánh `fix/physics-v4-handoff-audit` có thể nhận commit mới trong lúc làm; A1 kiểm tra trước, sau đó không tự rebase - báo Kiệt.
- 2 commit chỉ có trên nhánh handoff lịch sử (`179a255`, `6e30ebd`) là commit lưu trữ evidence WP3 (docs), không chứa code penalty; không cần đưa sang.
- Penalty 5 có thể không cải thiện slalom; kết quả âm vẫn phải báo nguyên trạng.
