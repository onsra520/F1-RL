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

1. Lượt cũ đã bắt đầu được xem là đã tiêu thụ seed. Domain mới `F1RL-TERMINAL-PENALTY-TRIAL-20261111-v2` tạo seed khác; đối chiếu toàn bộ 7.260 giá trị gồm PPO rank, sampling, perturbation và evaluation với plan v1 cho kết quả không trùng. Proposal vẫn kiểm tra lịch sử seed có thể truy cập; không chứng minh được global non-use và không tự áp dụng ngoại lệ 18 seed lịch sử cho seed mới. Chạy test/CI cho commit mới trước khi ký.
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

Bốn witness nằm trong `handoff/penalty-20261111/step3-witnesses/`; truyền đường dẫn thư mục đó bằng `--evidence`. `prepare` kiểm tra witness và ba checkpoint. Giữ output, plan và packet ngoài checkout, mỗi path mới và riêng biệt. Các lệnh dưới đây là **mẫu sau khi có commit/CI mới cho domain v2**; chúng không tự cấp quyền:

```text
python -B -m scripts.wp7_penalty_trial_cli prepare --output <NEW_OUTPUT> --plan <NEW_PLAN.json> --evidence <STEP3_WITNESSES_DIR>
python -B -m scripts.wp7_penalty_trial_cli sign --plan <NEW_PLAN.json> --private-key <OWNER_PRIVATE_KEY_PATH> --packet <NEW_PACKET.json>
python -B -m scripts.wp7_penalty_trial_cli execute --packet <NEW_PACKET.json> --evidence <STEP3_WITNESSES_DIR>
```

`sign` hiển thị scope, yêu cầu người giữ key gõ `DONG Y TRAIN5 PENALTY` và nhập passphrase tại terminal; không gửi passphrase qua chat. `execute` kiểm tra chữ ký và tạo output độc quyền, sau đó lần lượt 12 lượt rồi đánh giá 7.200 episode. Dùng log process và `trial-journal.jsonl` để theo dõi; `training-result.json` chỉ có sau 12 lượt, `paired-evaluation-journal.jsonl` và `paired-evaluation-result.json` thuộc bước sau. Kiểm tra model hash, số bước và ba checkpoint gốc; không tuyên bố thành công chỉ vì process bắt đầu. Khi có kết quả, báo cáo cả điểm không đạt và những giới hạn ở trên.

## Cập nhật 11/11 - Kiệt tiếp nhận

- Nhánh làm việc: `kiet/physics-v4-penalty-rerun`. Nền: `origin/fix/physics-v4-handoff-audit` @ `224d6ff` (Khánh Băng, "Use fresh seed domain after interrupted trial"). Nền ban đầu `21a37dd` đã được rebase lên `224d6ff`.
- Quyết định seed: dùng `DOMAIN = F1RL-TERMINAL-PENALTY-TRIAL-20261111-v2` theo `224d6ff`, có test đối chiếu với seed v1. Domain v1 coi như đã tiêu thụ, kể cả phần chưa chạy. Lý do: giữ ghép cặp baseline/penalty sạch. Lượt `baseline/train5_from_scratch/0` của v1 đã `started` và không hoàn tất. Plan bước B1-B5 đã được `224d6ff` thay thế; không sửa thêm `scripts/wp7_penalty_trial_protocol.py`.
- Test `test_wp7_penalty_trial_fake_batch.py` cần bản sao byte của `penalty-20261111/step3-witnesses/` đặt ngoài checkout tại `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`. Sau khi copy, test pass. Đây là điều kiện chạy môi trường, không phải thay đổi test.
- CI: workflow thêm nhánh `kiet/physics-v4-penalty-rerun` vào `on.push` và danh sách cho phép. Pin hash trong `configs/research/wp7-train5-v4-run-manifest-review-draft.json` được cập nhật đúng hai mục: owner-key và workflow.
- Ủy quyền (phương án B): Owner Onsra (Tiến) chọn phương án B, Kiệt chuyển lại từ Zalo ngày 11/11/2026. Agent không tự xác minh được phản hồi này. Phạm vi: chỉ thay `owner_public_key_hex` trong `configs/research/wp7-train5-internal-owner-key-v1.json`, chỉ có hiệu lực trên nhánh `kiet/...`. File key này cũng là key của luồng batch/development v4 cũ, nên trên nhánh này key của Kiệt thay owner ở các luồng đó. Plan này chỉ chạy penalty trial.
- Key đã đăng ký: fingerprint OpenSSH `SHA256:1l21uIA631AsYG64A72A7psS1YhJRh5XQhOksYO9Vpo`, `owner_public_key_hex` `222be74baeba66d5c7f6c076acca16a2b04a31da71bf271b3bc18a6d5f34c8bc`. Key cũ của owner có fingerprint `SHA256:6S0rHCQyWkbb/aV+GyxpAm4nSjDPbtdmD7Eh9CYZrVY` được thay trên nhánh này.
- Người duyệt phương án B: Tiến (Onsra), chuyển lại qua Kiệt. Dấu vết ngoài repo (ảnh chụp Zalo hoặc comment của Tiến) chưa có trong tài liệu này; cần bổ sung.
- Giới hạn còn nguyên: non-use toàn cục của seed v2 chưa được chứng minh; chưa có review độc lập; chưa HELD-OUT; không tuyên bố ý nghĩa thống kê với 3 replicate. Đây là đánh giá nội bộ trên TRAIN.

## Cập nhật 11/11 chiều - chạy song song

- Quyết định của Kiệt (không chờ duyệt): bỏ bước ký Ed25519 trên nhánh này. Nguồn xác nhận: Tiến (chủ repo) trả lời qua chat 11/11 (11:05-11:16), ghi "tiến xác nhận ko cần ký nữa"; Kiệt chuyển lại. Agent không kiểm chứng được nguồn này ngoài nội dung chat do Kiệt cung cấp. Chữ ký cũ không còn là điều kiện chạy; `verify_trial_scope` chỉ kiểm tra ràng buộc proposal/source và output nằm ngoài checkout. Module `wp7_penalty_trial_authorization.py` cũ vẫn còn, không được gọi trong đường chạy trial.
- Seed domain v3 cho cả 12 lượt và toàn bộ seed đánh giá. Seed của v1 và v2 (train và eval) được đưa vào tập dành riêng, có test chứng minh v3 rời hoàn toàn với v1 và v2. Lý do: v2 đã tiêu thụ một phần (lượt 1 xong, lượt 2 dừng).
- Trạng thái batch v2 (`D:\REL301m\F1RL-runs\penalty-v2-20261111`): lượt 1 (baseline/train5_from_scratch/0) hoàn tất lúc 13:16, 200.704 bước; lượt 2 đã bắt đầu. Output v2 giữ nguyên, KHÔNG dùng cho so sánh. Ghi lại giờ dừng và dòng journal cuối khi dừng ở bước E3.
- Cách chạy mới: mỗi lượt một process, `torch.set_num_threads(1)` trong mỗi process; đánh giá chia 30 shard theo (cohort, replicate, track), ghép lại đúng thứ tự seed. Cấu hình khoa học không đổi: 12 lượt, 200.704 bước/lượt, 8 env `DummyVecEnv`, PPO không đổi, 5 track TRAIN, penalty 0 và 5, 60 episode x 2 condition, không HELD-OUT.
- Giới hạn giữ nguyên: non-use toàn cục của seed mới chưa được chứng minh; chưa có review độc lập; chưa HELD-OUT; không còn chữ ký ủy quyền trong đường chạy.
- Runner không resume. Mất điện hoặc sleep làm mất các lượt đang chạy; khi đó phải đổi domain (v4) và packet mới.

## Kết quả batch v3 (penalty-v3-20261111) - mô tả, chưa kiểm định

- Batch đã chạy xong: execute `EXIT=0`, 12 lượt, 7.200 bản ghi đánh giá. Source commit `5c7f832`. Plan: `D:\REL301m\F1RL-runs\penalty-v3-20261111-plan.json`. Output: `D:\REL301m\F1RL-runs\penalty-v3-20261111\` (không sửa sau khi chạy xong).
- Phân tích chỉ đọc ngày 12/11. Số liệu và script nằm ngoài repo tại `D:\REL301m\F1RL-runs\penalty-v3-analysis-20261112\` (`analysis-report.md`, `analysis-summary.json`, `cells.csv`, `models.csv`, `model_pair_diffs.csv`, `analyze_v3.py`).
- Đây là thống kê mô tả trên tập TRAIN. Không có kiểm định thống kê (D3). Không phải bằng chứng HELD-OUT.

### Kết luận theo ngưỡng plan

Penalty 5 không đạt cả ba ngưỡng. Baseline cũng không đạt cùng các ngưỡng đó.

Ngưỡng dùng theo handoff, với mẫu số khớp batch v4: finish từng track gộp hai condition (720 episode mỗi track mỗi intervention) phải từ 80% trở lên; certified crash toàn bộ (3.600) không quá 1%; nominal finish gộp các track (1.800) phải từ 90% trở lên.

| Chỉ số | Baseline | Penalty 5 | Ngưỡng | Kết quả penalty |
|---|---|---|---|---|
| Slalom finish (720) | 344/720 = 47,8% | 396/720 = 55,0% | >= 80% | không đạt |
| Certified crash toàn bộ (3.600) | 346/3.600 = 9,6% | 301/3.600 = 8,4% | <= 1% | không đạt |
| Nominal finish (1.800) | 1.601/1.800 = 88,9% | 1.598/1.800 = 88,8% | >= 90% | không đạt |

Finish gộp hai condition của bốn track còn lại, penalty: straight 95,1%, right-turn 98,5%, ramp 96,1%, left-turn 97,5%. Baseline tương ứng: 100%, 100%, 99,0%, 99,7%.

### Chênh lệch ghép cặp (penalty trừ baseline, cùng seed môi trường và policy)

| Phạm vi | Δ finish (điểm %) | Δ crash (điểm %) |
|---|---|---|
| Tổng thể (3.600) | -0,9 | -1,3 |
| Nominal (1.800) | -0,2 | -1,9 |
| Perturbation (1.800) | -1,6 | -0,6 |
| Slalom gộp (720) | +7,2 | -5,6 |

Trên 6 cặp (cohort x replicate), chênh lệch finish slalom có trung bình +7,2 điểm, SD 31,7, nhỏ nhất -31,7, lớn nhất +57,5. Ba cặp penalty tốt hơn baseline, ba cặp tệ hơn. Không có hướng nhất quán giữa các lượt train. Không mô tả penalty là cải thiện slalom.

### Hai điểm mơ hồ của plan (chờ nhóm chốt, chưa chọn cách đọc)

1. Ngưỡng crash không quá 1%: áp dụng cho tổng thể (handoff) hay từng track (D2 ghi "mỗi track")?
   - Tổng thể: baseline 9,6%, penalty 8,4%. Không đạt.
   - Từng track, gộp condition: slalom baseline 47,1%, penalty 41,5%. Ramp baseline 0,97%, penalty 0,28%. Straight, right-turn, left-turn: 0% cả hai. Slalom không đạt.
   - Từng track x condition: baseline ramp perturbation 1,1% (không đạt); penalty ramp perturbation 0,6%. Slalom không đạt cả hai condition.
2. Ngưỡng finish từ 80% trở lên: gộp hai condition (handoff, mẫu số 720) hay tính từng track x condition (D2)?
   - Gộp condition: slalom baseline 47,8%, penalty 55,0%. Không đạt.
   - Từng track x condition: slalom nominal baseline 45,8%, penalty 56,4%; slalom perturbation baseline 49,7%, penalty 53,6%. Không đạt. Bốn track còn lại đạt ở cả hai condition, trừ đã ghi ở trên.

Ở mọi cách đọc, slalom không đạt cả ngưỡng finish lẫn ngưỡng crash. Penalty không đạt ngưỡng nominal finish ở mọi cách đọc.

### Quan sát ngoài plan (mô tả, chưa kết luận)

Plan không đo các mục dưới đây. Ghi lại để nhóm đánh giá. Không dùng chúng để kết luận gate.

- `hard_route_violation` (không tính là finish, cũng không là crash): baseline 39, penalty 115, trên 3.600 episode mỗi intervention.
- Theo track, penalty: straight-sprint 35 (baseline 0), ramp 26 (baseline 0), left-turn 18 (baseline 2), right-turn 11 (baseline 0), slalom 25 (baseline 37).
- Theo model, 110 trên 115 lượt route violation của penalty nằm ở ba lượt: `train5_from_frozen_checkpoint` rep 0 (59), `train5_from_scratch` rep 1 (34), `train5_from_scratch` rep 0 (17). Ba lượt còn lại có 0, 2 và 3. Phân bố không đều.
- Penalty làm giảm finish trên bốn track không phải slalom: tổng 83 lượt (straight 35, right-turn 11, ramp 21, left-turn 16). Trên slalom penalty tăng 52 lượt finish. Tổng thể finish của penalty là 3.184, baseline là 3.215.

### Định nghĩa kết cục (đã đọc mã)

- `finished`: `metrics.success and not metrics.truncated` (`scripts/wp7_train5_v4_development.py:35-36`). `success` nhận từ `final_finished` (`f1rl/learning/evaluation/evaluator.py:349, 403`), là cờ `finished` của bước cuối. Nguồn của cờ này trong môi trường chưa được đọc.
- `hard_route_violation`: không bị cắt cụt, `reason == "hard_route_violation"` và `valid_sample` là True (`scripts/wp7_train5_v4_development.py:39-40`).
- `certified_physical_crash`: không bị cắt cụt, `reason` là `physical_crash` hoặc `physical_crash_development`, và qua `validate_registered_crash_terminal_info` hoặc `validate_crash_terminal_info` (`scripts/wp7_train5_v4_development.py:45-49`).
- `evaluate_one` yêu cầu đúng một sự kiện terminal và `valid_sample` là True, nếu không thì ném lỗi (`scripts/wp7_train5_v4_development.py:80, 83`). Batch này không có sự kiện `failed` và không có kết cục ngoài ba nhãn trên.

### Điều kiện nguồn khi chạy

- `_require_clean_source` (kiểm tra `git status --porcelain` rỗng) chỉ được gọi một lần, lúc bắt đầu execute: `prepare_output` (`scripts/wp7_penalty_trial_runner.py:43`), gọi từ `scripts/wp7_penalty_trial_cli.py:94`. Không gọi lại cho từng lượt.
- Mỗi lượt `run-one` gọi `activate` (`scripts/wp7_penalty_trial_capability.py:43`), gọi `verify_trial_scope`, rồi `validate_proposal`. `validate_proposal` dựng lại toàn bộ proposal từ source và so sánh (`scripts/wp7_penalty_trial_protocol.py:140-142`). Vì vậy băm của các file ràng buộc được kiểm tra lại ở đầu mỗi lượt. Cả 12 lượt hoàn thành, nên không lượt nào thấy lệch.
- Bằng chứng gián tiếp về cây sạch lúc chạy: `claim-batch` được ghi lúc 14:25:28, sau kiểm tra sạch ban đầu. Không có file nào trong `scripts/`, `f1rl/`, `tests/`, `configs/`, `docs/` có thời điểm sửa sau đó. HEAD không đổi từ 14:24:37 (reflog). Không có log riêng ghi kết quả kiểm tra sạch cho từng lượt.
- Chưa xác minh trực tiếp: cây làm việc sạch hoàn toàn (kể cả file không ràng buộc) ở từng lượt. Mã không kiểm tra điều này cho từng lượt, dù quy tắc 5 của plan mô tả việc đó.

### Đã thực hiện trong đợt này

- Miền seed mới v3 cho cả 12 lượt và toàn bộ seed đánh giá. v1 và v2 nằm trong `retired_seed_domains`.
- Chạy song song: mỗi lượt một process (12 process). Đánh giá chia 30 shard theo (cohort, replicate, track), ghép lại đúng thứ tự seed.
- Bỏ bước ký Ed25519 trên nhánh này, theo phần "chạy song song" ở trên. `verify_trial_scope` thay thế chữ ký.

### Giới hạn

- Không có kiểm định thống kê (D3). Chỉ là thống kê mô tả trên TRAIN.
- 6 cặp train (2 cohort x 3 replicate), tức 12 model. Mỗi ô (track x condition x model) có 60 episode.
- HELD-OUT chưa được đánh giá (`heldout_evaluated = false`).
- Chưa xác minh trực tiếp điều kiện nguồn ở từng lượt, như nêu trong mục trên.
- Chưa đọc nguồn cờ `finished` trong môi trường (`evaluator.py`, mục định nghĩa kết cục).
- Non-use toàn cục của seed v3 chưa được chứng minh. Chưa có review độc lập.

### Câu hỏi mở cho nhóm

1. Ngưỡng crash không quá 1% áp dụng cho tổng thể hay từng track?
2. Ngưỡng finish từ 80% gộp hai condition hay tính từng track x condition?
3. `hard_route_violation` có đưa vào báo cáo kết quả như một chỉ số không?
