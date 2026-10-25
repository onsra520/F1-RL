# WP3 Issue #51 — Bước 2: Versioned Crash Terminal Contract v1

**Ngày:** 24/10/2026
**Worktree:** `D:/F1-RL-WP3-EVAL-PHYSICAL-COMPLETION`
**Branch:** `fix/wp3-eval-physical-completion`
**Handoff HEAD nền:** `ce0e55d7a428826b2d4ed3218641105bd0206f4b`
**Frozen training source:** `fe83f517479f3b990028d6409304fae83b8199a9`

## Kết luận

**HOÀN THÀNH PHẦN TRIỂN KHAI, ĐĂNG KÝ VÀ NGHIỆM THU KỸ THUẬT CRASH TERMINAL V1.**

**CHƯA ĐƯỢC PHÊ DUYỆT KHOA HỌC / CHƯA ĐƯỢC PHÉP CHẠY REGISTERED FULL EVAL.** Mã nguồn hiện chưa commit, Physics v4 chưa có chữ ký phản biện độc lập và crash physics source vẫn mang tên v4 development trong source proof. Bước 2 hoàn tất phần kỹ thuật, không thể tự cấp phê duyệt khoa học.

## Hợp đồng và định nghĩa đã tích hợp

- **Race version:** `wp7_directed_lifecycle_crash_contract_v4`.
- **Scoring contract:** `wp3_physical_crash_terminal_v1`.
- **Physics version:** `spatial_road_edge_partial_lifecycle_v4` opt-in; default/legacy v3 giữ nguyên.
- **Result kind:** `valid_terminal_failure`, reason `physical_crash`, không tính là finish hoặc truncated.
- **Impact:** chỉ nhận từ phiên bản source áp dụng xung lượng và đã được `PhysicsConfig.validate_transition` kiểm toán trước khi tạo Gym info; không chấp nhận chỉ gắn cờ giả vào metadata.
- **Thời gian:** phân biệt `crossing_time_origin_s`/transition start tuyệt đối với `impact_time_s`/reward prefix tính tương đối trong tick; luôn xác minh `origin + fixed_dt == actual elapsed/end_time`.
- **Momentum:** đối chiếu chặt xung lượng tuyến tính, xung lượng góc với trước/sau va chạm, khối lượng, mômen quán tính và quaternion được khóa trong physics configuration.
- **Energy:** năng lượng động sau va chạm không tăng, loss > 0 và before-after = loss.
- **Event/reward prefix:** chỉ đến thời điểm va chạm đầu tiên; dùng `preimpact` cho reward, không nhận crossing hay checkpoint tương lai. Thực thi `Simulation -> Gym -> evaluator`, crash state absorbing.
- **Provenance:** `race.physical_crash_contract` là schema định phiên bản, gắn vào SHA-256 của race config/course/outcome; `physics.sha256` được tái kiểm tra và source crash kind/solver được kiểm tra.
- **Dev compatibility:** `wp7_directed_lifecycle_crash_dev_v3` hoạt động riêng; không tự nâng cấp các bản ghi lịch sử.

## Bằng chứng trên ba checkpoint/seed gốc — cùng source v4 và contract mới

Source: `docs/research/WP3-V4AJ-REGISTERED-CRASH-V1-THREE-REPLAY-20261024.json`. Đã kiểm tra SHA-256 model ZIP và env/policy seed từ nguyên gốc, chính sách CPU stochastic, 4 TRAIN tracks, 4m task, no TRAIN curriculum/HELD-OUT.

| Model / track / stratum / episode | Kết quả kỹ thuật theo schema mới |
|---|---|
| C-seed0 / right-turn-v1 / nominal / e0 | `physical_crash` — **valid FAILED crash**, không phải finish |
| A-seed1 / ramp-v1 / randomized_train / e17 | **finish** |
| C-seed0 / slalom-v1 / nominal / e2 | `physical_crash` — **valid FAILED crash**, không phải finish |

Ba ca hoàn thành ở mức kỹ thuật, không có unscored physics exception. Cả ba vẫn `scientific_evidence=false`. Replay đầu tiên ở phiên bản hợp đồng mới đã fail do cách hiểu sai time origin (absolute vs tick-local); hồ sơ `WP3-V4AI-...` được giữ riêng, sau khi sửa có bản replay mới độc lập `WP3-V4AJ-...`. Không ghi đè báo cáo cũ.

## Kiểm thử

- **372 passed / 0 failed**, pytest exit code 0, nhóm regression physics, evaluator, WP3 campaign, versioned crash source, tamper tests và source-bound registry.
- Receipt cuối: `docs/research/WP3-CRASH-TERMINAL-V1-FINAL-PYTEST-20261024.txt`.
- Test chứng nhận riêng contract: `tests/integration/test_wp3_registered_crash_terminal_contract.py`.
- Test integrity registry: `tests/research/test_wp3_crash_terminal_registry.py`.
- Các test giả mạo timestamp, seed, model, checksum, quantum quaternion, impulse, energy loss, hậu va chạm, danh tính obstacle và chữ ký khoa học đều bị chặn.
- Các test crash development cũ, owner-v2 fail-closed, hard-route valid failed episode vẫn PASS.

## Registry và source binding

- **Engineered technical registration:** `docs/research/WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json`.
- **Contract descriptor SHA-256:** `3fe6da8e4398de6da2ced760bf44800b00a94851e79362212cce968fcf7ebf82`.
- **12 source-file bundle SHA-256:** `681482bc1a2817a439a0eb70f3bb51cc03129d5e212497a1ee433c67a7b0627e`.
- **Registry record SHA-256:** `e049d2f5c894a45c36bad97495ff5be5e178a3747b03867a0a9bcaf7f54343c3`.
- `scripts/wp3_register_crash_terminal_v1.py` ghi exclusive `x`, không overwrite; `scripts/wp3_verify_crash_terminal_v1.py` đối chiếu lại source SHA, 12 checkpoint gốc, roster seed, replay, pytest receipt và registry digest.
- `independent_physics_review_signature=null`, `owner_scientific_signoff=false`, `scientific_scoring_approved=false`, `full_5760_episode_evaluation_authorized=false`.

## Boundary của bước 2

Phần mềm đã có schema versioned và kết quả crash kỹ thuật hợp lệ, nhưng **đây KHÔNG phải bằng chứng khoa học có người duyệt**. Cần review physics độc lập, sau đó khóa mã nguồn ở commit sạch (bước 3) rồi cập nhật hồ sơ nguồn/receipt mới, chạy smoke Eval-only (bước 4) trước khi khởi động 5.760 episode (bước 5). Biên bản kỹ thuật bước 1 trước đó có fingerprint nguồn cũ; sau khi bước 2 thay đổi evaluator/race/gym, fingerprint bước 1 phải được tái phát hành trên commit khóa mới chứ không tái sử dụng như bằng chứng của HEAD hiện tại.

Không huấn luyện lại, không dùng HELD-OUT, không sửa frozen TRAIN checkpoint hoặc lịch sử 96-cell Eval, không commit/push và không mở cổng Eval ở lượt này.
