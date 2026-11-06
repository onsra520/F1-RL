# F1-RL — Phase 2A: Root Cause Audit

**Kết luận:** Phase 1 không chạy Physics v4. Smoke dùng profile `spatial_rotating_owner_v3`, profile này cấu hình `spatial_rotating_compliant_lifecycle_v3` (Physics v3) và race `wp7_directed_lifecycle_owner_v2`. Vì thế 8 lần solver từ chối transition ngoài miền chứng nhận đã bị harness ghi là `invalid_physical_transition`. Khi áp dụng đúng runtime overlay của FULL-06 (Physics v4 + crash terminal v4 + tilt envelope đã đăng ký cho đợt đó), cùng 8 model/track/reset-seed và policy deterministic đều kết thúc bằng **valid `physical_crash`**, không còn invalid.

## Phạm vi và tiêu chí

- Chỉ đọc worktree `D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit`; không sửa code/config, không port/retrain/commit/merge, không chạy test suite, không đọc geometry HELD-OUT.
- Mọi script, replay và báo cáo Phase 2A được lưu ngoài worktree tại `D:\Projects\.evidence\F1-RL\phase2a-root-cause-audit-20261106`.
- Tái hiện đạt: **8/8** đúng model, track, reset seed `20261106`, failed-attempt step, exception class/message.
- Counterfactual cùng model/track/reset-seed với v4 runtime overlay đạt: **8/8** valid terminal failure `physical_crash`.
- Không có thay đổi source trong worktree. Worktree baseline đang ở nhánh unborn (chưa có `HEAD`; files là snapshot untracked). Các file source/config/track liên quan được so sánh blob hash và trùng với source commit Phase 1 `6c479540134d98d7d27331879c926c8ba82c2965` (11/11).

## Bảng 8 lỗi Phase 1

Tọa độ là trạng thái xe ngay trước transition bị từ chối; action theo thứ tự `[steering, throttle, brake]`. Với 6 lỗi sweep, contact của từng bánh có chứng cứ trong JSON replay; diện tích footprint ngoài road union chỉ hơi vượt ngưỡng `_EPS = 1e-9 m²`, trong khi cả 4 bánh có reference contact và chassis clearance dương. Với 2 lỗi ramp, bánh chỉ vượt giới hạn bump 0.12 m dưới 0.7 mm.

| Model | Track | Bước lỗi / trạng thái xe `(x,y,z), speed` | Action | Bằng chứng tại transition v3 | Kết quả v4 overlay cùng seed | Phân loại lỗi transition |
|---|---|---|---|---|---|---|
| C-seed0 | Slalom | 91; `(4.801, 0.495, 55.200)`, 4.491 m/s | `[-1,0,0]` | `SuspensionResolutionError`, `unsupported_contact`; footprint uncovered `1.000188e-9 m²`, bánh 4/4 có contact | `physical_crash`, bước 150 | `EXPECTED_REJECTION` — v3 yêu cầu full-body road coverage |
| C-seed0 | Right Turn | 90; `(-0.309, 0.504, 67.655)`, 6.207 m/s | `[1,1,0]` | `unsupported_contact`; uncovered `1.008065e-9 m²`, bánh 4/4 có contact | `physical_crash`, bước 93 | `EXPECTED_REJECTION` — v3 yêu cầu full-body road coverage |
| B-seed1 | Slalom | 121; `(-4.940, 0.502, 82.413)`, 7.471 m/s | `[1,1,0]` | `unsupported_contact`; uncovered `1.000785e-9 m²`, bánh 4/4 có contact | `physical_crash`, bước 125 | `EXPECTED_REJECTION` — v3 yêu cầu full-body road coverage |
| B-seed1 | Right Turn | 78; `(-0.330, 0.500, 67.553)`, 8.674 m/s | `[1,1,0]` | `unsupported_contact`; uncovered `1.010458e-9 m²`, bánh 4/4 có contact | `physical_crash`, bước 81 | `EXPECTED_REJECTION` — v3 yêu cầu full-body road coverage |
| B-seed1 | Ramp | 55; `(0.185, 0.527, 49.519)`, 9.265 m/s | `[-1,1,0]` | `spatial wheel support/travel missing`; front-right deflection `0.120264 m`, vượt bump limit `0.12 m` **0.264 mm** | `physical_crash` do wheel hard-stop, bước 55 | `EXPECTED_REJECTION` — v3 không chứng nhận contact vượt bump limit |
| A-seed2 | Slalom | 82; `(4.530, 0.494, 54.068)`, 6.064 m/s | `[-1,1,0]` | `unsupported_contact`; uncovered `1.000672e-9 m²`, bánh 4/4 có contact | `physical_crash`, bước 86 | `EXPECTED_REJECTION` — v3 yêu cầu full-body road coverage |
| A-seed2 | Right Turn | 82; `(-0.162, 0.504, 67.875)`, 7.852 m/s | `[0.925,1,0]` | `unsupported_contact`; uncovered `1.032902e-9 m²`, bánh 4/4 có contact | `physical_crash`, bước 84 | `EXPECTED_REJECTION` — v3 yêu cầu full-body road coverage |
| A-seed2 | Ramp | 55; `(-0.571, 0.493, 48.613)`, 8.581 m/s | `[1,1,0]` | `spatial wheel support/travel missing`; front-left deflection `0.120620 m`, vượt bump limit `0.12 m` **0.620 mm** | `physical_crash` do wheel hard-stop, bước 55 | `EXPECTED_REJECTION` — v3 không chứng nhận contact vượt bump limit |

**Phân loại run/harness:** `IMPLEMENTATION_BUG` — Phase 1 gắn nhãn “Physics v4” nhưng chọn profile/runtime v3. Đây là lỗi wiring cấu hình smoke, không phải bằng chứng 8 lỗi bắt nguồn từ implementation Physics v4. Không thấy `INVALID_GEOMETRY` trong 4 TRAIN assets hoặc lỗi model import. Sáu case cạnh đường sát ngưỡng là nơi cần regression số học riêng; audit chưa kết luận đó là round-off hay overhang hình học cực nhỏ.

## Luồng gây lỗi và chuyển thành terminal

1. `ppo-smoke.py:22–23` gọi `load_registration(..., profile="spatial_rotating_owner_v3")` rồi `prepare_scenarios(...)`; model action được đưa trực tiếp vào `env.step` (`ppo-smoke.py:110–112`). Smoke không overlay Physics v4.
2. `wp7_protocol.py:176,193` lấy `ground_motion_version` từ registration và tạo race `wp7_directed_lifecycle_owner_v2`. Registration JSON ghi rõ `spatial_rotating_compliant_lifecycle_v3`.
3. `world.py:207–212` dispatch v3 sang `integrate_spatial_lifecycle(... phase_domain="atlas_rotating_v3")`. Trong RK stage, thiếu wheel support/travel ném `SuspensionResolutionError` (`spatial_contact_dynamics.py:258–268`); sweep v3 gọi verifier full-body support mặc định, rồi ném `unsupported_contact` khi endpoint ngoài miền (`:885–900`). Đây là đường đi của 6 sweep và 2 ramp failures.
4. Exception thoát `Simulation.step` trước khi race/terminal được tính. `gym_environment.py:435–440` bắt exception và ghi `invalid_physical_transition`; vì vậy v3 không thể đổi lỗi đó thành physical crash terminal.
5. Khi cấu hình v4, `world.py:207` chọn `integrate_spatial_road_edge_lifecycle`. V4 dùng finite-wheel coverage (`spatial_contact_dynamics.py:258–268`), chứng nhận swept chassis/road và phát `SpatialRoadCrash` hoặc wheel-hard-stop crash (`:802–869`). `world.py:223–235` bắt event, trả crash snapshot cho race layer; `gym_environment.py:576–600,635–643` ghi terminal contract/valid `physical_crash`.

Counterfactual JSON ghi riêng loại crash: các Slalom/Right Turn gồm `bounded_road_body` hoặc `wheel_hard_stop`; hai Ramp kết thúc bằng `wheel_hard_stop`. Bước kết thúc khác bước Phase 1 ở phần lớn case vì Physics v4 làm thay đổi quỹ đạo từ đầu. Do đó đây là **đối chứng runtime cùng seed**, không phải tuyên bố cùng một integration substep v3 được replay nguyên trạng thành collision v4.

## Vì sao FULL-06 có 0 invalid

- FULL-06 source head trùng Phase 1: `6c479540134d98d7d27331879c926c8ba82c2965`.
- Summary FULL-06 (`OWNER-TRAINDEV-FINAL-SUMMARY.json`) ghi **96/96 cells**, **5,760 valid episodes**, **0 invalid**; có **1,253 physical crashes**, **1,256 hard-route outcomes** và **1,700 step-limit outcomes**. “0 invalid” không có nghĩa mọi episode đều finish.
- FULL-06 coordinator overlay rõ `spatial_road_edge_partial_lifecycle_v4`, race `wp7_directed_lifecycle_crash_contract_v4` và `max_abs_tilt_rad=.7` (`wp3_eval_only_coordinator.py:25–26,149–155`), chạy stochastic theo episode seed. `heldout_access=false`; đây là train/dev engineering evidence, scientific gate vẫn blocked.
- Phase 1 lại chạy deterministic một episode/model-track, reset seed chung `20261106`, cap 2,500 bước, với Physics v3/race v2. V3 ném exception trước terminal; FULL-06 v4 biểu diễn road/wheel contact như terminal outcome hợp lệ. Cùng source commit/model và đúng các track TRAIN nhưng config/runtime contract khác.
- Đối chứng trực tiếp trong audit: thay runtime sang FULL-06 v4 overlay cho 8 model-track-reset-seed Phase 1 thì cả 8 là valid `physical_crash`. Điều này xác nhận config bundle giải thích discrepancy; stochastic sampling và roster episode cũng khác FULL-06 nhưng không cần dùng chúng để giải thích 8 invalid.

## Git history, assets và import

| Commit | Ngày | Ý nghĩa liên quan |
|---|---|---|
| `1732fdf51aa8b33e127c0ebba728e6099b090554` | 2026-10-18 | Đăng ký rotating owner lifecycle; profile đã khai báo ground motion v3. Blob profile này vẫn không đổi tại Phase 1 source commit. |
| `4297bb7e3250c37b51f04eb0194885a5e80e00ae` | 2026-10-24 | Tích hợp Physics v4 và versioned crash terminal. |
| `c057135a329af59e07d9823a7bedf4b2b06f8682` | 2026-10-25 | Finite-tire edge support và certified hard-stop terminal. |
| `6c479540134d98d7d27331879c926c8ba82c2965` | 2026-10-27 | Bounded-road airborne impact lifecycle; commit được Phase 1 và FULL-06 manifest ghi nhận. |

Bảy file code/config liên quan và cả 4 TRAIN track JSON trong snapshot worktree có Git blob hash trùng source commit `6c47954` (**11/11**). Ba model archive Phase 1 có SHA-256 bằng bản frozen dùng cho FULL-06: C-seed0 `15373c0237e883fa2f980dfecbda4809aabc283c0d77a8b4d783e2ab5bf11492`, B-seed1 `695f1c649235775eb1446ac0718b69b5951a1242b389221a5318db540a716021`, A-seed2 `e2a0eb0deef34fad7bf2bf2322c0c13854869372263a4652d4c6f1c422d43adc`. Không có bằng chứng asset drift hoặc checkpoint import mismatch. Không có dấu hiệu khác biệt Python/dependency cần thiết để giải thích; cùng target source tái hiện v3 và chạy v4 overlay được.

## Phase 2B đề xuất — chưa thực hiện

1. **Khóa cấu hình smoke:** chuyển harness sang khai báo Physics v4 + race/crash contract v4 rõ ràng; ghi effective config hash trong preflight và mỗi run; chặn chạy nếu profile effective không khớp tên audit.
2. **Regression cho 8 case:** frozen model hashes, TRAIN track hashes, reset seed, policy deterministic; v4 phải cho valid terminal `physical_crash` hoặc finish, không `invalid_physical_transition`. Khẳng định không redraw case invalid.
3. **Giữ regression v3 fail-closed:** các sweep outside và bump >0.12 m vẫn bị từ chối nếu gọi v3; test tách biệt để không biến một rejection hợp đồng thành false pass.
4. **Kiểm tra boundary tolerance:** test body union coverage tại đúng biên, lệch vào/ra và diện tích quanh `1e-9 m²` cho Slalom/Right Turn; xác định nếu 6 case là overhang thật hay artifact clipping/round-off trước khi cân nhắc đổi epsilon.
5. **Ramp suspension boundary:** kiểm tra hai bánh/track quanh bump limit, finite tire partial support và wheel hard-stop; kiểm chứng crash witness, impulse/energy, terminal prefix, không mất contact hoặc tạo terminal không chứng nhận.
6. **End-to-end contract:** wheel support → integration → sweep/collision → race terminal, bao gồm wheel-hard-stop và bounded-road body-contact; assertion phải phân biệt physical crash hợp lệ với solver invalid.
7. Sau khi Phase 2B được nghiệm thu mới đề xuất fix. Chưa mở HELD-OUT và chưa chạy scientific evaluation.

## Bằng chứng và trạng thái kiểm tra

- `v3-eight-case-instrumented-replay.json`: **PASS, exact reproductions 8/8**, gồm trạng thái xe/suspension, wheel support, road patches, PPO actions, RK substep witness và exception.
- `v4-overlay-counterfactual.json`: **8/8 valid terminal `physical_crash`** dưới runtime overlay FULL-06.
- `source-history-verification.json`: provenance/hash comparison, model hashes và xác nhận target worktree unborn.
- `replay_v3_eight.py`, `replay_v4_overlay.py`: diagnostic drivers chạy ngoài worktree với Python 3.12.15, CPU, bytecode writing disabled.
- **Không chạy test suite** vì không sửa code; không có claim PASS cho unit/integration tests.

**Kết thúc Phase 2A. Dừng tại audit; chưa sửa code, chờ nghiệm thu trước Phase 2B/fix.**
