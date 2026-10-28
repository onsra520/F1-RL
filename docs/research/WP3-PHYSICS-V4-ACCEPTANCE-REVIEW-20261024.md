# WP3 — Physics v4 | Biên bản nghiệm thu kỹ thuật bước 1

**Ngày:** 24/10/2026 (Asia/Ho_Chi_Minh)
**Nhánh:** `fix/wp3-eval-physical-completion`
**Worktree:** `D:/F1-RL-WP3-EVAL-PHYSICAL-COMPLETION`
**Commit nền (không ghi đè):** `ce0e55d7a428826b2d4ed3218641105bd0206f4b`
**Physics opt-in:** `spatial_road_edge_partial_lifecycle_v4`

## Kết luận nghiệm thu theo phạm vi

**Nghiệm thu kỹ thuật (engineering / software preflight): PASS.** Đủ ba replay bắt buộc, đã chạy bộ kiểm thử trọng tâm bảo toàn trạng thái/tiếp xúc/crash, kiểm tra toàn bộ 12 checkpoint, khóa SHA-256 cho 21 tệp mã vật lý/evaluator trọng yếu và lưu biên bản máy đọc được. Đây là **mức nghiệm thu kỹ thuật nội bộ**, không phải giấy phê duyệt khoa học hoặc cam kết cơ học thực nghiệm có hiệu chuẩn.

**Nghiệm thu physics-domain độc lập: PENDING.** Chưa có chữ ký và danh tính người đánh giá độc lập được xác minh; chưa có bằng chứng đo thực tế để hiệu chuẩn vật liệu, ma sát, giảm xóc hoặc tính mất năng lượng vào nền đường; nhánh còn thay đổi chưa commit. Do đó không thể đánh dấu `approved_physical_scoring_contract=true` hay mở Eval chính thức.

| Tiêu chí | Kết quả | Bằng chứng |
|---|---|---|
| Phiên bản v4 opt-in, không thay v3 | **PASS** | PhysicsConfig, world/gym wiring, test source |
| Tiếp xúc 0/1/2/3/4 bánh, giới hạn bump/droop | **PASS kỹ thuật** | Synthetic dynamics tests |
| Bảo toàn xung lượng, kiểm toán work/energy, fail-closed | **PASS kỹ thuật** | RK4 tests, edge-release tests, impulse audits |
| Road first-contact, rigid-body impact, earliest event | **PASS cho fixture / replay bắt buộc** | Geometry tests + 3 frozen replay |
| Crash terminal chỉ giữ reward và checkpoint trước va chạm | **PASS development** | Simulation → Gym → Evaluator integration tests |
| Bộ regression physics/evaluator | **300 PASS / 0 FAIL** | `WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt` |
| Kiểm tra giả mạo SHA/seed/approval | **20 PASS / 0 FAIL** | `tests/research/test_wp3_v4_technical_acceptance.py` |
| Mã nguồn Python quan trọng | **21/21 parse PASS** | AST parse |
| Môi trường thư viện | **PASS** | `pip check`: No broken requirements found |
| 12 checkpoint final ban đầu | **12/12 SHA PASS** | Provenance và model ZIP gốc |
| 3 model/seed replay bắt buộc | **3/3 không exception** | `WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json` |
| Độc lập hiệu chuẩn và reviewer ký nghiệm thu | **CHƯA ĐẠT** | Không có tài liệu/chữ ký xác minh |
| Clean HEAD và profile được đăng ký để chấm Eval | **CHƯA ĐẠT** | Source chưa commit, outcome `*_dev` |

### Ba ca replay cùng mã hiện tại

- **C-seed0 / right-turn-v1 / nominal / episode 0:** hoàn thành dưới `physical_crash_development`; không được coi là finish.
- **A-seed1 / ramp-v1 / randomized_train / episode 17:** hoàn thành kỹ thuật, `success_flag_unaccepted=true`.
- **C-seed0 / slalom-v1 / nominal / episode 2:** hoàn thành dưới `physical_crash_development`; không được coi là finish.

Cả ba sử dụng đúng model-final SHA và cặp environment/policy seeds từ roster gốc. `scientific_evidence=false` ở tất cả ba bản ghi. Kết quả này không xác nhận rằng mọi episode trong 96 cell sẽ chạy hợp lệ.

### Fingerprint nguồn và bằng chứng

- **21-file Physics/Evaluator bundle SHA-256:** `eb0c03e7f422d4a9cf5d4faccc988fb5831dfa4c8891aa3e7a2e0d8188e94856`.
- **Biên bản JSON:** `WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json`.
- **Checksum nội dung biên bản:** `40833fb321ab17a1644ac828eef283a2b67fd0bc68c8564fc23bdd17a09c2649`.
- **Replay ba ca:** `WP3-V4AH-PHYSICS-V4-ACCEPTANCE-REPLAY-20261024.json`.
- **Test receipt:** `WP3-V4-PHYSICS-ENGINEERING-PYTEST-20261024.txt`.
- **Verifier:** `scripts/wp3_v4_verify_technical_acceptance.py`. Verifier tái so sánh chính xác mã nguồn SHA, 12 weights, seed roster, replay và test receipt; không tự nhận chữ ký độc lập.

### Các ranh giới và phần còn phải ký

1. **Physics reviewer độc lập:** rà soát tiếp xúc thân–đường hình học, nghiệm bắt đầu từ trạng thái không chồng lấn, lực ma sát, dữ liệu vật liệu, mô hình truyền năng lượng khi bánh rời nền đường, bump-stop impulse, bảo toàn động lượng góc và độ chính xác phân chia RK4 tại sự kiện.
2. **Người phụ trách nghiên cứu:** phê duyệt hoặc bác bỏ định nghĩa *valid failed crash episode* và miền tham số của physics v4; nêu cụ thể tolerance và phương pháp đối chiếu độc lập với implementation.
3. **Khóa nguồn:** phân loại tệp chẩn đoán, loại bỏ mã tạm nếu phù hợp, commit trên nhánh riêng sau review và chạy lại trên **HEAD sạch**. Đây là bước 3, chưa thực hiện.
4. **Eval release:** bước 2 phải thay outcome `physical_crash_development` bằng contract đã đăng ký và được chứng nhận; cần preflight Eval-only trước 5.760 episode. Chưa chạy Eval chính thức, chưa mở HELD-OUT.

### Cách xác minh trên Windows

```powershell
cd D:\F1-RL-WP3-EVAL-PHYSICAL-COMPLETION
& 'D:\F1-RL-WP3\.venv\Scripts\python.exe' -B scripts\wp3_v4_verify_technical_acceptance.py
& 'D:\F1-RL-WP3\.venv\Scripts\python.exe' -B -m pytest -q tests\research\test_wp3_v4_technical_acceptance.py
```

**Trạng thái chính xác:** `TECHNICAL_GATE_PASS_SCIENTIFIC_SIGNOFF_PENDING` — chỉ hoàn thành phần kiểm chứng kỹ thuật của bước 1. Cổng `production_and_registered_eval_admitted=false`; chưa commit/push, không sửa model gốc hay dữ liệu Eval cũ.
