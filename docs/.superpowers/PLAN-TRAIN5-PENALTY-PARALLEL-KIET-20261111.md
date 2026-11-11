# PLAN - TRAIN-5 penalty: chạy song song + bỏ bước ký (Kiệt, 11/11/2026)

Người thực thi: Claude Code (hoặc Codex nếu hết quota). Người quyết định: Kiệt.
Hạn: xong train + eval trước sáng 12/11. Ưu tiên đúng và nhanh, không mở rộng phạm vi.

## 0. Bối cảnh đã đo (14:00-14:20 ngày 11/11)

- Checkout đang chạy: `D:\REL301m\F1-RL`, nhánh `kiet/physics-v4-penalty-rerun` @ `c078c99`, sạch.
- Batch v2 đang chạy: PID 137688 (cha 129364), output `D:\REL301m\F1RL-runs\penalty-v2-20261111`.
  - Lượt 1 (baseline/train5_from_scratch/0) xong 10:35 -> 13:16 = 2h41, 200.704 bước (~21 bước/giây).
  - Lượt 2 bắt đầu 13:16. Tuần tự 12 lượt sẽ xong ~18:45 ngày 12/11, chưa tính eval. Không kịp.
- Process train chỉ dùng ~0,3 của 1 nhân; máy 12 nhân / 16 luồng, tải tổng ~6%. GPU không liên quan (physics Python thuần, PPO `device: cpu`).
- Nguyên nhân process chờ ~70% thời gian CHƯA xác định. Không xử lý trong plan này (Kiệt: không kịp).

## 1. Quyết định của Kiệt (đã chốt, không hỏi lại)

- D1. Bỏ bước ký Ed25519. Căn cứ: Tiến (chủ repo) xác nhận qua Zalo 11/11, 11:05-11:16, rằng trên main không có private key, code ký đến từ lần merge nhánh của Băng, vốn ở trạng thái disable và bị agent bật lại; Kiệt chuyển lời "không cần ký nữa".
- D2. Bỏ kết quả seed domain v2 (lượt 1 xong, lượt 2 dở). Dùng domain v3 cho cả 12 lượt và toàn bộ seed eval.
- D3. Sửa runner sang chạy song song ngay, không chờ Tiến duyệt trước; ghi lại trong tài liệu bàn giao.
- D4. Song song hóa cả phần eval 7.200 episode.
- Giữ nguyên: 12 lượt, 200.704 bước/lượt, 8 env `DummyVecEnv`, cấu hình PPO, 5 track TRAIN, penalty 0 vs 5, 60 episode x 2 condition, KHÔNG HELD-OUT.

## 2. Nguyên tắc an toàn

- KHÔNG sửa, commit, checkout, pull trong `D:\REL301m\F1-RL` khi PID 137688 còn sống. Mỗi lượt của batch cũ gọi lại `build_proposal` (hash mọi file tracked + `git rev-parse HEAD`); đụng vào là batch cũ chết.
- KHÔNG dừng batch v2 cho tới khi batch v3 ghi xong `training-result.json` (xem bước E3). Batch cũ chỉ tốn ~0,3 nhân và là đường lui.
- Không sửa gì trong `f1rl/` (physics, env, PPO adapter). Chỉ sửa `scripts/wp7_penalty_trial_*.py`, test tương ứng và tài liệu bàn giao.
- Không sửa pin/hash/manifest để ép test xanh. Gặp pin fail: DỪNG, báo Kiệt.
- Mọi output nằm ngoài checkout, trong `D:\REL301m\F1RL-runs\`.

## 3. Phase A - Bản làm việc riêng

A1. `git clone https://github.com/onsra520/F1-RL D:\REL301m\F1-RL-par`, rồi `git switch kiet/physics-v4-penalty-rerun`. Xác nhận HEAD = `c078c99`.
    - Bắt buộc đặt clone ngay dưới `D:\REL301m`: `build_proposal` đọc `root.parent\TRAIN5-v4-internal-20261109\evaluation-journal.jsonl` và `root.parent\slalom-fresh-audit.json` (cả hai đang tồn tại).
    - 12 file `models/*/model-final.zip` là file tracked nên clone có sẵn; kiểm tra hash qua test, không copy tay.
A2. Dùng lại venv `D:\REL301m\.venv-f1rl`. Chạy từ cwd = clone và in ra để xác nhận import đúng bản clone:
    `D:\REL301m\.venv-f1rl\Scripts\python.exe -B -c "import f1rl,scripts;print(f1rl.__file__)"` phải ra `D:\REL301m\F1-RL-par\...`. Nếu ra `D:\REL301m\F1-RL\...`: DỪNG, báo (không `pip install -e` lại vì sẽ đổi môi trường của batch cũ).
A3. Baseline test trước khi sửa: chạy 6 file `tests/research/test_wp7_penalty_trial_*.py` + `test_wp7_terminal_crash_penalty.py`. Ghi pass/skip/fail.

## 4. Phase B - Sửa code (1 commit code+test)

B1. Seed domain v3. Trong `scripts/wp7_penalty_trial_protocol.py` đổi `DOMAIN` thành `'F1RL-TERMINAL-PENALTY-TRIAL-20261111-v3'`. Làm đúng cách commit `224d6ff` đã làm cho v1 -> v2 (xem `git show 224d6ff`), mở rộng test để chứng minh seed train + eval của v3 rời hoàn toàn với v2 (và v1 nếu test cũ đã phủ).

B2. Bỏ ký.
    - Thêm hàm `verify_trial_scope(root, proposal, output, *, allow_existing=False)`: giữ `validate_proposal(root, proposal)` (ràng buộc commit + hash nguồn) và kiểm tra output nằm ngoài checkout / chưa tồn tại. Không envelope, không chữ ký, không hạn 168 giờ.
    - `runner`, `capability`, `evaluate`: thay mọi lời gọi `verify_trial_authorization(...)` bằng `verify_trial_scope(...)`, bỏ tham số `envelope`. Trong `capability.admitted` bỏ điều kiện hết hạn; GIỮ kiểm tra cấu trúc env (DummyVecEnv, đúng 8 env, đúng amount/seed/scenarios), GIỮ ngân sách bước trong `before_step`, GIỮ file `.claim-<intervention>-<cohort>-<replicate>` mở chế độ 'x' (mỗi lượt chỉ chạy một lần; nội dung ghi tên batch thay cho nonce).
    - CLI: bỏ subcommand `sign`; `execute` nhận `--plan` thay cho `--packet`.
    - Không xóa `wp7_penalty_trial_authorization.py`, file owner-key hay test của nó (ngoài phạm vi); chỉ ngừng gọi.

B3. Train song song - mỗi lượt một process.
    - Tách thân vòng lặp trong `run_trial` thành `run_one(root, proposal, output, evidence_root, intervention, cohort, replicate)`. Logic bên trong giữ nguyên từng dòng (scenarios, `_prepare_v4_runner`, `build_trial_environment`, `activate`, `PPOModel.create`, initializer, `learn`, kiểm tra số bước, `save`, `_final_model_observer`, kiểm tra checkpoint gốc). Cuối lượt ghi `<run_dir>\run-result.json` (mode 'x') gồm identity, steps, model, sha256.
    - Đầu `run_one`: `torch.set_num_threads(1)`. Truyền `progress_hook` vào `model.learn(...)` (tham số đã có sẵn ở `PPOModel.learn`) để in `steps=<n>` mỗi 2.048 bước ra stdout, flush. Hook chỉ in, không đụng RNG.
    - CLI thêm `run-one --plan --evidence --intervention --cohort --replicate` (worker) và `execute --plan --evidence --jobs N` (launcher).
    - Launcher: kiểm tra source sạch + `verify_trial_scope`; `output.mkdir(exist_ok=False)`; spawn 12 worker bằng `subprocess.Popen([sys.executable, '-B', '-m', 'scripts.wp7_penalty_trial_cli', 'run-one', ...])`, tối đa N cùng lúc, env thêm `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; stdout+stderr mỗi worker vào `output\logs\<run>.log`. Worker lỗi: KHÔNG kill worker khác, chờ hết rồi exit code != 0 và in danh sách lượt lỗi. Đủ 12 `run-result.json`: ghi `training-result.json` (mode 'x') theo đúng thứ tự cũ (baseline rồi penalty, theo `proposal['pairs']`), cùng schema cũ.
    - Journal: mỗi worker ghi `<run_dir>\run-journal.jsonl` (started/completed); không ghi chung một file từ nhiều process.

B4. Eval song song - chia shard theo (cohort, replicate, track) = 30 shard, mỗi shard 240 episode.
    - Tách vòng lặp trong `evaluate_trial` thành `evaluate_shard(..., cohort, replicate, track)`: nạp 2 policy của cặp đó (kiểm tra path/hash/steps + `validate_loaded_policy` như cũ), chạy các hàng `evaluation_seed_rows` của shard theo đúng thứ tự gốc, mỗi hàng chạy lần lượt các intervention, ghi `output\eval-shards\<cohort>-<rep>-<track>.jsonl` (mode 'x', flush + fsync từng dòng như cũ). `torch.set_num_threads(1)` ở đầu.
    - CLI thêm `evaluate-shard` (worker) và `evaluate --plan --evidence --jobs N` (launcher + merge). `execute` gọi train xong thì gọi evaluate như cũ.
    - Merge: đọc 30 shard, ghép theo thứ tự `evaluation_seed_rows` thành `paired-evaluation-journal.jsonl`, yêu cầu đúng 7.200 dòng `completed`, không trùng, không thiếu; tính `cells`, `cohort_summary` bằng đúng công thức hiện tại; ghi `paired-evaluation-result.json` (mode 'x').
    - Cơ sở cho việc chia shard: `evaluate_one` tự cô lập RNG từng episode (`torch.random.fork_rng` + `manual_seed(policy_seed)`, env seed truyền tường minh). Đây là suy luận từ đọc code, phải được test B5 chứng minh.

B5. Test (sửa/thêm trong các file test penalty hiện có):
    - seed v3 rời v2;
    - runner/capability/evaluate chạy được không cần envelope; claim trùng vẫn bị từ chối; vượt ngân sách bước vẫn raise;
    - trên fake batch (`test_wp7_penalty_trial_fake_batch.py`): launcher song song cho `training-result.json` cùng tập lượt, cùng thứ tự; eval chia shard + merge cho summary BẰNG kết quả chạy tuần tự trên cùng input;
    - các assert cũ giữ nguyên: 12 lượt, 2.408.448 bước, 3.600 hàng, 7.200 episode.
    Chạy lại toàn bộ bộ test ở A3. Có fail: sửa code của mình, không sửa pin.

B6. Commit 1 (code + test), tiếng Anh. Push `kiet/physics-v4-penalty-rerun`. KHÔNG chờ CI xanh mới chạy (CI skip test witness, test local mới là cổng); ghi SHA và kết quả CI vào báo cáo sau.

## 5. Phase C - Tài liệu (commit 2, trước khi prepare)

Thêm mục "Cập nhật 11/11 chiều - chạy song song" vào `docs/.superpowers/handoff/.../TRAIN5-TERMINAL-PENALTY-HANDOFF-20261111.md` (đúng file bàn giao hiện có), nêu:
- D1-D4 ở mục 1, kèm nguồn xác nhận của Tiến (Zalo 11/11, 11:05-11:16, do Kiệt chuyển lời);
- trạng thái v2: lượt 1 xong (sha256 model `7264e02d...c52353`), lượt 2 bị dừng chủ động; output v2 giữ nguyên tại chỗ, KHÔNG dùng cho so sánh;
- thay đổi cách chạy: mỗi lượt một process, 1 luồng torch/process, eval 30 shard; cấu hình khoa học không đổi;
- giới hạn giữ nguyên: non-use toàn cục của seed mới chưa chứng minh, chưa review độc lập, chưa HELD-OUT; thêm: không còn chữ ký ủy quyền.
Copy file plan này vào `docs/.superpowers/PLAN-TRAIN5-PENALTY-PARALLEL-KIET-20261111.md` trong cùng commit. Kiểm tra UTF-8. Push.

## 6. Phase D - Chạy (trong clone, terminal riêng, có log)

D1. `git status --porcelain` rỗng. Ghi HEAD.
D2. prepare:
    `python -B -m scripts.wp7_penalty_trial_cli prepare --output D:/REL301m/F1RL-runs/penalty-v3-20261111 --plan D:/REL301m/F1RL-runs/penalty-v3-20261111-plan.json --evidence D:/F1-RL-PHYSICS-V4-EVIDENCE-20261106`
D3. execute, nền, log ra file:
    `python -B -m scripts.wp7_penalty_trial_cli execute --plan D:/REL301m/F1RL-runs/penalty-v3-20261111-plan.json --evidence D:/F1-RL-PHYSICS-V4-EVIDENCE-20261106 --jobs 12`
    stdout/stderr vào `D:\REL301m\F1RL-runs\penalty-v3-20261111-execution.log`.
D4. Cổng đo sau ~5 phút: đọc dòng `steps=` trong 12 log, tính bước/giây từng lượt và tổng.
    - Mốc tuần tự: ~21 bước/giây. Nếu mỗi lượt >= ~15 bước/giây: ETA train <= ~3h45.
    - Nếu TỔNG < 42 bước/giây (chưa gấp đôi tuần tự): báo Kiệt ngay kèm số đo, vẫn để chạy (dừng không lợi gì vì seed v3 đã tiêu).
    - Báo Kiệt ETA train ngay sau lần đo này.
D5. Trong lúc chạy: không sửa/commit trong clone.

## 7. Phase E - Kết thúc

E1. Train xong: xác nhận 12 `run-result.json`, 12 model, `training-result.json`. Lượt nào lỗi: DỪNG, báo Kiệt (không tự xóa claim, không tự chạy lại).
E2. Eval tự chạy tiếp theo `--jobs 12`. Sau 5 phút đo số dòng/giây trong `eval-shards` và báo ETA. Đây là số chưa ai đo.
E3. Chỉ khi `training-result.json` của v3 đã tồn tại: dừng batch v2 (`taskkill /PID 137688 /T`; kiểm tra 129364 cũng hết). Ghi giờ dừng và dòng journal cuối của v2 vào tài liệu bàn giao.
E4. Báo cáo cho Kiệt (trong chat, không tạo report mới): theo từng track và từng nhánh reward: finish rate, certified physical crash rate, so với ngưỡng cũ (slalom >= 80% hoàn thành, crash <= 1%); kể cả khi kết quả xấu. Kèm: SHA commit, số test pass/skip/fail, kết quả CI, thời gian train/eval thực đo, mọi lệch khỏi plan.
E5. Commit 3 (docs): cập nhật tài liệu bàn giao với kết quả và trạng thái v2. Push.

## 8. Điều kiện DỪNG và hỏi Kiệt

- Import `f1rl` trỏ về checkout cũ (A2).
- Test pin/hash/manifest fail vì thay đổi của plan này.
- Seed v3 va chạm (`ValueError` từ `build_proposal`).
- Test B5 cho thấy eval chia shard KHÁC eval tuần tự.
- Bất kỳ lượt train hoặc shard eval nào lỗi.

## 9. Tự rà (Claude, 11/11 14:30)

- Đã đọc: `wp7_penalty_trial_{runner,capability,cli,evaluate,authorization,protocol}.py`, `check_active_step`, `evaluate_one`, `PPOModel.create/learn` trên `c078c99`.
- Chưa kiểm chứng runtime: (1) tốc độ khi chạy 12 process - nếu thứ process đang chờ là tài nguyên dùng chung thì không tăng tuyến tính; (2) thời gian eval; (3) `torch.set_num_threads(1)` có đổi số học so với mặc định hay không - không ảnh hưởng so sánh vì cả 12 lượt v3 cùng cài đặt, nhưng model v3 không so bit-với-bit được với v2.
- Rủi ro còn lại: không resume; mất điện/sleep là mất các lượt đang chạy và phải sang v4. Máy phải cắm sạc, không sleep tới khi xong.
- Tham khảo cách làm ngoài: tài liệu SB3 khuyên PPO MlpPolicy chạy CPU; các framework RL (Hydra + joblib launcher, RL Zoo) chạy mỗi seed một job/process song song. Plan chọn process-per-run thay vì `SubprocVecEnv` vì cách sau đổi cấu trúc env mà capability đang kiểm tra (DummyVecEnv x 8).
