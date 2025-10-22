# Simple Static End-to-End Verification (Container Constraints)

## 1. Legacy profile/mode purge
- `mode`: still present in Wi-Fi UI state attributes, Wi-Fi helper commands, and historical reports (`reports/purge_report.txt`). No backend decision logic depends on it. (`rg -w "mode"`)
- `profile`: only referenced inside Windows Wi-Fi provisioning utilities (`netsh wlan add/delete profile`) and PyInstaller manifests; no runtime profile configuration remains. (`rg -w "profile"`)
- `STRICT` / `BALANCED` / `RELAXED` / `hysteresis` / `frameskip` / `sensitivity`: only appear inside the historical purge report, confirming they were removed from code and tests. (`rg` searches)
- `loader`: survives exclusively in frontend CSS/HTML loader spinners and PyInstaller bootstrap metadata; no backend loaders or profile files. (`rg -w "loader"`)
- `find -iname '*profile*.json|yaml'` returned no files, confirming that legacy profile JSON/YAML payloads are not present in the repository.

## 2. Backend parameters (synthetic run captured in repo)
- `params_effective.json` records per-class `enter`/`release` thresholds, consensus `2/4`, `cooldown_s ≈ 0.7`, `listen_window_s = 5.5`, `process_every_n = 3`, and `roi_margin = 0.05`. (`logs/params_effective.json`)
- `/engine/status` snapshots (`engine_status_before.json`, `engine_status_after.json`) match the above thresholds and temporal parameters and confirm the `simple-static` engine with `frame_stride = 3` and synthetic stream source.
- Backend session log shows the production model, dataset fallback, threshold application, and SSE client connection, but only for a synthetic dataset (no real camera frames recorded).

## 3. Event stream & UI traces
- `session_events.log` contains SSE payloads flowing sequentially (`gesture_detected` alternating `Start`/`Clima`) without duplicate connections, confirming the event channel works for the synthetic session.
- Frontend loader/ring glow/tutorial behaviours were not re-run in this container; prior structure tests still reference the Wi-Fi loader element.

## 4. Metrics captured
- `engine_status_after.json` includes latency statistics (`avg_latency_ms ≈ 2.24`, `p95 ≈ 2.73`), reason counters, and indicates zero real frames processed (synthetic pipeline reports `frames_total = 0`).
- Session logs do not report false positives; all detections came from the scripted synthetic dataset.

## 5. Platform coverage & limitations
- Container image lacks webcam/V4L2 access, Windows build tooling, and Raspberry Pi hardware, so real-camera acquisition, cross-platform packaging smoke tests, and thermal measurements were not executed here.
- No attempt was made to force TensorFlow/TFLite failures; the captured session already ran in fallback-only mode (XGBoost pipeline).

## 6. Follow-up recommendations
1. Re-run the checklist on Windows with a physical webcam to capture real `/health`, `/engine/status`, `/net/*` responses and confirm Wi-Fi provisioning (netsh) works end-to-end.
2. Repeat on Raspberry Pi 64-bit with V4L2/libcamera to verify real frame ingestion, MediaPipe landmarks, cooldown behaviour, and kiosk persistence, logging temperatures to ensure < 75 °C.
3. If tolerances remain tight in the field, consider small threshold adjustments (±5 %), toggling `consensus` 2/4↔3/6, or `frame_stride` 3↔4 as per acceptance criteria, but gather hardware evidence before changing defaults.
