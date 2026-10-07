# Paper claims ledger

Source: Federated_Symptom_Checker_Paper_Revised.pdf

Statuses: **Met** (code reproduces it), **Fixable** (code change needed), **Erratum** (paper contradicts itself or describes removed behaviour), **Out of scope** (needs new datasets, training, or hardware).

## Tabular experiment (Section V, Table II, Fig. 7-9)

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 1 | Centralized model reaches 100% test accuracy | Met | `benchmarks/results/comparison_results.csv` (re-run) |
| 2 | FedAvg without DP reaches 99.8% at round 2 and 100% from round 4 | Met | `benchmarks/results/run_log.txt` |
| 3 | FedAvg + DP accuracy: 0.0467 / 0.0335 / 0.2358 / 0.3841 at eps 0.5 / 1.0 / 2.0 / 5.0 | Met | Re-run reproduces these exactly |
| 4 | Macro-F1 and loss reported | Met | same CSV |
| 5 | 4,920 records, 41 classes, 132 symptoms, 80/20 split (3,936 / 984) | Met | `benchmarks/real_comparison_experiment.py` |
| 6 | Dirichlet alpha = 0.5 across 10 clients | Met | same script |
| 7 | 10 rounds, 1 local epoch, batch size 32 | Met | same script (`ROUNDS`, `LOCAL_EPOCHS`, `BATCH_SIZE`) |
| 8 | Per-client noise calibration via Opacus RDP accountant | Met | `create_dp_config` via `client_noise_multiplier` |
| 9 | delta = 10^-5 for the DP experiment | Met | `DELTA = 1e-5` in the script |
| 10 | Table II produced by Opacus DP-SGD | Erratum (wording) | The DP step is manual per-sample clipping + Gaussian noise. Opacus is used for noise calibration only. |

## Erratum candidates (paper text to correct)

- **E1.** Section III-C says delta defaults to 1/n. Section III-D says delta = 10^-5. Table II uses 10^-5.
- **E2.** Section III-B and Fig. 2 say 131 symptoms. The data has 132 (Section III-D says 132).
- **E3.** Section V-D says "naive per-round composition." The code composes the noise over all rounds (`epochs = LOCAL_EPOCHS * ROUNDS`).
- **E4.** Section IV describes a "built-in demonstration fallback" that supplies illustrative predictions. This was removed on purpose.
- **E5.** Fig. 6 shows "121 / 500" and "107 / 500" rounds. The experiment runs 10 rounds, so the dashboard figures do not match the experiment.
- **E6.** Fig. 4 and Fig. 5 captions say "HAM10000-trained" and "ICBHI-based" branches. No such models are trained or served.
- **E7.** The abstract and contributions say all three modalities are "supported end to end." Sections V-D and VI defer the skin and respiratory evaluation.
- **E8.** Algorithm 1 caption says the pipeline is implemented end to end including ExecuTorch/LiteRT redeployment. Section VI lists this as future work. No export code exists (not yet verified by grep).

## Engineering status (not in the paper's numbers)

- **Flower path** (`server/run_simulation.py`, default `--local_epochs 3`, `--rounds 20`): reached about 58% on the symptom MLP. This is not the configuration in the paper and is not what Table II reports.
- **Browser inference (symptom branch):** done. `client-app/model.js` runs the eps=5.0 model on-device. Parity with PyTorch: 984/984 argmax matches, max logit difference 1.7e-5 (`node tests/js/model_parity.test.js`). The dashboard reads ε, round, and accuracy from the same file, so the displayed values match the shipped weights.
- **Android:** `model.js` and the model file are copied into the assets, but `android-app/.../app.js` is not wired to use them yet. Android still calls the API.
- **Skin (HAM10000), centralized:** pretrained MobileNetV3-Small with BatchNorm kept. Adam, lr 3e-4, class-weighted loss, flip/rotation augmentation, cosine decay, 40 epochs. Test accuracy 86.0%, macro-F1 0.80 (`benchmarks/results/skin/sanity_adam_cosine_final.json`). Split is random, not grouped by lesion, so this may be optimistic. Target of about 90% not reached.
- **Skin, federated and DP:** earlier runs collapsed to the majority class under the GroupNorm conversion and SGD lr 0.01. Not yet redone with the centralized settings. No federated skin claim is supported yet.
- **Respiratory (ICBHI):** `benchmarks/respiratory_federated_experiment.py` written and started. No results yet.
- **Skin and respiratory in the browser:** not exported. The page shows the "research prototype" notice.
