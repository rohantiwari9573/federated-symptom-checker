# About this repository

This is Rohan Tiwari's personal working copy of the Privacy-Preserving
Federated Symptom Checker project, published independently of the
original repository at `saarcasm/Federated-Symptom-Checker` (no shared
git history, no fork relationship).

It contains work verifying and extending the project against
`Federated_Symptom_Checker_Paper_Revised.pdf`:

- Reproduced the paper's Table II (tabular symptom model) exactly from
  `benchmarks/real_comparison_experiment.py`.
- In-browser (on-device) inference for the symptom model, with a parity
  test against PyTorch (`client-app/model.js`,
  `tests/js/model_parity.test.js`).
- Removed fake/mock prediction fallbacks from the web and Android clients.
- Centralized and federated experiments for the skin (HAM10000) and
  respiratory (ICBHI) branches, with results and known gaps tracked in
  `docs/paper_claims_ledger.md`.
- A claim-by-claim ledger of what the paper states versus what is
  verified in code, including paper errata to correct.

See `docs/paper_claims_ledger.md` for the current, honest status of
every claim in the paper.
