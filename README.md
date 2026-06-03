# Constraint Matching: Which Brain Mechanisms Transfer to a Frozen-LLM Cortex?

Code and result logs for the position paper of the same name
([`paper/main.pdf`](paper/main.pdf)).

**Thesis.** Brain-faithful cognitive architectures port neural mechanisms by functional
analogy. We propose a narrower criterion — *constraint matching*: the presence, in the
target substrate, of the constraint that motivated a mechanism is **necessary, not
sufficient**, for porting its implementation to help. On dense, high-precision compute with
exact addressing (a frozen LLM and its adapters), porting the *function* often helps; porting
the *implementation* can be redundant or harmful when the constraint that motivated it is absent.

## The five experiments

| # | Claim | Run | Result doc |
|---|---|---|---|
| E1 | Freezing recovers co-training when the info is present; a linear probe predicts it | `python -m stage4.hippo_recall` (representation toy in `stage1`–`stage3`) | `docs/superpowers/RESULTADO-2026-06-03-toy-derisk.md` |
| E2 | Pretraining quality matters (real frozen model); probe is necessary, not sufficient | `stage2/stage2_probe.py` (Apple Silicon / MPS) | `docs/superpowers/RESULTADO-2026-06-03-stage2-probe.md` |
| E3 | DG/CA3 micro-circuit is redundant/harmful; collapses on real LLM embeddings | `python -m stage4.hippo_recall`; `stage4/real_embed_mac.py` (MPS) | `docs/superpowers/RESULTADO-2026-06-04-hipocampo.md` |
| E4 | Basal-ganglia RL helps where a frozen LLM can't learn from reward (weak test) | `python -m stage5.bg_rl` | `results/basal_ganglia.md` |
| E5 | **Pre-registered**: eligibility traces beat a competent baseline under delayed reward | `python -m stage5.delayed_credit` | `docs/superpowers/RESULTADO-2026-06-04-eligibility.md` |

E5 was pre-registered: the prediction (`docs/superpowers/PRE-REGISTRO-2026-06-04-eligibility.md`)
is committed to git **before** the result commit.

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pytest matplotlib
pytest -q                          # the module/unit tests
python -m stage4.hippo_recall      # E3 ablation (synthetic)
python -m stage5.bg_rl             # E4
python -m stage5.delayed_credit    # E5 (pre-registered)
python paper/make_figs.py          # regenerate the paper figures
```

The real-model experiments (E2, and E3 on real embeddings) need Apple Silicon / MPS and
`transformers`; see `stage2/` and `stage4/real_embed_mac.py`. Everything else runs on CPU.

## Build the paper

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Honest scope

This is a position paper with **toy-scale** evidence. The strong, non-trivial test (constraint
absent → don't port) rests on one micro-mechanism (DG/CA3). Threats to validity are stated in
full in the paper (§ Threats to validity). The contribution is a falsifiable criterion plus a
cheap pre-qualification instrument (the linear probe), not a system or scale.

## AI collaboration

This work was developed in collaboration with an AI research assistant (Anthropic's Claude),
which contributed to experiment design, implementation, analysis, and drafting under the
author's direction. All claims, code, and results are the author's responsibility.

## License

MIT (code), see [`LICENSE`](LICENSE).
