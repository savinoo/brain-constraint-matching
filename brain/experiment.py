import numpy as np
import torch
from brain.data import collect_oracle_dataset
from brain.train import pretrain_cortex, train_cotrain, train_freeze_adapt, train_monolith


def _np(x):
    return np.asarray(x, dtype=np.float32)


def _t1(x):
    """numpy 1D -> tensor (1, D)."""
    return torch.tensor(_np(x)).unsqueeze(0)


def run_experiment(n_episodes_data=200, epochs=200, eval_episodes=50, Ks=(1, 5, 10), seed=0):
    from brain.bridge import eval_policy

    data = collect_oracle_dataset(n_episodes=n_episodes_data, seed=seed)

    # treinar as 3 condições
    (cx_co, ctrl_co), _ = train_cotrain(data, epochs=epochs, seed=seed)
    cx_frozen, _ = pretrain_cortex(data, epochs=epochs, seed=seed)
    (_, adapter, ctrl_fa), _ = train_freeze_adapt(data, cx_frozen, epochs=epochs, seed=seed)
    monolith, _ = train_monolith(data, epochs=epochs, seed=seed)

    for m in (cx_co, ctrl_co, cx_frozen, adapter, ctrl_fa, monolith):
        m.eval()

    @torch.no_grad()
    def cotrain_cortex(scene):
        return cx_co(_t1(scene)).squeeze(0).numpy()

    @torch.no_grad()
    def cotrain_policy(pos, z):
        return ctrl_co(_t1(pos), torch.tensor(_np(z)).unsqueeze(0)).squeeze(0).numpy()

    @torch.no_grad()
    def fa_cortex(scene):
        z = cx_frozen(_t1(scene))
        return adapter(z).squeeze(0).numpy()

    @torch.no_grad()
    def fa_policy(pos, z):
        return ctrl_fa(_t1(pos), torch.tensor(_np(z)).unsqueeze(0)).squeeze(0).numpy()

    def mono_cortex(scene):
        return _np(scene)

    @torch.no_grad()
    def mono_policy(pos, scene_z):
        return monolith(torch.tensor(_np(scene_z)).unsqueeze(0), _t1(pos)).squeeze(0).numpy()

    conds = {
        "cotrain": (cotrain_policy, cotrain_cortex),
        "freeze_adapt": (fa_policy, fa_cortex),
        "monolith": (mono_policy, mono_cortex),
    }

    results = {}
    for name, (pol, cor) in conds.items():
        per_K = {}
        for K in Ks:
            per_K[K] = eval_policy(pol, cor, K=K, n_episodes=eval_episodes, seed=10_000 + K)
        results[name] = per_K

    # veredito: recuperação = (A - C) / (B - C) em retorno, por K. ~1.0 => congelar não custa.
    recovery = {}
    for K in Ks:
        a = results["freeze_adapt"][K]["return"]
        b = results["cotrain"][K]["return"]
        c = results["monolith"][K]["return"]
        denom = (b - c)
        recovery[K] = float((a - c) / denom) if abs(denom) > 1e-6 else float("nan")

    results["verdict"] = {
        "recovery_by_K": recovery,
        "note": "recovery~1 => congelado+adaptador iguala co-treino; <0.5 => congelar custa caro; "
                "queda forte quando K cresce => fragilidade sob a ponte de duas velocidades.",
    }
    return results


def main():
    res = run_experiment()
    print("\n=== Retorno medio (maior = melhor) e sucesso por condicao/K ===")
    for cond in ("cotrain", "freeze_adapt", "monolith"):
        for K, m in res[cond].items():
            print(f"{cond:13s} K={K:<3d} return={m['return']:8.3f}  success={m['success']:.2f}")
    print("\n=== VEREDITO -- recuperacao do gap (freeze_adapt vs co-treino) ===")
    for K, rec in res["verdict"]["recovery_by_K"].items():
        print(f"  K={K:<3d} recovery={rec:.2f}")
    print("  " + res["verdict"]["note"])


if __name__ == "__main__":
    main()
