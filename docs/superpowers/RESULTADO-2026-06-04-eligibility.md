# RESULTADO — pré-registro de traços de elegibilidade (commit POSTERIOR ao pré-registro)

Predição registrada em `PRE-REGISTRO-2026-06-04-eligibility.md` (commit anterior no git).
Teste forte e cego do lado positivo da regra de casamento de restrição.

## Resultado (5 seeds, acurácia greedy, orçamento fixo de 150 episódios)

| | K=0 | K=4 | K=8 | K=16 | K=24 |
|---|---|---|---|---|---|
| TD de um passo (λ=0, baseline competente) | 1.00 | 0.07 | 0.17 | 0.27 | 0.30 |
| TD(λ=0.9) (traços, cérebro-fiel) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vantagem | +0.00 | +0.93 | +0.83 | +0.73 | +0.70 |

## A predição se manteve?

- **Predição 1 — TD(λ) vence o TD de um passo na recompensa atrasada: ✅ CONFIRMADA, forte.**
  TD(λ)=1.00 em todo gap; TD de um passo cai a 0.07–0.30 quando há atraso (K≥4), abaixo até do
  acaso (0.25) em K=4.
- **Predição 3 — em K≥8, TD(λ)≥0.9 e TD de um passo <0.6: ✅ CONFIRMADA.** (1.00 vs 0.17–0.30.)
- **Predição 2 — a vantagem CRESCE monotonicamente com K: ⚠️ PARCIAL.** A direção acertou (K=0
  empata; qualquer atraso K>0 abre uma vantagem grande). Mas não é crescimento suave: a vantagem
  é máxima em K=4 (+0.93) e **decresce** levemente até K=24 (+0.70), porque o TD de um passo fica
  *menos errado* (mais perto do acaso) conforme K cresce, em vez de mais errado. Registro como
  acerto da direção, erro da forma.

## Leitura honesta

O teste é **forte** (baseline competente, não incapaz como no E4) e **cego** (predição commitada
antes). O mecanismo cérebro-fiel — traços de elegibilidade — supera o baseline competente
exatamente no regime onde a restrição (crédito temporal através de um gap) está presente, e empata
quando ela está ausente (K=0). Isso é o que a regra de casamento de restrição prevê para o lado
positivo, agora **prospectivamente**, não por racionalização post-hoc.

Ressalvas que permanecem: tabular e toy (não escala); a sub-predição quantitativa (forma da curva)
errou; e "constraint presente → ajuda" continua sendo necessidade demonstrada num caso, não lei.
Mas o paper sai de "lado positivo só com baseline incapaz (E4)" para "lado positivo com baseline
competente e predição cega que se manteve na direção". É o upgrade de explicativo para preditivo
que o §5 do paper pedia.
