# PRÉ-REGISTRO — teste forte do lado positivo da regra de casamento de restrição

> **Este documento declara a predição ANTES de rodar o experimento.** O commit que o
> introduz precede, no histórico git, o commit que traz os resultados. É o teste que
> falta no position paper (`POSITION-2026-06-04-casamento-de-restricao.md`, §5): a regra
> foi *derivada* dos casos, não validada prospectivamente. Aqui ela faz uma predição cega.

## A regra (a ser testada prospectivamente)
*Porte a implementação de um mecanismo cerebral só se a restrição que ele resolve existe no
substrato — necessidade, não suficiência. Restrição presente → portar a função/implementação
pode ajudar.*

## O mecanismo (não testado antes neste projeto)
**Traços de elegibilidade — TD(λ) com λ>0.** Mecanismo cérebro-fiel de **atribuição de crédito
temporal**: uma marca decadente em cada par (estado, ação) recém-visitado permite que uma
recompensa distante atualize tudo o que a precedeu. Correspondência neural: elegibilidade
sináptica + plasticidade gateada por dopamina nos gânglios da base (Sutton & Barto; a base do
TD(λ)). O **baseline competente** é TD de um passo (λ=0) — não um agente incapaz (corrige a
fraqueza do E4, cujo baseline não podia vencer por construção).

## A restrição que ele resolve
**Atribuição de crédito através de um intervalo temporal** entre a ação e a recompensa.

## A restrição está presente no substrato de destino?
**Sim, por desenho da tarefa.** A tarefa terá recompensa **atrasada** por um gap de K passos
distratores entre a decisão e o reward. O gap K é a variável independente.

## PREDIÇÃO (declarada antes de rodar)
1. **TD(λ=0.9) aprende mais rápido que TD de um passo (λ=0)** na tarefa de recompensa atrasada
   (menos episódios para atingir o mesmo desempenho, ou desempenho maior em orçamento fixo).
2. **A vantagem de TD(λ) sobre TD de um passo CRESCE com o gap K.** Em K=0 (recompensa imediata)
   as duas devem empatar; conforme K aumenta, a separação aumenta.
3. Quantitativo declarado: em K≥8, espero TD(λ) ≥ 0.9 de acurácia greedy num orçamento onde
   TD de um passo fica abaixo de 0.6. (Número declarado para ser checável; se errar a magnitude
   mas acertar a direção, registro como acerto parcial.)

## Condição de FALSIFICAÇÃO
- Se TD(λ) **não** superar o TD de um passo, ou a vantagem **não** crescer com K, a predição falha.
  Como o baseline é competente, isso seria evidência **contra** o valor de portar traços de
  elegibilidade neste substrato — e, por extensão, contra o lado positivo da regra na sua forma forte.
- A predição é registrada **antes** de qualquer execução. O resultado vai num doc separado, em
  commit posterior, dizendo explicitamente se a predição se manteve.

## Por que isto fortalece o paper
O E4 atual testa o lado positivo de forma fraca (baseline incapaz; RL genérico, não um mecanismo
cérebro-fiel específico). Este pré-registro testa o lado positivo de forma **forte e cega**: um
mecanismo cérebro-fiel específico (traços de elegibilidade) contra um baseline **competente**
(TD de um passo), com a predição declarada antes. Acerto → a regra previu, não só explicou.
Erro → registramos honestamente como evidência contrária.
