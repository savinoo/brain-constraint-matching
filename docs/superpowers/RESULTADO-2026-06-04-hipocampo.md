# Hipocampo — resultado: o micro-mecanismo DG/CA3 é redundante (e isso importa)

Ablação causal (5 seeds, CPU) do micro-circuito hipocampal — **DG** (pattern separation:
projeção aleatória + k-WTA) e **CA3** (pattern completion: Hopfield = atenção) — contra **RAG**
(cosseno sobre o embedding cru), em recall de uma tentativa sob pista corrompida e interferência.

## Resultado: RAG ganha em TODO regime; DG atrapalha

### Recall × corrupção da pista (d=64, interferência média)
| condição | corr=0.0 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| rag | 1.00 | 1.00 | 1.00 | 0.99 | **0.54** |
| dg | 1.00 | 1.00 | 0.99 | 0.91 | 0.40 |
| dg+ca3 | 1.00 | 1.00 | 0.98 | 0.90 | 0.38 |

### Regime adversarial PRÓ-cérebro-fiel (d=16, interferência extrema, pista parcial)
| condição | corr=0.3 | 0.5 | 0.7 |
|---|---|---|---|
| rag | **0.77** | **0.52** | **0.29** |
| dg | 0.38 | 0.21 | 0.12 |
| dg+ca3 | 0.32 | 0.18 | 0.13 |

Mesmo onde o pattern separation *deveria* brilhar (baixa dimensão, itens quase idênticos,
pista degradada), o RAG **dobra** o recall do DG. CA3 não recupera. **Null robusto.**

## Por que (a explicação mecanicista, honesta)

O DG/CA3 evoluíram para um substrato **ruidoso, de baixa precisão e leitura limitada** (neurônios).
Sobre **embeddings densos e contínuos com cosseno exato**, eles são redundantes ou nocivos:
- O **k-WTA do DG joga fora 95% das dimensões** — exatamente a informação que o cosseno usa para
  separar vetores quase-ortogonais. Sparsificar uma representação já rica **destrói sinal**.
- A **completação do CA3** (atrator sobre chaves esparsas) não bate o cosseno suave sobre vetores densos.

## Por que isso é um achado importante (não um fracasso)

Fecha um fio que atravessa o projeto inteiro:
1. Estágio 1–2: **features congeladas ricas** (LLM, R² 0.87–0.97) tornam o adaptador viável; features
   profundas aleatórias falham → **a qualidade da representação é o que importa**.
2. Hipocampo: a mesma representação rica torna o **pattern separation/completion redundante**.

**Princípio (publicável):** *portar a FUNÇÃO e a ORGANIZAÇÃO do cérebro ≠ portar micro-mecanismos que
são soluções para restrições biológicas que a computação densa de alta precisão não tem.* Alguns
circuitos cérebro-fiéis são **respostas a limitações de neurônios**, não a melhor solução de engenharia
quando você tem floats de 32 bits e produto interno exato. Este é um **ângulo de paper honesto**:
*"Quando a memória cérebro-fiel ajuda uma LLM? Uma ablação causal mostrando que separação/completação
de padrões são redundantes sobre representações ricas."*

## A virada para o módulo (o que de fato tem valor)

O valor de "hipocampo para LLM" **não está no micro-circuito DG/CA3** — está no **nível de sistema**:
- **store episódico externo persistente** (memória de uma tentativa, sem retreino),
- **escrita gateada por surpresa** (só eventos salientes),
- **recuperação por similaridade ∪ contiguidade temporal** (sequência/contexto).

Isso é essencialmente **RAG + contiguidade = EM-LLM** — que já ajuda (validado por terceiros), mas
**não é novidade nossa**, e o nosso DG/CA3 não o melhora. Construímos e testamos o store + gate +
contiguidade (código pronto em `brain/hippocampus/`), mas o **benefício deles é de sequência/temporal**
(não medido aqui — recall@1 não testa contiguidade) e é mérito conhecido do EM-LLM.

## Caveat honesto
Testamos o **micro-mecanismo DG/CA3 para recall** — null robusto. Não re-litigamos o benefício
**sistema-nível** (store+surpresa+contiguidade ≈ EM-LLM), que é de tarefa temporal e já é da literatura.
A contribuição científica honesta aqui é o **null causal de DG/CA3 + o princípio** sobre quais
mecanismos cérebro-fiéis vale portar.

*Código: `brain/hippocampus/` (dg, ca3, index, contiguity, surprise_gate, API), `stage4/hippo_recall.py`.
34 testes verdes.*
