# Módulo HIPOCAMPO — Design Recomendado (3a)

> **Status:** design travado, pronto para virar plano de implementação.
> **Lente:** Ciência (rigoroso, avaliável, publicável) — não produto.
> **Estilo:** de-risco, CPU/Mac-runnable, módulos pequenos + testes + experimento com REPORT.md/JSON (espelha `study.py`/`stage2`/`stage3`).
> **Fundação (não re-derivar):** `FUNDACAO_3a.md` §1.5 (linhas 23-24), §157 (módulo 7), §223. Toda a pesquisa de 5 frentes foi verificada adversarialmente; este documento usa os vereditos.

---

## 1. Resumo executivo

O HIPOCAMPO dá à LLM-córtex congelada o que ela não tem: **memória episódica persistente, de uma tentativa (one-shot), sem retreino e sem tocar os pesos da LLM**. A cadeia é fiel à neurociência verificada: **DG** faz *pattern separation* (projeção aleatória esparsa de expansão + k-Winners-Take-All, fixa, não-treinável) para ortogonalizar episódios parecidos; **CA3** faz *pattern completion* via Hopfield moderno — que é *literalmente* a atenção do Transformer, `X·softmax(β·Xᵀξ)` (Ramsauer et al. 2020, arXiv:2008.02217) — recuperando o item certo a partir de uma pista parcial/ruidosa num único update; e um **índice esparso** guarda *ponteiros* para o conteúdo (Indexing Theory; Teyler & Rudy 2007), não o conteúdo bruto. A escrita é **gateada por surpresa** (`−log p > μ_janela + γ·σ_janela`, padrão EM-LLM; Fountas et al. 2025) e a leitura combina **similaridade ∪ contiguidade temporal** (efeito de contiguidade humano; Howard & Kahana 2002). A LLM permanece 100% congelada; a memória entra como um **canal extra de informação** na mesma interface latente que o adaptador já validado consome (gate R² 0.87–0.97). O entregável científico não é "mais um sistema de memória" — é um **teste mecanicista com ablação causal** de cada peça hipocampal, medindo *quando e por quê* cada componente cérebro-fiel ganha sobre baselines fortes (RAG denso, EM-LLM), em recall de uma tentativa sob interferência alta e pista degradada.

---

## 2. Arquitetura concreta: DG → CA3 → índice

### 2.1 Visão da cadeia

```
                    escrita (one-shot, sem gradiente)                       leitura (cue parcial/ruidoso)
   experiência ──► LLMCortex.hidden ──► e (R^d)                  cue (R^d) ──► DG ──► k_dg_cue
                          │                  │                                          │
                   gating por surpresa       │ DG: pattern separation                   │ CA3: pattern completion
                   −log p > μ+γσ ? ──────────┤  k_dg = kWTA(P·e)                         │  k̂ = X_keys·softmax(β·X_keysᵀ·k_dg_cue)
                          │ sim               ▼                                          ▼
                          └──► store: X_keys.append(k_dg)         vizinho mais próximo de k̂ no índice
                                     content_store.append(e)             │
                                     meta.append({t, surprise})          ▼  resolve ponteiro
                                                                  e_recuperado (R^d)  +  contiguidade temporal (±n vizinhos)
                                                                         │
                                                                         ▼
                                                          adaptador recebe concat(z_atual, e_recuperado)
                                                          (LLM CONTINUA CONGELADA)
```

**Convenção fiel à Indexing Theory:** o hipocampo guarda o par `(chave_DG_esparsa, ponteiro)`. O **ponteiro** é um `content_id` que aponta para o embedding episódico real, guardado num `content_store` separado. CA3 **completa a chave** (não o conteúdo); o ponteiro então "sobe" o conteúdo. Exatamente Teyler & Rudy: índice esparso de loci, reativação reinstaura o padrão.

### 2.2 Escolhas travadas

| Decisão | Escolha travada | Justificativa / fonte |
|---|---|---|
| **DG (pattern separation)** | Projeção aleatória **fixa** (não-treinável) `P ∈ R^{D×d}`, `P ~ N(0,1)/√d`, seguida de **k-WTA** (`topk`, zera o resto, retém valor). | JL preserva distância; expansão+esparsidade separa similares. Fixa = isola a variável (não confunde ganho com treino) e é fiel ao espírito de-risco. Marcar como **simplificação** (DG real usa inibição aprendida). Litwin-Kumar et al. 2017 (expansão esparsa maximiza dimensionalidade — *cerebelo, usado por analogia ao DG*). |
| **Expansão D/d** | Começar `D = 4·d` (varrer 2×–10×). | Linha mossy-fiber/fly; é um knob do experimento. |
| **Esparsidade k** | `k ≈ 5%` de `D` (varrer 2%–10%). | k-WTA padrão; knob do experimento. |
| **CA3 (pattern completion)** | Hopfield moderno em **3 linhas de torch/numpy**: `k̂ = X·softmax(β·Xᵀ·ξ)`. **UM** update. | = atenção do Transformer (Ramsauer 2020, arXiv:2008.02217), provado. β alto → recupera 1 padrão isolado (regime desejado). |
| **β (nitidez do atrator)** | Knob varrido; default alto (ex. `β=8`/√D após normalização) para regime de ponto-fixo. | Baixo = mistura/generaliza; alto = item exato. Ligar à neuromodulação NE→temperatura depois (FUNDACAO §128, §152). |
| **Índice** | **Busca exata por produto interno em numpy/torch** (`X @ ξ`, `argsort`) no caminho default. | Para a escala do de-risco (10²–10⁴ memórias), exata é o **recomendado** pela própria literatura de ANN — não hnswlib. Honesto e sem dependência. |
| **Índice ANN (opcional, atrás de flag)** | `faiss-cpu` (`IndexFlatIP` → IVF/HNSW só se N≫10⁴). | Risco #3: ANN em vetores esparsos de alta-dim pode degradar recall — **medir recall vs exato antes de confiar**. |
| **Embedding `e`** | `LLMCortex.hidden(scene)` (já existe, `brain/cortex_llm.py`), L2-normalizado (cosine). | Reuso zero-código do córtex congelado (stub no servidor, Qwen2.5-0.5B MPS no Mac). |
| **Bibliotecas** | `torch` + `numpy` (já no `requirements.txt`). **Nenhuma dep nova obrigatória.** `faiss-cpu` opcional atrás de flag. | Mantém CPU/Mac-runnable e o estilo minimalista do repo. |

> **NÃO usar** a lib oficial `ml-jku/hopfield-layers` — está parada em PyTorch 1.6 (req. ≥1.5), risco real em torch 2.x; e a regra é trivial e queremos controle de β e da capacidade. (Verificado.)

### 2.3 Math central (pseudocódigo travado)

**DG — pattern separation:**
```python
# P fixa por seed (NÃO treinável); D = expand * d
P = torch.randn(D, d, generator=g) / d ** 0.5
def dg(e):                       # e: (d,) ou (B, d), L2-normalizado
    h = e @ P.T                  # expansão d -> D
    return kwta(h, k)            # zera tudo menos top-k (por |valor|); retém valor
```

**CA3 — pattern completion (Hopfield moderno = atenção):**
```python
def ca3_complete(cue_dg, X_keys, beta):
    # X_keys: (N, D) chaves armazenadas; cue_dg: (D,) pista parcial/ruidosa
    p = torch.softmax(beta * (X_keys @ cue_dg), dim=0)   # (N,) — UM update
    return X_keys.T @ p                                   # k̂: chave completada
```

**Gating por surpresa (escrita):**
```python
# S_t = -log p(x_t | x_<t); na fase toy, preditor barato de próximo-embedding, ou NLL nativa da LLM
boundary = S_t > (mu_window + gamma * sigma_window)      # janela móvel w (ex. w=32)
# gamma varrido em {0.5, 1.0, 1.5, 2.0}; defaults de γ/k_s/k_c a confirmar no repo em-llm/EM-LLM-model
```

**Leitura híbrida (similaridade ∪ contiguidade):**
```python
def read(cue, k_s, k_c, n_neighbors=1, beta=BETA):
    k_hat   = ca3_complete(dg(cue), X_keys, beta)         # completa a chave
    sims    = X_keys @ k_hat
    sim_ids = topk(sims, k_s)                              # etapa 1: similaridade
    cont_ids = []                                         # etapa 2: contiguidade temporal
    for i in sim_ids:
        for dd in range(-n_neighbors, n_neighbors+1):
            if dd != 0 and 0 <= i+dd < len(store): cont_ids.append(i+dd)
    cont_ids = dedup_keep_order(cont_ids)[:k_c]
    ids = sim_ids + cont_ids
    return [content_store[i] for i in ids]                # ponteiros -> conteúdo e
```
Defaults de partida (espelhando EM-LLM "4k+2k"): `k_s = 2·k_c` (ex. `k_s=8, k_c=4, n_neighbors=1`).

### 2.4 Conexão com a LLM-córtex congelada (interface 3a)

- **Escrita:** o loop rápido, a cada episódio saliente, chama `hippo.write(e=cortex.hidden(scene), surprise=S_t)`. Reusa `LLMCortex.hidden` (stage3) — zero código novo de córtex. Escreve **só se** o gate de surpresa disparar (novidade alta = vale memorizar).
- **Leitura:** dada a pista atual (hidden parcial/ruidoso), `hippo.read(cue) → [e_recuperado, vizinhos_temporais]`. O **adaptador já validado** (R² 0.87–0.97) recebe `concat(z_atual, e_recuperado)` (ou soma residual): o conteúdo recuperado entra como **contexto extra na MESMA interface latente** que o adaptador consome.
- **Invariante duro:** a LLM **nunca** é fine-tunada. A memória adiciona um canal de informação; não muda pesos do córtex. O único objeto treinável (e só na fase de consolidação, §3) é um adaptador externo pequeno.
- **Surpresa como moeda comum:** o `−log p` que gateia a escrita é o mesmo escalar que o barramento 3a já carrega na unidade de mensagem (FUNDACAO §61) e que acorda o System 2 (§137, §148).

---

## 3. Consolidação / replay — MVP vs depois. SR — dentro ou fora.

### 3.1 Successor Representation: **FORA do MVP** (decisão travada)

SR é refino posterior (módulo separado "mapa preditivo"), **não** parte do primeiro corte. Três razões, em ordem de peso:

1. **Tensão epistêmica com a tese do módulo.** O pitch do HIPOCAMPO é "memória de UMA tentativa, sem retreino". SR/SF aprende por **TD bootstrapping** — multi-amostra e incremental por construção (Dayan 1993; a própria existência de SF-episodic-control, arXiv:2111.03110, se justifica *porque* SR-por-TD é lento). Misturar viola exatamente a propriedade que queremos medir e publicar.
2. **O benefício episódico não depende de SR.** O competidor-referência EM-LLM entrega ganho episódico (supera RAG/InfLLM, escala a ~10M tokens) **sem SR** — usando surpresa + similaridade + contiguidade. Indexing (DG/CA3) + contiguidade já basta para o primeiro corte. SR adiciona *generalização preditiva e valor* — claim diferente, mais perto de planejamento/RL (território dos gânglios da base).
3. **Custo de superfície.** SR traz γ, α, base de φ, dim dos eigenvectors, loop TD e a questão ambígua de "o que é transição entre embeddings de LLM" — cada um um vetor de bug que atrasa o de-risco do módulo principal.

> **Nuance herdada da contiguidade:** o *buffer de contiguidade temporal* que ENTRA no MVP (§2.3) é o vizinho-temporal de EM-LLM, justificado por CMR/contiguidade humana (Howard & Kahana 2002) — **não** é SR. SR plena (M=(I−γT)⁻¹, eigenvectors ≈ grid cells) fica para o refino, com T empírico tirado *de graça* da ordem de escrita já registrada. Atenção: a novidade de "SR sobre LLM congelada" é incremental (SF-episodic-control já existe); o ângulo publicável tem de ser específico, não "SR+memória" genérico.

### 3.2 Consolidação / replay (CLS): **núcleo episódico no MVP; replay→adaptador como fase 2**

Princípio organizador (Spens & Burgess 2024; van de Ven et al. 2020; McClelland et al. 1995, *Psychol Rev* 102:419-457): **store episódico externo primeiro; o "lento" só é tocado por replay intercalado; a LLM nunca é fine-tunada.**

**No MVP (etapas 1–4):**
- Store episódico **append-only** (rápido, exato): `(key=k_dg, content_id, t, surprise)`. Esquecimento é estruturalmente impossível na LLM (congelada) e no store (append-only).
- Gating de escrita por surpresa (§2.3) já faz a esparsidade/forgetting-curve básica (só eventos salientes entram).

**Fase 2 (etapa 6, opcional, depois do núcleo robusto):**
- **Buffer de consolidação + gating de "sono"** (offline, disparado por regime — Singh, Norman & Schapiro 2022, PNAS 119:44; código `github.com/schapirolab/SinghNormanSchapiro_PNAS22`), **não** contínuo. Priorização por `priority = α·surprise + β·salience + γ·log(1+access) − δ·age`.
- **Dois alvos lentos a COMPARAR:**
  - **(a) Sumarização/compressão** (barata, sem gradiente): k-means sobre chaves → centróide+medoide por cluster. Store semântico muito menor, recuperação mais rápida, generaliza para queries próximas.
  - **(b) Adapter lento `g_θ: key → value`** (a hipótese forte do CLS): MLP pequeno treinado **offline com replay intercalado** (cada minibatch mistura novos + re-amostra dos antigos = "generative replay" honesto sem treinar gerador), update lento por **EMA** dos pesos (`θ_slow ← (1−η)θ_slow + η·θ_new`). LLM 100% congelada; só `g_θ` aprende. Em inferência: `value ≈ g_θ(key)` para o consolidado, **fallback ao store episódico exato** para o resto (= "dependência do hipocampo decai com a consolidação", Spens & Burgess).
- **Prior art próxima a citar e bater como baseline:** SuRe (arXiv:2511.22367) faz surprise-prioritised replay + fast/slow LoRA com merge EMA — confiar no *método* como referência, **auditar os números antes de reusar**. Nossa novidade não é o mecanismo, é o recorte (LLM congelada + store episódico externo + medição rigorosa do trade-off fidelidade↓/generalização↑/custo↓).
- **Anti-esquecimento catastrófico (a aposta central da fase 2):** o único lugar onde esquecimento pode ocorrer é `g_θ`. O experimento deve **provar** que sem replay `g_θ` esquece e com replay não (van de Ven et al. 2020 prevê isso; não medido no nosso setup com LLM congelada).

---

## 4. O experimento científico

**Pergunta científica central:** *quais componentes hipocampais (DG, CA3, contiguidade, surpresa-gate) causam ganho em recall episódico de uma tentativa, e em que regime (interferência alta / pista parcial / disambiguação temporal)?* Tudo CPU/Mac, com `results/REPORT.md` + JSON, ≥5 seeds (média ± desvio), estilo de-risco.

### 4.1 Tarefas

**Tarefa A — recall associativo controlado (toy, primeiro corte).** N embeddings episódicos (da LLM congelada via `cortex.hidden`, ou sintéticos com correlação controlada). Pista = padrão **corrompido** (mascarar p% das dims ou +ruído gaussiano σ). Mede recall a partir de pista degradada — força CA3.

**Tarefa B — EpBench (benchmark primário, publicável).** Portar o gerador sintético de `ahstat/episodic-memory-benchmark` (arXiv:2501.13121): "livros" de eventos = tupla **(tempo, local, entidade, conteúdo)** de um universo de ~100 valores, com **repetição temporoespacial proposital** → força disambiguação (= trabalho da DG). Leve, sem rede, sem contaminação web; gerar livros pequenos (50–100 eventos) p/ rodar no Mac. Tarefas = recall por cue (qualquer combinação da tupla); métricas F1 (LLM-as-judge) e Kendall's τ (cronologia). Achado deles a reproduzir: F1 cai de **0.81 (1 evento que casa) → 0.53 (6+ eventos)** — exatamente o regime de interferência que pattern separation deveria resolver.

### 4.2 Baselines OBRIGATÓRIOS (do mais fraco ao mais forte)

1. **Sem-memória / full-context** — LLM congelada com tudo no contexto quando cabe (teto offline).
2. **RAG denso** — embedding + top-k cosine sobre conteúdo bruto, **sem DG, sem CA3** (o RAG ingênuo; o baseline que todo paper de memória precisa bater).
3. **EM-LLM** (similaridade + contiguidade, **sem DG/CA3**) — o concorrente direto. Reimplementação mínima como baseline (o repo oficial é pesado p/ CPU).
4. **Cadeia completa DG→CA3→índice + contiguidade + surpresa-gate.**

### 4.3 Ablação causal (o coração científico) — ligar/desligar por flag

`RAG puro → +DG → +CA3 → +contiguidade → +surpresa-gate` e as ablações cruzadas:
- **CA3 sem DG** (isola o ganho do pattern separation: baseline 2 vs 4).
- **DG sem CA3** (isola o ganho do pattern completion: DG+k-NN vs cadeia completa).
- `surprise_write + sim_only` vs `write_all + sim+contig` vs completo (isola gating e contiguidade separadamente).
- **Oráculo** (fronteiras de evento ground-truth + recuperação ideal) = teto que separa "ideia ruim" de "instrumento ruim" — mesma lição do estágio 2 (a sonda valida o instrumento).

Reportar a **contribuição marginal** de cada componente.

### 4.4 Métricas que isolam o benefício cérebro-fiel

- **Acurácia top-1 / recall@k** do conteúdo correto **por nível de corrupção da pista** (0→90%) — mede CA3 completion.
- **F1 por número de eventos que casam o cue** (0/1/2/3-5/6+) — hipótese: **DG ganha mais no regime 6+ (interferência alta)**, onde o RAG denso degrada.
- **Capacidade observada:** N até a recuperação one-shot colapsar — testar empiricamente a previsão **2^(d/2)** (assintótica/otimista; padrões correlacionados e β finito derrubam muito — medir, não assumir).
- **lag-CRP** das memórias recuperadas: a contiguidade reproduz o pico-em-lag-pequeno com assimetria forward (Howard & Kahana 2002)? **Assinatura científica publicável** — testa se a contiguidade transfere como *mecanismo*, não só analogia.
- **Kendall's τ** (cronologia, onde contiguidade/SR deve brilhar).
- **Esparsidade do gating** (fração de eventos gravados) e **custo** (tamanho do store, ops/latência vs N).
- **Erro de reconstrução** do embedding recuperado.

**Varreduras:** corrupção da pista (0→90%); k do DG; β do CA3; expansão D/d; γ do gating; k_s/k_c; n_neighbors; N (até o ponto de virada de capacidade/custo).

### 4.5 O resultado publicável

> **A cadeia cérebro-fiel mantém recall one-shot fiel sob alta interferência / alta corrupção da pista onde o RAG denso degrada — e a ablação mostra *qual* componente causa o ganho e *em que regime*.** Concretamente: (i) DG dá ganho exatamente no regime 6+ eventos do EpBench (alta interferência); (ii) CA3 recupera de pista parcial onde a similaridade pura falha; (iii) a contiguidade reproduz o lag-CRP humano. EM-LLM mostrou que contiguidade só ajuda em **44% das tarefas** — nós mostramos *quando e por quê*.

**Falsificável (dizer no relatório):** se RAG denso bater a pilha cérebro-fiel em todos os regimes, a tese cai. Se DG não ajudar (embeddings da LLM já descorrelacionados o bastante — R² 0.87–0.97 sugere representação rica), *isso é em si um achado honesto e interessante* — medir antes de assumir.

---

## 5. Novidade honesta

**O que NÃO é novo (cada peça já existe isolada):**
- Surpresa Bayesiana como fronteira de evento + retrieval por similaridade ∪ contiguidade → **EM-LLM** (Fountas et al. 2025) já faz, e é o SOTA-vizinho. **Não tem DG nem CA3** (verificado: o paper não menciona pattern separation/completion/atrator).
- Indexing Theory literal (KG + Personalized PageRank como neocórtex+hipocampo) → **HippoRAG** (arXiv:2405.14831, NeurIPS 2024) já faz, mas é multi-hop factual via grafo, não single-trial episódico com atrator/completação.
- DG (top-k) + CA3 (Hopfield) + replay CLS → **VAE+MHN** (arXiv:2507.11393) e **HiCL** (arXiv:2508.16651, paper *distinto* do anterior) já fazem — mas em **MNIST com pesos treináveis**, não LLM congelada, não texto episódico.
- Paginação OS-like (MemGPT/Letta), notas Zettelkasten (A-MEM), extração de fatos (Mem0), memory stream recência+importância+relevância (Generative Agents) → todos **heurísticos/funcionais, sem compromisso cérebro-fiel** (sem separação/completação por atrator).
- Replay priorizado por surpresa + fast/slow EMA → **SuRe** (arXiv:2511.22367) já faz em LLM.

**O que É genuinamente novo (o delta publicável):**
> Ninguém integrou **DG (pattern separation) + CA3 (pattern completion como Hopfield moderno = atenção) + contiguidade temporal num store episódico instance-specific por cima de uma LLM CONGELADA, avaliado em single-trial recall com ablação causal de cada componente** — medindo *quais* mecanismos hipocampais causam ganho e *em que regime* (interferência alta / pista parcial / disambiguação temporal), mais o loop replay→adaptador-externo que conecta o hipocampo ao núcleo já validado sem destravar a LLM. O produto não é o sistema; é o **teste mecanicista causal**. Posicionar contra o paper de posição arXiv:2502.06975, que enumera as 5 propriedades de memória episódica (long-term storage, explicit reasoning, single-shot learning, instance-specific, contextual) que nenhum sistema reúne — DG/CA3/contiguidade mapeiam quase 1:1 nelas.

**Risco de novidade #1 (o maior):** se só re-empacotarmos EM-LLM+HippoRAG, **não é publicável**. A mitigação é a ablação causal sobre LLM congelada + o loop replay→adaptador, não o sistema em si. Fazer busca confirmatória dirigida lendo HEMA (2504.16754), BMAM (2601.20465), "Hippocampus" (2602.13594) — verificados, **nenhum colide** (são memória funcional/orquestrada por LLM, sem mecanismo de separação/completação).

---

## 6. Componentes / arquivos a construir (reusando o repo)

Espelha `study.py`/`stage2`/`stage3`: módulos pequenos em `brain/` + experimento em `stage4/` + testes em `tests/`. Docstrings em pt-BR. **Sem dep nova obrigatória.**

| Arquivo | Conteúdo | Reuso |
|---|---|---|
| `brain/hippocampus/__init__.py` | API pública `Hippocampus(write, read, ...)` | — |
| `brain/hippocampus/dg.py` | `dg(e)`, `kwta(h,k)`, `P` aleatória fixa | — |
| `brain/hippocampus/ca3.py` | `ca3_complete(cue, X, beta)` (Hopfield = atenção, 3 linhas) | — |
| `brain/hippocampus/index.py` | store exato (numpy/torch) + `content_store` + ponteiros; flag `backend="exact"|"faiss"` | — |
| `brain/hippocampus/surprise_gate.py` | `S_t = −log p`, limiar móvel `μ+γσ` | preditor barato; depois NLL nativa de `LLMCortex` |
| `brain/hippocampus/contiguity.py` | buffer de vizinhos temporais ±n (fila k_c) | meta `t` do store |
| `brain/hippocampus/replay.py` *(fase 2)* | buffer de consolidação, priorização, `g_θ` + EMA, sumarização | adaptador validado |
| `stage4/hippo_recall.py` | experimento Tarefa A + ablação + varreduras → REPORT.md/JSON | `cortex_llm.LLMCortex.hidden`, `probe._probe_r2_on_features` |
| `stage4/epbench_gen.py` | port mínimo do gerador EpBench (sintético, sem rede) | — |
| `stage4/run_mac.py` | runner MPS (Qwen2.5-0.5B real), espelha `stage3/run_mac.py` | `LLMCortex(backend="mps")` |
| `tests/test_hippocampus.py` | asserts comparativos (estilo `test_probe.py`) | — |
| `results/REPORT.md` + `results/hippocampus.json` | saída padrão do repo | — |

**Testes-chave (pytest, estilo `test_probe.py` — asserts comparativos):**
- `assert recall_full_chain > recall_knn_cru` sob corrupção alta.
- `assert recall_dg_ca3 >= recall_sim_only` (contiguidade/completação ajuda).
- CA3 completa pista corrompida: `recall(cue_corrompido) ≈ recall(cue_limpo)` até p% moderado.
- gating dispara em fronteira de evento e silencia em stream previsível.
- lag-CRP **decrescente** com o lag (assinatura de contiguidade).
- capacidade observada **cresce com d**.

### 6.1 Ordem por etapas (cada uma robusta sozinha)

1. **DG + índice exato + escrita/leitura por similaridade pura** (= RAG denso com expansão DG). Testa pattern separation isolado. *Entregável robusto: store funcional + ablação DG-on/off.*
2. **+ CA3 (Hopfield = atenção)** para completar pista parcial/ruidosa. Testa pattern completion. *Entregável: curva recall × corrupção, CA3-on/off.*
3. **+ contiguidade temporal** (buffer ±n). Testa similaridade ∪ contiguidade + lag-CRP. *Entregável: ablação contiguidade + assinatura lag-CRP.*
4. **+ gating de escrita por surpresa**. Testa esparsidade da escrita. *Entregável: cadeia episódica completa + EpBench (Tarefa B) + REPORT.md/JSON com a matriz de ablação completa e baselines (RAG denso, EM-LLM mínimo).* **← MVP completo e publicável.**
5. **Ponte ao núcleo (downstream).** Adaptador recebe `concat(z, e_recuperado)`; medir se a tarefa downstream melhora, não só o recall isolado. *Entregável: prova de valor de sistema, não de componente — a afirmação que conecta o hipocampo ao núcleo R²0.87–0.97.*
6. **Consolidação/replay (fase 2, opcional).** `replay.py`: sumarização vs `g_θ`+EMA, ablação replay-on/off (anti-esquecimento), varredura de N até o ponto de virada. *Entregável: curva de esquecimento por condição + trade-off fidelidade/generalização/custo.*

---

## 7. Riscos duros

1. **Capacidade exponencial 2^(d/2) é teórica/assintótica.** Padrões correlacionados + β finito reduzem muito a capacidade efetiva e geram **estados metaestáveis espúrios** (recupera a média de um subconjunto, não o item). Mitigar com β alto + DG (descorrelaciona) — e **medir**, não assumir.
2. **DG por projeção aleatória fixa não é ótimo** e pode ser **supérfluo** se os embeddings da LLM de fronteira já forem suficientemente descorrelacionados (R² 0.87–0.97 sugere representação rica). Pattern separation pode não dar ganho — isso é, em si, um achado honesto. Marcar como simplificação (DG real usa inibição aprendida); medir antes de assumir.
3. **CA3 pode degenerar em "só mais um retrieval por similaridade"** se o teste não **forçar pista parcial/degradada**. A diferença (recuperação por atrator de cue ruidoso) só aparece com `Recall@k` sob corrupção forte — senão vira RAG com passos extras.
4. **Redundância vs EM-LLM/HippoRAG (risco de novidade).** Se for só re-empacotamento, não publica. Mitigação: ablação causal + LLM congelada + loop replay→adaptador.
5. **Surpresa ≠ relevância.** Alta NLL pega tokens raros/ruído, não só eventos importantes (gravar lixo saliente). Limiar adaptativo `μ+γσ` ajuda mas não resolve; refino por grafo (modularidade, EM-LLM §4) é flag opcional. npj 2023 (s41539-023-00166-x): o efeito de prediction-error no encoding é **dependente do desfecho** (não um efeito limpo) — citar com essa ressalva.
6. **ANN sobre chaves DG esparsas pode degradar recall.** HNSW sofre em alta dimensionalidade; FAISS lida melhor. Para a escala do de-risco, **busca exata é o recomendado** — ANN só quando N≫10⁴, e **medir recall vs exato antes de confiar**. (Corrige a racional "hnswlib brilha em índices pequenos", que não é fato de performance estabelecido.)
7. **Instrumento toy vs sinal real.** Na fase toy, a qualidade da surpresa vem de um preditor barato; preditor fraco → fronteiras ruins que não refletem a LLM. Manter o **teto-oráculo** para separar "ideia ruim" de "instrumento ruim" (lição do estágio 2).
8. **LLM-as-judge (EpBench) introduz ruído/viés.** Ancorar com exact-match onde possível e checar concordância (EpBench reporta >97% c/ humano, mas verificar no nosso setup).
9. **Custo de leitura O(N·d).** Acima de ~10⁴ memórias a busca exata fica lenta → flag de ANN. Não é problema no regime do MVP.
10. **Integração com EM-LLM é não-trivial** (ele vive no KV-cache; nossa consolidação sai do KV para `g_θ`, mudando a semântica de recuperação). Risco de interface, não medido no de-risco — manter separação estrita do núcleo episódico.
11. **Datas futuras / fontes 2025-2026** (SuRe, BMAM, "Hippocampus", EpBench-ICLR): verificadas como reais, mas **não basear *números* de design** nelas sem leitura do método; e não afirmar venue (ex. "ICLR 2025" do EpBench) sem checar OpenReview.

---

### Apêndice — correções da pesquisa aplicadas neste design

- **Hopfield moderno = atenção → Ramsauer 2020 (arXiv:2008.02217), NÃO Krotov** (Krotov-Hopfield 2016 só introduziu capacidade super-linear). Atribuição correta em FUNDACAO §1.5 e §223.
- **Litwin-Kumar 2017** é cerebelo (granule/Kenyon cells), usado **por analogia** ao DG — não medida direta do DG.
- **hnswlib > faiss "em índices pequenos"** era exagero: a literatura recomenda **busca exata/Flat** em escala pequena (reforça nossa escolha default).
- **SuRe (arXiv:2511.22367)** é real e verificável — citar como prior art / baseline, auditar números.
- **EM-LLM** título canônico = "Human-**inspired** Episodic Memory for Infinite Context LLMs"; "EM-LLM não consolida" é **inferência por ausência** (store cresce), não afirmação citada.
- **SF-NEC:** arXiv:2111.03110 (2021) e s41598-024-65687-w / arXiv:2508.16651 são trabalhos **distintos** — não conflar.
- **EpBench-ICLR** e demais números dos autores (EM-LLM +30,5%/44%, EpBench 0.81→0.53, HippoRAG +20%, Mem0 +26%) verificados como reais, mas **dos autores, sem replicação independente** — o nosso experimento é a replicação mínima.
