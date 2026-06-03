# Documento de Design — Arquitetura Cognitiva Cérebro-Fiel (Recorte 3a)

**Data:** 2026-06-03
**Status:** Design para revisão (etapa ② do roteiro)
**Fontes:** [`FUNDACAO_3a.md`](../../../FUNDACAO_3a.md) (pesquisa citada e verificada) + [`RESOLUCOES_3a.md`](../../../RESOLUCOES_3a.md) (5 buracos resolvidos)
**Disciplina:** toda afirmação carregada de peso sobreviveu a verificação adversarial. Saltos do arquiteto estão marcados como **extrapolação de design**.

---

## 0. Resumo executivo

Construir um agente cujo "cérebro" é uma **arquitetura cognitiva fiel aos princípios e à organização do cérebro** — não uma cópia molécula-por-molécula. Uma **LLM de fronteira congelada** ocupa o papel do córtex pré-frontal/associativo (System 2, lento). Em volta, **módulos especializados** rodam cada um o algoritmo que a neurociência atribui à sua região (visão/áudio preditivos, hipocampo episódico, amígdala afetiva, gânglios da base de reforço, cerebelo de timing), conectados por um **barramento de mensagens com gramática única**: *predição desce, erro sobe, precisão = ganho = atenção = neuromodulação*. Um **loop sensório-motor rápido** (System 1) fecha percepção→ação sem esperar a LLM; as duas velocidades conversam por **chunking assíncrono**.

**A contribuição original** (calibrado pelo cenário competitivo — ver §10) **não é nenhum componente, nem "LLM congelada como córtex"** — isso já tem nome (*Frozen-LM Conditioning*: NeuroLoRA, PaceLLM, Brainstacks, 2025-2026) e benchmarks. É **só a síntese cérebro-completa integrada** (LLM congelada + o conjunto dos módulos + duas velocidades físicas + gramática única), e mesmo essa é uma **aposta condicionada a duas hipóteses não testadas**: (1) que a síntese rende mais que a soma das partes; (2) que congelar + adaptar não custa o ganho do co-treino. Defensável como aposta — não como feito.

---

## 1. Visão, objetivo e não-objetivos

### 1.1 O que é
Um sistema que **percebe um ambiente continuamente, lembra, prioriza pelo que importa, e age** — com cada função ancorada na função cerebral correspondente, e a LLM fazendo a deliberação/linguagem/planejamento de horizonte longo.

### 1.2 Por que (o destino)
Provar uma arquitetura nova e demonstrável (publicável e potencialmente comercial), construída **por etapas verificáveis**, começando por um módulo de percepção que roda sozinho.

### 1.3 Não-objetivos (o teto honesto, travado)
- **NÃO** é simulação molecular/spiking completa de um cérebro humano (sem conectoma humano, sem entendimento algorítmico de cada região — ver Human Brain Project).
- **NÃO** assume que "a LLM é o córtex" como identidade mecanística. A literatura diz **convergência parcial** de princípios (Goldstein et al. 2022; Tuckute et al. 2024). É uma **aposta arquitetural útil**, construída de olhos abertos.
- **NÃO** usa IIT/Φ como mecanismo, nem promete consciência. GWT é copiada como **engenharia** (blackboard com gating), não como teoria da consciência.
- **NÃO** roda a LLM-córtex em hardware neuromórfico (neuromórfico = periferia event-driven, não o córtex).

---

## 2. Princípio unificador (a espinha)

Quatro frameworks independentes convergem para a **mesma família** matemática, e é isso que torna a arquitetura coerente em vez de colcha de retalhos. *(Calibração pós-cenário: o mapeamento entre elas — ex. XCAL / predictive coding ↔ energia livre de Friston — é **análogo, não identidade provada**. Tratar como "mesma família", não "exatamente a mesma equação". Ver §10.)*

| Framework | Onde | O que diz |
|---|---|---|
| Predictive coding (Rao & Ballard 1999) | córtex, retina, V1, A1 | predição desce, erro sobe |
| Free energy / active inference (Friston) | tronco, amígdala, ação | percepção e ação minimizam a mesma surpresa |
| Precisão = ganho = atenção = neuromodulação (Parr & Friston 2017) | todos | cada erro é ponderado por confiança; reescalar precisão *é* atenção *é* neuromodulação |
| Homeostatic RL (Keramati & Gutkin 2014) | tronco/hipotálamo | reward = redução de drive; valor depende do estado interno |

**Unidade de mensagem do barramento (a interface de tudo):**
```
{ conteúdo: embedding,
  erro: escalar|vetor (surpresa ponderada),
  precisão: escalar (confiança/ganho),
  origem: tag-de-módulo,
  t_origem: quando a observação foi colhida,
  t_alvo:   instante para o qual a predição é válida,
  t_validade: prazo após o qual a mensagem é descartada }
```
Nunca trafegam dados brutos — só **erro de predição comprimido e ponderado por precisão**.

---

## 3. Arquitetura — módulos e fluxo de dados

```
        NEUROMODULAÇÃO GLOBAL (4 escalares) — reescala precisão de TODOS
   DA=recompensa/vigor · NE=temperatura/exploração · ACh=ganho sensorial · 5-HT=horizonte
                                   │
                                   ▼
   ┌──────────── LLM-CÓRTEX (System 2 lento, ~0.5–9 Hz, CONGELADA) ───────────┐
   │  dlPFC (working memory, raciocínio) · emite objetivo-latente + atenção ↓  │
   │  surpresa nativa (−log p) = erro de alto nível ↑                          │
   └───────▲───────────────────────────────────────────┬──────────────────────┘
        ↑ erro+saliência (top-k)               objetivo-latente / predição / atenção ↓
   ┌───────┴──────────────── TÁLAMO (roteador/gate, não relé) ─────────────────┐
   │ top-k por precisão×relevância · tônico↔burst (acorda a LLM) ·              │
   │ PFC–MD infere contexto + anti-esquecimento · TRN competição lateral ·     │
   │ WATCHDOG: heartbeat por módulo, precisão→0 se cai                          │
   └──▲──────▲──────▲──────▲──────▲──────▲──────▲───────────────────────────────┘
   ┌──┴─┐ ┌──┴─┐ ┌──┴──┐ ┌─┴──┐ ┌─┴───┐ ┌┴────┐ ┌┴─────────────┐
   │VISÃO│ │ÁUDIO│ │HIPO-│ │AMÍG│ │GÂNGL│ │CERE-│ │SISTEMA DE VALOR│
   │retina│ │cóclea│ │CAMPO│ │DALA│ │BASE │ │BELO │ │ OFC + amígdala │
   │LGN/V1│ │ A1  │ │índice│ │valên│ │RL/Go│ │fwd  │ │ (moeda comum)  │
   │preditivo│ │SSA/MMN│ │+SR+ │ │cia+ │ │NoGo+│ │model│ └────────────────┘
   │      │ │     │ │replay│ │assoc│ │gating│ │+timing│  ┌──────────────┐
   └──┬───┘ └──┬──┘ └──┬──┘ └─┬──┘ └──┬──┘ └─┬──┘   │ ACC-EVC      │
      │        │       │      │       │      │ ◄────┤ meta-controle │
      │   (camada 1.5: affordances + competição → subobjetivo motor) │ "acordar LLM"│
      ▼                                                  └──────────────┘
   ┌──────── LOOP SENSÓRIO-MOTOR RÁPIDO — System 1 (50–200 Hz) ───────────────┐
   │  world model (Dreamer/JEPA) + policy condicionada por subobjetivo +       │
   │  cerebelo (alinhador temporal) + anel reflexo subcortical (sempre on)     │
   │  fecha percepção→ação SEM esperar a LLM — ponte: Real-Time Chunking       │
   └──────────────────────────────────────────────────────────────────────────┘
```

### 3.1 O que SOBE (feedforward = erro/saliência)
Erro de predição comprimido por módulo: visão (tokens de novidade por região foveada e canal M/P/K — em cena estável, ≈0), áudio (eventos tipo-MMN acima de limiar), hipocampo (memórias completadas + vizinhos temporais), amígdala (valência `{pos,neg}` + arousal + interrupt), gânglios da base (ação selecionada + Q + δ), cerebelo (predição da consequência + timing), LLM (entropia/logprobs = incerteza interna).

### 3.2 O que DESCE (feedback = predição/atenção)
Predição/contexto esperado (subtrator preditivo de cada módulo), atenção/ganho (modula o gate do tálamo), **objetivo-latente** para o System 1 (um único vetor contínuo, estilo Helix — **não** comandos motores), comandos de memória (gravar/consolidar/esquecer), reavaliação de valor (extinção na amígdala).

---

## 4. Decisões de design travadas (os 5 buracos resolvidos)

> Detalhe e citações completas em `RESOLUCOES_3a.md`. Aqui ficam as decisões.

### D1 — Córtex, motor e granularidade do PFC
- **UM córtex-LLM único e congelado.** Não há vários LLMs competindo — os "especialistas que competem" já são os módulos periféricos; o workspace é o tálamo+GWT. Especialização *dentro* do córtex via **MoE/adapters** roteados pelo gate interno, nunca instâncias separadas.
- **Hierarquia motora em 3 camadas** entre objetivo e atuador: (1.5) **PPC+pré-motor** gera K affordances candidatas, TRN faz a competição → subobjetivo; (1) **M1/cerebelo** (world model+policy) condicionado pelo subobjetivo, emite action chunks; **ponte RTC** congela as `d` ações em execução e faz inpainting do resto. (Todorov-Jordan 2002; Cisek 2007; Botvinick-Niv-Barto 2009.)
- **Granularidade do PFC:** dlPFC (working memory/raciocínio) **DENTRO** da LLM; **OFC-valor** e **ACC-conflito/EVC** **FORA**, como módulos dedicados (OFC funde com amígdala = "sistema de valor"; ACC = sinal escalar de "acordar a LLM").
- **Gating habitual↔goal-directed graduado** (não binário): System 1 roda sozinho enquanto erro/conflito/EVC baixos; a LLM é acordada por surpresa, conflito, troca de contexto ou ACh alta. **Esse gate É o modo tônico/burst do tálamo.**

### D2 — Neuromodulação ↔ LLM congelada
- **Adaptador Neuromodulatório (NMA) de duas vias.** Só vira *steering na ativação* o que não tem knob nativo: **NE→temperatura/top_p** (trivial), **5-HT→orçamento de tokens/profundidade** (trivial), **ACh→ganho sensorial** e **DA→vigor** (steering via CAA/ActAdd).
- **Um neuromodulador por camada distinta** (van der Weij 2024: combinar na mesma camada falha). Aplicação saturante calibrada `hˡ ← hˡ + gain(m)·‖hˡ‖·v_modˡ`.
- **Canal de subida = probes lineares** (reading vectors), não tokens de prompt: lê `[valência, arousal, confiança]` de volta ao barramento.
- **Mitigação obrigatória:** steering inverte em até ~50% dos inputs (Tan 2024; Braun 2025) — perigoso para sinal *sempre-ligado*. Monitorar o probe; **fallback: desligar o steering daquele eixo** ao detectar inversão.

### D3 — Regime de treino (a regra de ouro)
- **A LLM NUNCA recebe gradiente.** Toda adaptação vive em adapters e módulos plásticos.
- **Ponte entre espaços latentes = Global Latent Workspace treinado** (encoder→GW / decoder←GW por módulo, 4 perdas: tradução + demi-cycle + cycle-consistency + contrastiva), backbones congelados. **Não** contrastivo puro (perde generalização cross-modal — Maytié 2025, verbatim).
- **Plasticidade particionada:** LLM congelada; encoders/decoders GW offline; world model+policy / gânglios / cerebelo / hipocampo online.
- **Currículo subcortical→cortical** (Knudsen 2004): Fase 0 sensores sozinhos → Fase 1 ponte GW → Fase 2 System 1 → Fase 3 LLM plugada.
- **Motivação intrínseca com freio anti-dark-room:** Expected Free Energy = epistêmico (ICM/RND, reusa o erro que já sobe) + pragmático (DA) + empowerment como regularizador.
- **Consolidação CLS:** replay offline → treina adapters (e LoRA externo opcional), **nunca os pesos da LLM**.
- **Honestidade injetada:** o SOTA de VLA favorece **co-treino end-to-end** (Helix retropropaga S1→S2). A 3a congela a LLM por **restrição de projeto** (LLM frontier inacessível ao gradiente), não por consenso de campo. Apresentar assim fortalece a credibilidade.

### D4 — Sincronia e tolerância a falha
- **Não há relógio mestre.** Barramento timestamp-driven; **erro é computado contra a predição cujo `t_alvo` casa com o instante real**, nunca contra a mais recente (resolve credit assignment temporal).
- **Cerebelo = alinhador temporal:** único módulo autorizado a propagar o estado de `t_origem`→`t_alvo` (forward rollout curto). Os demais só carimbam timestamp.
- **Fault-tolerance em 3 anéis:** reflexo subcortical (sempre on, parada segura *bounded*) → S1 (estende último chunk válido se stale) → S2 (tálamo rebaixa precisão a ~0 se a LLM dá timeout/alucina). **Watchdog no tálamo** (heartbeat por módulo).
- **Calibração:** Helix/RTC = confiança **MÉDIA** (fontes corporativas/preprints próprios; réplica independente fraca do "+200 ms"; incerteza de transferência ao regime de latência da LLM).

### D5 — Avaliação e posicionamento
- **Bateria em 3 camadas:** (A) benchmarks consagrados — BabyAI → **Crafter** (primária) → Animal-AI → MineDojo/BEHAVIOR-1K/Habitat; (B) **qualidade do loop cérebro-fiel** (latência-sob-carga, recuperação de falha por ablação, aprendizado contínuo, eficiência amostral, troca de tarefa sem esquecer) — **rotulada como contribuição metodológica a validar, não suíte pronta**; (C) honestidade (eixo de custo + baseline ablado por órgão + reprodutibilidade — Kapoor-Narayanan 2024).
- **Posicionamento honesto:** a novidade é a *integração*; defensável hoje, com **prazo de validade curto** — datar a varredura (jun/2026); o gêmeo mais próximo (Theater of Mind, abr/2026) não tem um único experimento.

---

## 5. Ordem de construção (deriva direta das decisões)

Há um **spike de de-risco (passo 0)** que usa uma LLM congelada de forma **mínima**, só pra testar a aposta central; **se ela passar**, o build completo segue, e nele a LLM volta a ser o **último** órgão a entrar (passo 5). A "visão fiel" (retina/whitening/M-P-K) vira refinamento do passo 0 depois do veredito.

| # | Etapa | Entrega | LLM? |
|---|---|---|---|
| **0** | **Loop mínimo visão→ação que TESTA a aposta do congelamento — É O MVP** (ver §6) | veredito: adaptador recupera o gap do co-treino? | sim (congelada, mínima) |
| 1 | Barramento + gramática + timestamps + watchdog | espinha de mensagens `{...t_origem,t_alvo,t_validade}` | não |
| 2 | Ponte GW (Fase 1) | encoders/decoders 4 perdas, ≥2 sensores estáveis | não |
| 3 | System 1 (Fase 2) | world model+policy online; cerebelo + gânglios da base; **1º loop fechado** | não |
| 4 | Camada motora 1.5 + tálamo top-k + TRN | affordances/competição sobre o System 1 | não |
| 5 | System 2 plugado (Fase 3) | LLM congelada lê/escreve no GW via RTC; NMA + ACC-EVC | **sim** |
| 6 | Consolidação CLS + bateria A/B/C | replay→adapters; avaliação contínua | — |

**De-risco crítico em paralelo:** medir a latência real da LLM-alvo em bancada isolada **durante as Fases 0–2**, antes de depender dela no passo 5. Se for lenta demais, o papel da LLM se redefine para **planejador lento** (não co-controlador).

---

## 6. Escopo da PRIMEIRA entrega — Loop mínimo visão→ação que TESTA a aposta do congelamento

Esta é a única parte que vamos especificar e implementar **a seguir** (terá seu próprio plano). Mudou em relação ao plano ingênuo: em vez de "visão fiel pura", a primeira entrega **de-risca a aposta que sustenta o projeto inteiro** — e já é a visão→ação nascendo. Usa peças prontas; não constrói tudo do zero.

### 6.1 Objetivo
Responder, barato e cedo, à pergunta nº 1 (ver §7): **"LLM-córtex congelada + adaptador treinável (o tálamo-roteador)" recupera a performance de "co-treino end-to-end"** numa tarefa mínima de visão→ação? Ninguém testou isso diretamente. Se não recuperar, repensamos o congelamento **antes** de investir no resto.

### 6.2 O que constrói (mínimo viável, peças prontas)
- **Percepção JEPA-like** (predição latente, **não** gerativa): partir de **V-JEPA 2** (pesos abertos). A retina fiel (whitening / M-P-K / gargalo) entra como refinamento **depois** do veredito.
- **Loop rápido (System 1):** policy/world model pequeno num domínio simples — **openpi/π0-style** ou **DreamerV3**, em simulação (MuJoCo/Isaac) ou tarefa de mesa. Controle a 100+ Hz.
- **Córtex:** LLM congelada emitindo um **objetivo-latente**, conectada por **cross-attention a camada intermediária (~12)**, ponte assíncrona por **RTC** (LeRobot).
- **Adaptador treinável** (o "tálamo-roteador") absorvendo o que o co-treino faria.
- **Sinal de erro vindo do sensório/forward-model — NUNCA da LLM se autojulgando** (padrão LLM-Modulo).

### 6.3 O experimento decisivo (o coração da entrega)
Ablação direta no domínio mínimo: **(A)** congelado + adaptador treinável · **(B)** co-treino end-to-end · **(C)** baseline modelo único. Medir performance, eficiência amostral e o **gap que o adaptador recupera**. Garantir **assimetria física genuína** (controle 100+ Hz vs percepção ~9 Hz) — sem ela o resultado evapora (aviso Coda-Forno).

### 6.4 Critérios de aceite
- O experimento roda e dá um **veredito claro**: o adaptador recupera o gap? (sim / parcial / não).
- **Se sim/parcial-aceitável:** seguimos para a visão fiel + arquitetura completa.
- **Se não:** repensamos o congelamento antes de gastar no resto — **esse é o valor: falhar barato.**
- **Instrumentado desde o dia 1** (ablação TrueSkill; workspace-vs-sem; cycle-vs-sem).

### 6.5 Peças prontas a reusar
V-JEPA 2 (visão latente) · openpi/π0 (VLA) · DreamerV3 (world model) · RTC/LeRobot (ponte) · pymdp + RxInfer (seleção/active inference) · ngc-learn (módulos backprop-free) · Monty (córtex sensório-motor) · AXIOM (referência de gramática única como agente).

---

## 7. Riscos, limites e mapa de confiança

### 7.1 Riscos-mestre (de engenharia, não de ciência)
1. **Latência da LLM congelada quebrando a ponte RTC** — o maior risco real; só testável no passo 5, por isso **de-riscar cedo** (§5).
2. **Casar duas velocidades sem co-treino** — a 3a congela a LLM contra o SOTA; mitigado por adapter+GW treinados, mas é composição inédita em escala.
3. **Estabilizar 4 botões neuromodulatórios globais sem oscilar** — tratar como sistema de controle (saturação, separação de escalas de tempo), não mapeamento; steering pode inverter (fallback obrigatório).

### 7.2 Mapa de confiança (resumo — detalhe em `FUNDACAO_3a.md` §6)
- **ALTA:** predictive coding como princípio; retina/LGN; SSA/MMN auditivo; tálamo MD/TRN + PFC-MD com código; hipocampo (indexação, CA3=atenção, SR); DA=RPE; cerebelo filtro adaptativo; mapa Doya; prova Keramati-Gutkin; dual-process existe em robótica de fronteira.
- **MÉDIA:** "LLM = algoritmo do córtex" (convergência parcial); actor-critic dos BG (simplificado); GWT/IIT como consciência (Cogitate 2025 falsificou ambas — usar GWT como engenharia); Helix/RTC (corporativo/preprint); EM-LLM (sem réplica); "ninguém juntou o conjunto" (negativa datada).
- **BAIXA:** 5-HT (≥4 teorias — implementar configurável); "low road" subcortical em humanos; binding por sincronia gama; SNN+LLM neuromórfico para o córtex; co-acoplamento S2→S1 sem co-treino.

---

## 8. Critérios de sucesso do projeto
- **Fase 0:** o módulo de visão satisfaz §6.4.
- **MVP integrado (passos 0–5):** loop percepção→ação fecha estável a dezenas de Hz; a LLM entra assíncrona sem ser gargalo; drives de erro e estado interno variável observáveis.
- **Validação científica:** Camada B mostra que cada órgão contribui causalmente (ablação) e que o sistema tem propriedades que nenhum baseline ablado tem.

## 9. Questões em aberto (prioridade menor, resolver ao chegar)
- **Glia/astrócitos:** fora de escopo (declarado) — faixa temporal segundos-minutos não modelada.
- **Geradores de padrão central (CPGs) / reflexo medular:** absorvidos no "anel reflexo subcortical" (D4); detalhar quando houver embodiment.
- **Custo do empowerment** (capacidade de canal em alta dimensão) — avaliar antes de incluir no loop online.
- **Drift dos adapters do GW ao longo de meses** — monitorar, re-treino em lote.
- **Correções de metadados de venue** antes de qualquer publicação (lista em `RESOLUCOES_3a.md`).

---

## 10. Lições do cenário competitivo (incorporadas — 2026-06-03)

Da varredura `CENARIO_E_LICOES.md` (6 clusters, verificada adversarialmente). O que mudou no design:

### 10.1 Posicionamento (honestidade)
- **Ineditismo rebaixado.** "LLM congelada como córtex" não é espaço em branco — é *Frozen-LM Conditioning* (NeuroLoRA, PaceLLM, Brainstacks, 2025-2026), já com benchmark. O diferencial é **só** a síntese integrada completa + duas velocidades físicas + gramática única, **como aposta a provar**. Vender "primeiro a congelar LLM como córtex" é falso e destrói credibilidade.
- **Gramática calibrada:** "mesma família que predictive coding / energia livre", não "exatamente Friston" (mapeamento análogo, não identidade).

### 10.2 Decisões adotadas
- **Visão JEPA-like (predição latente), não gerativa** — V-JEPA 2 como ponto de partida. Contrastivo puro não basta; usar broadcast + cycle-consistency.
- **Erro nunca vem da LLM se autojulgando** (auto-correção intrínseca degrada — Huang, ICLR 2024). Vem do cerebelo/sensório. Padrão **LLM-Modulo**: córtex nunca emite sem crítico externo.
- **S1 conecta ao córtex por camada intermediária (~12), não a final** (GR00T N1).
- **Neuromodulação inclui sinal de progresso/curiosidade** (ganho de informação da EFE/AXIOM, com regulação), não só valor — senão prende em ótimo local (PBWM).
- **Vocabulário de cada módulo estável; só roteamento e ganho mudam** (Princípio 17 Leabra) — pro RTC ter semântica fixa a transportar.
- **Memória procedural com poda/unlearning + sinal de sucesso externo** (Error Fossilization do Voyager).
- **Instrumentar e benchmarkar desde o dia 1** (ablação TrueSkill; workspace-vs-sem; cycle-vs-sem).
- **Etapas só valem se cada uma for robusta sozinha** (Leabra: "chip pela metade não funciona").

### 10.3 Risco nº 1 (vira a primeira entrega — §6)
Ninguém em robótica congela o S2 — **todos co-treinam**, e há resultado negativo (Coda-Forno) de que comunicação por latente não basta sem assimetria física real. A aposta "congelado + adaptador recupera o gap" **nunca foi testada**. → é o objetivo do MVP.

### 10.4 Código aberto a reusar
V-JEPA 2 · openpi/π0 · DreamerV3 · RTC/LeRobot · pymdp · RxInfer · ngc-learn · Monty · AXIOM (referência) · emergent/Leabra (banco de testes de integração).

### 10.5 Observar de perto
VERSES/AXIOM · VanRullen/GLW (Toulouse) · Physical Intelligence + NVIDIA (RTC) · Hong Jeong (Miniature Brain Transformer, mar/2026) · linha *Frozen-LM Conditioning* (risco competitivo imediato).
```
