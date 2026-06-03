Vou produzir o documento de resoluções de design diretamente. Tenho todo o material verificado nos cinco buracos; não preciso de ferramentas para sintetizar. Entrego o markdown puro.

# RESOLUÇÕES DE DESIGN — Arquitetura cognitiva cérebro-fiel 3a

> Anexo formal ao documento de arquitetura. Cada buraco crítico recebe: **decisão final acionável**, **justificativa citada** (apenas afirmações que sobreviveram à verificação adversarial), **incerteza residual + mitigação** e **confiança** (alta/média/baixa). Ao fim: calibração atualizada, posicionamento competitivo honesto e impacto na ordem de construção.
>
> Convenção de citação: afirmações marcadas como **extrapolação de design** são saltos do arquiteto, *não* resultados dos papers — sinalizados explicitamente, conforme exigido pela verificação.

---

## Buraco 1 — Córtex motor, vão objetivo→ação, granularidade do PFC, e um-vs-vários LLMs

### Decisão final

**1.1 — UM córtex-LLM único e congelado. NÃO vários LLMs competindo.** Os "especialistas que competem" já são os módulos periféricos (visão, áudio, hipocampo, amígdala, gânglios da base, cerebelo); o *workspace* já é o tálamo-roteador + GWT. Onde se quiser especialização *dentro* do córtex, usa-se **MoE/adapters roteados pelo gate interno da própria LLM** — nunca instâncias separadas. Esta é a decisão-mãe que condiciona todo o resto.

**1.2 — Inserir três camadas entre o objetivo-latente e o atuador:**

- **Camada 1.5 "PPC + pré-motor" (~20–50 Hz, frequência a calibrar).** Módulo de *affordances + competição*: recebe (a) o objetivo-latente da LLM (descendo como predição de alta precisão) e (b) a percepção do world-model; gera **K opções de ação candidatas em paralelo** e deixa o **TRN/competição lateral já existente** fazer o *biased competition*. Saída: **uma "opção" vencedora = subobjetivo motor**. Os gânglios da base (Go/NoGo já no 3a) fazem o *gating* final.
- **Camada 1 "M1/cerebelo" (50–200 Hz).** É o world-model + policy já existente, agora **condicionado pelo subobjetivo (opção), não por trajetória**. Implementa intervenção mínima (Todorov-Jordan): recebe custo/objetivo e fecha percepção→ação corrigindo só erros relevantes à tarefa. Emite **action chunks**.
- **Ponte assíncrona — Real-Time Chunking (RTC) literal.** Congela as `d` ações já em execução, faz *soft-mask inpainting* do resto.

**1.3 — Interface objetivo-latente = UM único vetor latente contínuo** (modelo Helix), projetado no espaço de tokens da Camada 1.5/1 e concatenado às features de percepção. Desce pela gramática do barramento como `{conteúdo=embedding, erro, precisão, origem=córtex, timestamp}`. **Texto NÃO é a interface primária** — fica como CoT/log interno opcional e canal de depuração.

**1.4 — Granularidade do PFC (o que sai da LLM):**

| Função | Localização | Mecanismo |
|---|---|---|
| **OFC-valor** (moeda comum) | **FORA** da LLM | Módulo de valoração; funde-se com a **amígdala** já existente → "sistema de valor". Escalar/embedding de baixa dimensão, atualizado a alta frequência por DA. |
| **OFC-mapa-de-estados** (inferência de estado latente) | **FORA**, acoplado ao hipocampo | É a função do **PFC-MD/hipocampo-índice** já existente inferindo contexto. |
| **ACC-conflito/esforço (EVC)** | **FORA**, módulo escalar de meta-controle | Lê divergência entre opções (Camada 1.5) e entre System 1 e a predição da LLM; computa **EVC = payoff esperado − custo do esforço**. Saída: sinal de "acordar a LLM" + ajuste de NE. |
| **dlPFC** (working memory, raciocínio multi-passo) | **DENTRO** da LLM | É exatamente o que a LLM congelada faz bem. |

**1.5 — Regra de gating habitual↔goal-directed (arbitragem graduada por incerteza, NÃO binária):**

- **System 1 roda sozinho** enquanto (i) erro de predição do world-model baixo, (ii) conflito entre opções baixo, (iii) EVC de acordar a LLM < 0.
- **Acorda a LLM** quando qualquer disparo: erro alto (surpresa), conflito alto entre affordances, mudança de contexto detectada pelo PFC-MD, ou ACh elevada (novidade).
- **Implementação:** este gate É o **modo tônico/burst do tálamo** já no 3a — burst = ignição GWT/broadcast pro córtex; tônico = System 1 segue sozinho.

### Justificativa citada

- **Todorov & Jordan (2002), *Nat. Neurosci.* 5:1226–1235** [CONFIRMADO, vol/pág exatos]: controle ótimo de feedback, princípio de intervenção mínima — o nível baixo recebe *objetivo*, não trajetória ponto-a-ponto.
- **Todorov (2004), *Nat. Neurosci.* 7(9):907–915** [CONFIRMADO]: hierarquia funcional — nível baixo transforma a dinâmica para o nível alto controlar otimamente. Base canônica da decomposição em duas camadas com interface por objetivo.
- **Cisek (2007), *Phil. Trans. R. Soc. B* 362(1485):1585–1599** [CONFIRMADO]: especificação paralela de ações + biased competition no próprio sistema sensório-motor. Mapeia 1:1 no TRN/tálamo top-k do 3a.
- **PPC/AIP–F5 como gerador de affordances** [CONFIRMADO no conteúdo]: o conceito (neurônios canônicos disparam à visão de objeto agarrável) é real e bem-suportado. **Correção obrigatória de citação:** usar **Sakreida et al. (2016), *Neurosci. Biobehav. Rev.*** e/ou **Maranesi, Bonini & Fogassi (2014), *Front. Psychol.* 5:538** — a etiqueta "Sakreida et al. 2014, Front. Psychol." estava errada (metadados trocados, conceito correto).
- **Botvinick, Niv & Barto (2009), *Cognition* 113(3):262–280** [CONFIRMADO]: options framework; pré-SMA forma sequências/chunks de ação. Ancoradouro de "objetivo-latente = opção/subobjetivo".
- **Wilson et al. (2014), *Neuron* 81(2):267–279** [CONFIRMADO]: OFC = mapa cognitivo do task space / inferência de estado.
- **Bartra, McGuire & Kable (2013), *NeuroImage* 76:412–427** [CONFIRMADO, N=206]: valor em moeda comum no vmPFC/OFC, dissociado do dlPFC/working memory.
- **Botvinick et al. (2001), *Psychol. Review* 108:624–652** [CONFIRMADO]: ACC = monitoramento de conflito = sinal escalar de demanda por controle.
- **Shenhav, Botvinick & Cohen (2013), *Neuron* 79(2):217–240** [CONFIRMADO]: EVC = payoff esperado − custo do esforço. **Ressalva:** operacionalizar EVC como gate "acordar a LLM" é **extrapolação de design**, não afirmação do paper.
- **Daw, Niv & Dayan (2005), *Nat. Neurosci.* 8(12):1704–1711** [CONFIRMADO]: arbitragem MB/MF por incerteza bayesiana. A aplicação como gate System1↔LLM é **extrapolação de design**.
- **Dolan & Dayan (2013), *Neuron* 80:312–325** [CONFIRMADO]: hábito↔goal-directed é *continuum* arbitrado — justifica gating graduado, não binário.
- **Helix (Figure AI, 2025)** [CONFIRMADO na página oficial]: **um único vetor latente contínuo** S2→S1; S2 (VLM 7B, 7–9 Hz) assíncrono escrevendo em shared-memory latent; S1 (80M, 200 Hz) tempo real, 35-DoF. *Fonte corporativa, sem peer-review — ver Calibração.*
- **π0.5 (Physical Intelligence, 2025, arXiv:2504.16054)** [CONFIRMADO]: hierarquia subtarefa-em-texto (autoregressivo discreto) → action-expert (flow matching, chunk 50 passos = 1 s). Valida texto como caminho *secundário*. *Preprint próprio.*
- **RTC — Black, Galliker & Levine, NeurIPS 2025 (arXiv:2506.07339)** [CONFIRMADO, peer-reviewed]: freeze das `d` ações + soft-mask inpainting; robusto a +200 ms. *Único item peer-reviewed da tríade de engenharia.*
- **VanRullen & Kanai (2021), *Trends Neurosci.* 44(9):692–704** [CONFIRMADO]: módulos especialistas + **um** Global Latent Workspace — não N córtices grandes.
- **Cai et al. (2024), arXiv:2407.06204** [CONFIRMADO]: MoE = especialização fina roteada por gate dentro de **um** backbone.

### Incerteza residual e mitigação

- **Treino com córtex congelado (maior risco).** Helix retropropaga S1→S2 com S2 *treinável*; aqui o S2 (LLM) está **congelado**. A validação empírica de Helix NÃO cobre "encoder fixo + só camada de projeção treinada". → **Mitigar:** treinar apenas a camada de projeção do latente + a camada de affordance, com a LLM como *encoder fixo* (adapters); aceitar que a expressividade do latente congelado para controle fino é não-confirmada e medir em protótipo.
- **Onde mora a SMA-sequência.** Deixada na Camada 1.5; pode precisar de módulo SMA dedicado se sequências longas falharem. → calibrar empiricamente.
- **Frequência da Camada 1.5 (~20–50 Hz)** é estimativa por analogia, sem número canônico. → calibrar.
- **Calibração do EVC** (custo do esforço em unidades comparáveis ao payoff) é não-trivial; gate mal-calibrado = acorda demais (lento) ou de menos (burro). → varredura empírica do limiar.

### Confiança
**ALTA** na estrutura (hierarquia de 3 camadas objetivo→opção→chunk; latente único; RTC como ponte; OFC-valor + ACC-EVC fora da LLM; um córtex único — cada peça com ancoragem dupla neurociência + engenharia 2025). **MÉDIA** nos detalhes de implementação (dimensionalidade do latente sob córtex congelado, frequência da Camada 1.5, calibração do EVC, necessidade de módulo SMA próprio).

---

## Buraco 2 — Canal concreto neuromodulação ↔ LLM congelada (steering vectors)

### Decisão final

**Implementar um Adaptador Neuromodulatório (NMA) de duas vias**, sentado no barramento entre o tálamo-roteador e a LLM congelada. Traduz os 4 escalares globais para intervenções e traduz ativações da LLM de volta para escalares.

**2.1 — Particionar por "trivial" vs. "steering na ativação":**

| Neuromod | Variável-alvo | Mecanismo | Por quê |
|---|---|---|---|
| **NE** (temperatura/exploração) | aleatoriedade da política | **TRIVIAL**: NE → `temperature` + `top_p` | É o "inverse temperature" de Doya. Knob nativo, monotônico, risco zero. |
| **5-HT** (horizonte) | profundidade de raciocínio / desconto | **TRIVIAL + leve steering**: 5-HT → orçamento de tokens / profundidade de CoT; opcional vetor "longo prazo vs. agir já" via CAA com sinal | Horizonte é majoritariamente alocação de compute/tokens. |
| **ACh** (ganho sensorial) | precisão bottom-up vs. prior | **STEERING na ativação** (CAA/ActAdd) + reescala de precisão no barramento | Sem knob nativo. ACh (a) multiplica a precisão das mensagens sensoriais no barramento *antes* da LLM e (b) empurra control vector "pesar evidência atual". |
| **DA** (recompensa/vigor) | vigor/assertividade, viés Go | **STEERING leve** (control vector "vigor") + entra primariamente nos gânglios da base (Go/No-Go) | O grosso do efeito de DA é fora da LLM. |

**Regra de ouro: só vira steering na ativação aquilo que não tem knob nativo equivalente.**

**2.2 — Construção dos control vectors (offline, uma vez, congelado junto com a LLM):** para cada eixo que precisa de steering (ACh-ganho, DA-vigor, 5-HT-estilo), extrair por **CAA** (pares de prompts contrastantes, média da diferença no resíduo). **CRÍTICO — alocar cada neuromodulador em uma camada diferente** (ACh em camada média-inicial onde o sensorial entra; DA-vigor em camada média-tardia próxima da decisão), para evitar interferência. NÃO somar os 4 vetores na mesma camada.

**2.3 — Aplicação online:** `hˡ ← hˡ + gain(m) · ‖hˡ‖ · v_modˡ`, com `m ∈ [−1,1]` e `gain(·)` **saturante calibrada** (tanh/sigmoide reescalada) na faixa segura validada offline. Sinal de `m` define direção (CAA: coef. negativo reverte). NE e 5-HT-orçamento NÃO passam por aqui.

**2.4 — Canal de subida (LLM → estado neuromod): probes lineares, NÃO tokens de prompt.** Anexar reading vectors no mesmo resíduo, lendo `[valência, arousal, confiança/precisão]` de volta ao barramento como mensagem padrão (origem=PFC). Alimenta a amígdala e o controlador neuromod, fechando o loop sem retreino. **Tokens de prompt ("seu NE=0.8") só como fallback de debug humano-legível.**

### Justificativa citada

- **Doya (2002), *Neural Networks* 15(4–6):495–506** [CONFIRMADO]: DA = TD error, 5-HT = desconto/horizonte, NE = inverse temperature, ACh = learning rate. NE→temperatura e 5-HT→horizonte estão literalmente no paper.
- **Aston-Jones & Cohen (2005), *Annu. Rev. Neurosci.* 28:403–450** [CONFIRMADO]: NE/LC = ganho cortical global; fásico=exploitation, tônico=exploration (análogo ao tônico/burst do tálamo).
- **ActAdd (arXiv:2308.10248)** [CONFIRMADO; título canônico atual "Steering Language Models With Activation Engineering"]: injeção de escalar por diferença de ativações, só forward pass. *Detalhes c≈3–15 e camadas 16/20 = ilustrativos, NÃO transferíveis.*
- **CAA (arXiv:2312.06681, ACL 2024)** [CONFIRMADO]: média da diferença no resíduo; coeficiente positivo/negativo controla grau e direção; "minimally reduces capabilities". Confirma escalar único com sinal para neuromodulador bipolar.
- **RepE (Zou et al., arXiv:2310.01405)** [CONFIRMADO]: Reading Vectors (LAT) + Control Vectors — o gabarito do canal bidirecional.
- **Templeton et al. (2024), "Scaling Monosemanticity"** [CONFIRMADO]: feature steering por clamping; saturação graciosa (obsessivo mas fluente) em valores extremos. *Rebaixado a opção futura (exige treinar SAE).*
- **Braun et al. (2025, arXiv:2505.22637)** [CONFIRMADO]: steering tem alta variância, às vezes inverte, é prompt-dependente; só confiável com direção coerente.
- **Tan et al. (NeurIPS 2024)** [CONFIRMADO]: mesmo vetor varia com prompt; vários datasets dão comportamento **oposto em ~50% dos inputs**.
- **van der Weij et al. (2024, arXiv:2403.05767)** [CONFIRMADO]: combinar múltiplos comportamentos num único vetor é "largely unsuccessful"; **vetores distintos em camadas distintas é "promising"** — base direta da regra "um neuromodulador por camada".
- **Sun et al. (2026, arXiv:2604.03147)** [CONFIRMADO]: subespaço valência-arousal de baixa dimensão, **simultaneamente legível (probe) e controlável (steering)**, near-monotônico bidirecional, replica em Llama-3.1/Qwen3. **Ressalva:** validado sobre *affect/refusal/sycophancy*, não sobre os 4 escalares de regime — a transferência neuromod→VA é **extrapolação de design**.

### Incerteza residual e mitigação

- **Fragilidade do steering para sinal sempre-ligado (risco subdimensionado).** Tan/Braun mostram inversão em até ~50% dos inputs — catastrófico para um sinal de *regime* (≠ comportamento pontual). → **Mitigar:** (a) escolher só eixos com direção coerente (alta separação pos/neg); (b) monitorar o readout do probe para detectar inversões; (c) **política de fallback explícita: desligar o steering daquele eixo** quando o monitor detecta inversão (cair para knob nativo / sem-steering).
- **Composição de 4 sinais simultâneos = extrapolação além de qualquer experimento publicado** (papers testam 1–2). "Um modulador por camada" reduz mas não elimina interferência não-linear. → Conceptors (Postmus et al. 2024, arXiv:2410.16314) / BiPO (Cao et al. 2024, arXiv:2406.00045) como mitigação, ambos confirmados existentes, mas não validados para 4 sinais de regime. Testar no modelo-alvo.
- **Mapa escalar→coeficiente é empírico e específico do modelo.** → sweep de camada e coeficiente por LLM-alvo; teto rígido em `gain(m)`; nunca mapear `m=±1` para coeficiente máximo bruto.
- **Latência das duas velocidades:** steering recalcula por janela da LLM (0.5–9 Hz); System 1 (50–200 Hz) não vê o steering atualizado — aceitável (neuromod global muda devagar), mas a malha probe→neuromod→steering precisa de buffer assíncrono compatível com RTC.

### Confiança
**ALTA** na integridade das citações e na espinha do mecanismo (trivial onde há knob, steering onde não há, probes em vez de tokens, um modulador por camada — tudo deriva diretamente das fontes). **MÉDIA-ALTA** na decisão completa, rebaixada por: composição confiável de 4 sinais de regime é extrapolação; fragilidade do steering é mais ameaçadora para sinal sempre-ligado; a evidência-chave do canal bidirecional (Sun 2026) é sobre affect, não regime.

---

## Buraco 3 — Plano de co-adaptação / quem treina o quê, bootstrapping e motivação intrínseca

### Decisão final

**Regra de ouro: a LLM NUNCA recebe gradiente. Toda adaptação acontece em adapters e nos módulos plásticos. Nada é co-treinado end-to-end através da LLM.**

**3.1 — A ponte entre espaços latentes = um "Global Latent Workspace" treinado, NÃO co-treino, NÃO contrastivo puro.** Cada módulo (V1, A1, hipocampo, amígdala, BG, cerebelo, E/S da LLM) ganha um par **encoder→GW / decoder←GW** (análogo ao conector LLaVA, mas bidirecional). Treinar com as **quatro perdas do GW (tradução + demi-cycle + cycle-consistency + contrastiva InfoNCE)**, com **todos os backbones congelados** (incluindo a LLM). **NÃO usar alinhamento contrastivo puro** — perde a generalização cross-modal. A precisão de cada mensagem do barramento vira o **peso por-modalidade na atenção do GW**, conectando direto à neuromodulação ACh/NE.

**3.2 — Particionamento da plasticidade:**

| Componente | Regime | Sinal |
|---|---|---|
| LLM (PFC) | **Congelada para sempre** | — |
| Encoders/decoders do GW | **Treino offline, depois congelados** (re-treino raro, em lote) | 4 perdas GW |
| World model + policy (System 1) | **Online, contínuo** (estilo DayDreamer) | erro de predição + RL |
| Gânglios da base (Go/NoGo) | **Online** | DA (recompensa) |
| Cerebelo (forward model/timing) | **Online** | erro motor |
| Hipocampo (índice + SR) | **Escreve online**, lê sempre | episódios |
| Memória de longo prazo | **NÃO há fine-tune da LLM.** Consolidação = escrita em memória externa + replay para treinar adapters | replay |

**3.3 — Currículo subcortical-antes-de-cortical:**
- **Fase 0 — sensores sozinhos.** V1/A1 preditivos em auto-supervisão (predição do próximo frame), sem LLM no loop.
- **Fase 1 — ponte (GW).** Com sensores estáveis, treinar encoders/decoders (4 perdas). LLM ainda fora.
- **Fase 2 — System 1.** World model + policy online (DayDreamer) usando o GW como percepção; gânglios da base e cerebelo entram aqui.
- **Fase 3 — System 2 plugado.** Só agora a LLM congelada lê/escreve no GW (0.5–9 Hz), ponte assíncrona por RTC. Hipocampo já está escrevendo.

**3.4 — Motivação intrínseca com freio (resolve o dark room):** recompensa composta **ICM/RND (erro de predição) para exploração + termo de empowerment (controle) como regularizador anti-dark-room**, estruturada como **Expected Free Energy = valor epistêmico (curiosidade, decai com o conhecimento) + valor pragmático (recompensa via DA)**. O erro de predição reaproveita o sinal que já sobe no barramento (custo zero). Mapear na neuromodulação: **NE = peso do termo epistêmico; DA = peso do pragmático; 5-HT = horizonte/desconto sobre o EFE.**

**3.5 — Consolidação sem esquecimento (CLS explícito):** memória externa primeiro, fine-tune por último. Hipocampo (índice+SR+replay) = sistema rápido; LLM = neocórtex lento, **mas não a destrave**. O que virar conhecimento durável: **replay offline do hipocampo treina (a) os adapters do GW e (b) opcionalmente um adapter LoRA externo à LLM** — nunca os pesos da LLM. Interleaved learning evita interferência; EWC sobre o *adapter* (não sobre a LLM) se precisar proteção entre tarefas.

### Justificativa citada

- **LLaVA (Liu et al., NeurIPS 2023, arXiv:2304.08485)** [CONFIRMADO]: estágio 1 alinha visão→linguagem com **um MLP de projeção, ViT e LLM congelados** (~558k pares). Prova de que um conector fino basta sobre backbone congelado.
- **Devillers, Maytié, VanRullen (arXiv:2306.15711)** [CONFIRMADO; publicado em **IEEE TNNLS 2024** — preferir essa citação]: GW com backbones congelados, perdas auto-supervisionadas (cycle/demi-cycle) reduzem dado pareado em **4–7×**.
- **Maytié, Devillers, Arnold, VanRullen (2025), RL Journal vol. 3, pp. 1410–1426 (arXiv:2403.04588)** [CONFIRMADO na fonte primária]: policy treinada sobre GW congelado **transfere zero-shot** entre modalidades; **a variante CLIP-like (contrastiva pura) NÃO obteve a mesma generalização** (verbatim). Achado mais acionável.
- **Bertin-Johannet, Scipio, Maytié, VanRullen (arXiv:2602.08597)** [CONFIRMADO; título real "An Attention Mechanism for Robust Multimodal Integration in a Global Workspace Architecture"; rótulo "ICANN" não-verificado]: atenção pondera modalidades por robustez/precisão.
- **DayDreamer (Wu et al., CoRL 2023, arXiv:2206.14176)** [CONFIRMADO]: world model + policy online no robô físico — quadrúpede anda do zero em ~1h, adapta a perturbações em ~10 min. Prova do System 1 plástico online.
- **ICM (Pathak et al., ICML 2017)** [CONFIRMADO]: recompensa = erro do forward model em espaço de inverse dynamics; ignora o incontrolável.
- **RND (Burda et al., ICLR 2019)** [CONFIRMADO]: bônus = erro contra rede alvo fixa aleatória; SOTA em Montezuma. Barato, plugável.
- **Empowerment (Klyubin et al. 2005; Salge, Glackin, Polani 2014, arXiv:1310.1863)** [CONFIRMADO]: capacidade de canal ação→estado futuro; quarto escuro tem empowerment baixo → anti-dark-room.
- **Friston, Thornton, Clark (2012), *Front. Psychol.* 3:130** [CONFIRMADO]: surpresa é relativa ao modelo generativo; quarto escuro é surpreendente para um modelo que espera estrutura.
- **Friston et al. (2015), *Cogn. Neurosci.* 6(4):187–214** [CONFIRMADO]: EFE = valor pragmático + valor epistêmico. **Ressalva:** "convergência da exploração quando o ganho de informação esgota" é síntese interpretativa do framework, não citação literal.
- **CLS (McClelland et al. 1995, *Psychol. Rev.* 102:419–457; Kumaran et al. 2016, *Trends Cogn. Sci.* 20:512–534)** [CONFIRMADO]: hipocampo rápido/pattern-separated, neocórtex lento via interleaved learning + replay.
- **EWC (Kirkpatrick et al. 2017, PNAS)** [CONFIRMADO]: penaliza mudança em pesos importantes via Fisher.
- **Knudsen (2004), *J. Cogn. Neurosci.* 16(8):1412–1425** [CONFIRMADO]: períodos sensoriais críticos mais curtos/precoces que executivos — justifica o currículo subcortical→cortical.
- **Multimodal Dreaming (Maytié, Bertin-Johannet, VanRullen, arXiv:2502.21142)** [CONFIRMADO]: GW + world model RL — tie-in legítimo.

**Correção de honestidade obrigatória (erro factual material identificado na verificação):** a literatura VLA de fronteira **NÃO está convergindo para "S2 congelado / sem gradiente cruzado"**. Helix [CONFIRMADO] co-treina S1↔S2 com **gradiente retropropagado de S1 para S2 via o vetor latente** — o oposto. **DIAL (arXiv:2603.29844)** congela só o ViT e a camada de text-embedding; os blocos do LLM são **totalmente atualizados** e o treino converge para end-to-end. Citar DIAL como evidência de "duas velocidades sem gradiente cruzado" é leitura **invertida** do paper. **Apresentar honestamente: o SOTA favorece co-treino end-to-end; a 3a congela a LLM por restrição de projeto (LLM de fronteira inacessível ao gradiente), não por consenso de campo.** Esta correção fortalece a honestidade da decisão sem derrubá-la.

### Incerteza residual e mitigação

- **Escala não comprovada.** GW de Devillers/Maytié demonstrado em ambientes pequenos (Simple Shapes/Factory), não com LLM de fronteira nem 6+ módulos heterogêneos. → validar em protótipo incremental, módulo por módulo.
- **Latência tri-frequência** (sensor / S1 50–200 Hz / S2 0.5–9 Hz) atravessando o GW: sem benchmark público. → ver Buraco 4 (RTC + timestamp-driven).
- **Replay para consolidação em sistema aberto** funciona em tarefas delimitadas; drift dos adapters do GW ao longo de meses é incerto. → monitorar drift, re-treino em lote.
- **Empowerment é caro de estimar** (capacidade de canal em alta dimensão). → aproximação variacional; pode não rodar barato no loop online — avaliar custo antes de incluir.

### Confiança
**MÉDIA-ALTA.** **Alta** nos pilares (frozen-backbone+adapter via LLaVA; GW 4 perdas e transferência cross-modal verificados na fonte; ablação CLIP-like confirmada verbatim; ICM/RND/empowerment/EFE/CLS/EWC/DayDreamer todos reais). **Média** na composição inédita: ninguém demonstrou esse stack específico (LLM de fronteira congelada + GW de tradução + System 1 online + EFE-curiosidade + CLS-replay) junto e em escala.

---

## Buraco 4 — Recalibrar Helix/RTC + sincronia multi-relógio + tolerância a falha

### Decisão final

**D1 — Dual-process assíncrono acoplado por RTC, não por handoff síncrono.** LLM congelada = S2 (0.5–9 Hz); World-Model+Policy = S1 (50–200 Hz). Ponte = RTC: S1 gera o próximo chunk *enquanto executa o atual*; primeiras `d` ações (`d = ceil(latência_S2 / Δt_S1)`) **congeladas**, resto **inpainted** com soft-mask exponencial. Parametrizar (regime PI): H≈50 passos, `d ≤ s ≤ H−d`. **Substitui qualquer ideia de S1 esperar S2.**

**D2 — Não há relógio mestre. Barramento timestamp-driven, cada módulo com seu passo.** Toda mensagem ganha **dois timestamps**: `t_origem` (quando a observação foi colhida) e `t_alvo` (instante para o qual a predição é válida). **Regra de ouro do credit assignment temporal: erro é sempre computado contra a predição cujo `t_alvo` casa com o instante real de execução/observação, nunca contra a predição mais recente em wall-clock.** Predições com `t_alvo` vencido são descartadas. Resolve "erro referido a predição obsoleta".

**D3 — Cerebelo (forward model) = alinhador temporal do barramento.** Promover o cerebelo a camada de compensação de atraso: propaga o estado de `t_origem` para `t_alvo` (forward rollout curto) para que (a) S1 condicione a política no **estado previsto para o instante de execução** e (b) o erro dos módulos lentos seja re-referido ao estado contemporâneo antes de subir. **O cerebelo é o único módulo autorizado a "viajar no tempo" o estado; os demais só carimbam timestamp.**

**D4 — Neuromodulação modula a janela de confiança temporal.** NE alto encurta `t_validade` (mundo volátil → predições antigas valem menos); ACh alto aumenta o peso do erro sensorial fresco vs. predição. Liga a sincronia ao resto do 3a sem novo mecanismo. *(Extrapolação de design — sem citação-âncora.)*

**D5 — Fault-tolerance em três anéis, fallback subcortical sempre vivo:**
- **Anel reflexo (subcortical, sempre on, ~1 kHz):** controlador de segurança independente de S1/S2 — equilíbrio, limite de torque, **retração/parada segura em tempo fixo**. Nunca depende de chunk válido.
- **Anel S1 (50–200 Hz):** se o chunk está stale (`t_alvo` vencido) ou a predição diverge do feedback além do limiar, S1 estende o último chunk válido com decaimento e sinaliza degradação; se persiste, devolve controle ao anel reflexo.
- **Anel S2 (0.5–9 Hz):** se S2 dá timeout ou alucina objetivo (detectável por "predição confiante e errada" = erro alto sob precisão alta, ou objetivo inconsistente com a valência da amígdala), o **tálamo+PFC-MD rebaixa a precisão de S2 a ~0** e o sistema opera em modo "piloto automático S1" sob a última intenção válida. Degradação graciosa e *bounded*.
- **Watchdog no tálamo:** como já é roteador top-k por precisão, é o lugar natural do timeout + heartbeat por módulo; módulo sem heartbeat tem precisão zerada (o gating existente o silencia; competição TRN segue).

### Justificativa citada

- **RTC — Black, Galliker & Levine, NeurIPS 2025 (arXiv:2506.07339)** [CONFIRMADO, peer-reviewed; todos os números batem contra a fonte primária]: 50 Hz, Δt=20 ms, H=50, latência base 76 ms (97 ms com RTC); **robusto a +100/+200 ms, suporta >300 ms (>30% do horizonte)**; baseline síncrono degrada linearmente; **temporal ensembling para de funcionar acima de +100 ms** (dispara protective stop); freeze das `d` + peso exponencialmente decrescente; restrição `d ≤ s ≤ H−d`. Confirmado também em Knowledge Insulation/π0.5 (arXiv:2505.23705): "+200 ms estável, métodos ingênuos despencam".
- **ACT (Zhao et al., RSS 2023)** [CONFIRMADO]: action chunking + temporal ensembling exponencial; quebra acima de +100 ms. Mantido só como suavizador local dentro de S1.
- **Dynamic Predictive Coding (Jiang & Rao, *PLOS Comp. Biol.* 2024)** [CONFIRMADO, afirmação textual no abstract]: níveis altos do córtex têm janelas temporais mais longas e modulam a dinâmica dos baixos, corrigindo via erro de predição. **Valida diretamente "não há relógio mestre; acoplamento por predição↓/erro↑".**
- **Cerebelo como preditor de timing** [CONFIRMADO]: O'Reilly et al. (*J. Neurosci.* 2008); Narain et al. (*Nat. Commun.* 2018, priors de intervalos); Tanaka et al. (*Front. Syst. Neurosci.* 2020, forward model). Fundamenta usar o cerebelo como compensador de atraso.
- **Helix (Figure 2025)** [CONFIRMADO na página oficial]: S2 7–9 Hz, S1 200 Hz, execução assíncrona. *Corporativo — ver Calibração.*
- **GR00T N1 (NVIDIA, arXiv:2503.14734)** [CONFIRMADO]: S2 Eagle-2 ~10 Hz; **S1 Diffusion Transformer a 120 Hz** (o relatório dizia ">50 Hz", verdadeiro mas subestima 2×). *Preprint próprio com release aberto.*
- **π0 (arXiv:2410.24164) / Gemini Robotics 1.5 (arXiv:2510.03342) / OpenVLA (arXiv:2406.09246)** [CONFIRMADOS]: π0/Gemini com separação fast/slow (Gemini = interleaved, não dois relógios fixos); OpenVLA = contraponto monolítico. *Preprints próprios.*
- **Spec-VLA (arXiv:2507.22424; venue "EMNLP 2025" NÃO-VERIFICÁVEL) / PD-VLA (arXiv:2503.02310, 2.52×)** [CONFIRMADOS existência]: speculative/parallel decoding reduz latência *do modelo* — **complementar** ao RTC, não substituto.
- **Follow-ups de latência citáveis (dez/2025, passado):** Leave No Observation Behind (arXiv:2509.23224), Training-Time Action Conditioning for RTC (arXiv:2512.05964), Delay-Aware Diffusion Policy (arXiv:2512.07697) [CONFIRMADOS] — reforçam D2/D3.

### Incerteza residual e mitigação

- **Transferência do regime PI (76–97 ms) para LLM-S2 congelada (potencialmente 200–1000+ ms) é o MAIOR RISCO REAL.** Se a LLM for muito mais lenta, `d` cresce, o "inpaintable" encolhe e a garantia "+200 ms" pode sair da zona validada. → **Mitigar:** medir a latência real da LLM-S2 no hardware-alvo e recomputar `d`/`s`; reproduzir o protocolo de atraso injetado (+100/+200/+300 ms) no S1 próprio antes de assumir robustez — tratar o número de PI como hipótese, não fato herdado. Se a LLM for lenta demais, **seu papel se redefine para planejador lento, não co-controlador.**
- **Replicação independente do "+200 ms" ainda é fraca** (número-âncora vem de PI/Levine; NeurIPS dá peso, mas sem réplica de grupo neutro). → reproduzir em hardware próprio.
- **Detector de "objetivo alucinado" é heurística não-validada** ("predição confiante e errada" + inconsistência com valência). → tratar como hipótese; testar em loop fechado matando S2 / injetando objetivo inconsistente.
- **Forward model do cerebelo acumula erro em rollout longo** (se S2 demora muito). → há teto de quanto atraso o forward model esconde; calibrar.
- **D4 (neuromodulação de `t_validade`) e o watchdog tálamo-orquestrado são design novo**, não validados como sistema. → testar fallback subcortical em falha real, confirmando parada segura *bounded* sem depender de chunk válido.

### Confiança
- Dual-process assíncrono **é a direção dos líderes 2025**: **ALTA**.
- Dual-process assíncrono **é comprovadamente superior**: **MÉDIA** (evidência majoritariamente corporativa/preprint próprio).
- **RTC esconde +100/+200 ms: MÉDIA** (rebaixada de média-alta — ver Calibração: réplica independente fraca + incerteza de transferência ao regime de latência da LLM).
- Sincronia timestamp-driven + cerebelo alinhador: **MÉDIA-ALTA** na fundamentação (predictive coding hierárquico e forward model cerebelar sólidos), não alta porque a composição específica não foi testada como sistema.
- Fault-tolerance 3 anéis: **MÉDIA** (graceful degradation switch-based é padrão; detector de objetivo alucinado e watchdog tálamo são novos).

---

## Buraco 5 — Bateria de avaliação + varredura honesta de concorrentes

### Decisão final

**Adotar uma bateria em 3 camadas + manifesto de posicionamento honesto, ambos como anexo formal.**

**Camada A — Cognição/competência** (benchmarks públicos consagrados, ordem fixa de dificuldade):
1. **BabyAI** → eficiência amostral de linguagem grounded (gate de entrada).
2. **Crafter** → exploração + long-horizon, barato, cérebro-agnóstico, baselines fortes. **Métrica primária** (score agregado + vetor de 22 conquistas).
3. **Animal-AI** → cognição cérebro-fiel sem linguagem (permanência de objeto, uso de ferramenta) — diferencia de agentes puramente linguísticos.
4. **MineDojo → BEHAVIOR-1K** (rampa de embodiment); **Habitat 3.0** (loop sob carga social).

**Camada B — Qualidade do loop cérebro-fiel (a parte original; rotular explicitamente como *contribuição metodológica a validar*, NÃO suíte pronta):** cinco métricas instrumentadas no barramento `{conteúdo, erro, precisão, origem, timestamp}`:
- **Latência-sob-carga (L50/L95 percepção→ação):** System 1 mantém controle no orçamento 50–200 Hz com a LLM saturada? Medir degradação vs. LLM ociosa (habilitado pelo RTC).
- **Recuperação de falha (perturbação ablativa):** derrubar módulo (mascarar visão, congelar cerebelo, zerar um neuromodulador) e medir tempo/sucesso de recuperação. **Teste de causalidade dos 4 escalares.**
- **Aprendizado contínuo:** ACC/BWT/FWT sobre sequência Crafter/Continual World, com a LLM **congelada** — a plasticidade deve vir da periferia (hipocampo SR+replay, gânglios da base, cerebelo).
- **Eficiência amostral:** passos para competência vs. DreamerV3 (world-model puro) e vs. Voyager (LLM pura). O 3a deve ficar entre/acima (sinergia).
- **Troca de tarefa sem esquecer:** acurácia na tarefa antiga após N novas, com PFC-MD inferindo contexto — sonda do anti-esquecimento talâmico.

**Camada C — Honestidade de avaliação (Kapoor/Narayanan 2024):** todo número com **eixo de custo** ($/tarefa + FLOPs/Hz), **baseline de complexidade** (versão ablada sem tálamo / sem neuromodulação / sem System 1, para isolar a contribuição de cada órgão), reprodutibilidade via Agentic Benchmark Checklist, e declaração explícita do que cada benchmark mede.

### Justificativa citada

**Concorrentes / prior art** (todos [CONFIRMADOS]):
- **CoALA (Sumers et al., TMLR 2024, arXiv:2309.02427):** framework taxonômico, sem tálamo/neuromodulação/world model/loop sub-segundo.
- **Voyager (Wang et al. 2023, arXiv:2305.16291):** lifelong em Minecraft sobre GPT-4 caixa-preta; aprende por acúmulo de código, sem fine-tune, sem world model preditivo, sem loop rápido.
- **Generative Agents (Park et al., UIST 2023, arXiv:2304.03442):** memory stream + reflexão; foco em crença social.
- **LIDA (Franklin):** modelo computacional da GWT, pré-LLM, simbólico/subsimbólico.
- **Soar/ACT-R+LLM (Cognitive LLMs arXiv:2408.09176; Bootstrapping Cognitive Agents arXiv:2403.00810):** frameworks conceituais, validação experimental largamente ausente.
- **Theater of Mind / GWA (Shang 2026, arXiv:2604.08206)** [CONFIRMADO por leitura do HTML completo — pilar do argumento de originalidade]: enxame de agentes-LLM + Global Workspace + STM/LTM + drive por entropia. **Nenhum experimento/benchmark; sem tálamo/hipocampo/amígdala/gânglios/cerebelo/neuromoduladores; sem predictive coding; sem System 1 rápido.** É o "gêmeo mais próximo" e preenche ~1,5 coluna.
- **Spaun (Eliasmith et al., Science 2012):** 2,5M neurônios spiking, 8 tarefas — cérebro-fiel até spikes, mas **sem LLM**.
- **DreamerV3 (Hafner 2023, arXiv:2301.04104) / Genie (Bruce et al. 2024, arXiv:2402.15391):** SOTA de world model, **sem LLM frontier como S2 acoplada**.

**Fundações neurocientíficas** (todas [CONFIRMADAS]):
- **Doya (2002):** base canônica dos 4 neuromoduladores.
- **Rao & Ballard (1999), *Nat. Neurosci.* 2(1):79–87:** predictive coding — feedback=predição, feedforward=erro. Base da gramática do barramento.
- **Halassa/Schmitt et al. (Nature 2017, nature22073):** MD-thalamus amplifica conectividade do PFC, sustenta representações de regra; TRN faz seleção top-down. (O "Halassa 2015" provavelmente Wimmer et al. 2015 — consistente, não isolado.)
- **Stachenfeld et al. (2017):** hipocampo como mapa preditivo (successor representation). **Correção de venue:** é ***Nature Neuroscience* 20:1643–1653**, NÃO *Nature Human Behaviour* (atribuição cruzada com Momennejad et al. 2017, esse sim Nat. Hum. Behav.).
- **Goyal et al. (2021, arXiv:2103.01197):** shared global workspace, módulos competem por acesso. **Correção de venue:** "Consciousness Prior" (Bengio, arXiv:1709.08568) **nunca foi paper NeurIPS** — só o keynote System 1→2 foi NeurIPS 2019.
- **RTC (Black et al. 2025, arXiv:2506.07339):** ponte assíncrona — ver Buraco 4.
- **Butlin, Long, Bengio et al. (2023, arXiv:2308.08708):** indicator properties — usar **apenas como checklist arquitetural, NUNCA como alegação de consciência**.

**Benchmarks e métricas** (todos [CONFIRMADOS]):
- **Crafter (arXiv:2109.06780):** 22 conquistas; **humano 50,5%, DreamerV3 14,5, Curious Replay 19,4, EMERALD 58,1% (primeiro a superar humano em 10M passos — verificado nesta rodada)**.
- **BabyAI (ICLR 2019), MineDojo (3142 tarefas, arXiv:2206.08853), BEHAVIOR-1K (1000 atividades, arXiv:2403.09227), Animal-AI (900 tarefas, NeurIPS 2019), AgentBench (arXiv:2308.03688)**.
- **Habitat 3.0 (ICLR 2024, arXiv:2310.13724):** Social Navigation + Rearrangement. *">1000 FPS" NÃO-VERIFICADO — confirmar antes de slides.*
- **GEM (Lopez-Paz & Ranzato 2017):** ACC/BWT/FWT. **Continual World (Wołczyk et al., NeurIPS 2021, arXiv:2105.10919):** priorizar forward transfer.
- **Kapoor, Narayanan et al. (2024, arXiv:2407.01502), "AI Agents That Matter":** exigir eixo de custo + reprodutibilidade.

### Incerteza residual e mitigação

- **Latência da LLM frontier congelada** pode tornar o acoplamento S1↔S2 mais frágil que o RTC robô-VLA original (API vs. policy local). L95-sob-carga pode expor que a LLM raramente fecha o loop a tempo → redefine seu papel para planejador lento. → ver Buraco 4.
- **Atribuição causal dos 4 neuromoduladores** é difícil de provar limpa (ablações com efeitos confundidos); risco de neuromodulação virar "tempero" não-falsificável. → ablações fatoriais + Camada C (baseline ablado).
- **Aprendizado contínuo com LLM congelada é aposta:** se a capacidade vive na LLM, a plasticidade periférica pode dar BWT/FWT fracos ("esperto mas não aprende"). → medir cedo, em Crafter/Continual World.
- **Métricas de Camada B ainda não existem como suíte padronizada** — construção original. → **rotular explicitamente como contribuição metodológica a validar** (não apresentar como ferramenta pronta — seria a mesma desonestidade que o doc condena nos concorrentes).
- **Janela móvel de prior art é o calcanhar real:** Theater of Mind é de abr/2026, 2 meses atrás. → **datar explicitamente a varredura (jun/2026)** e prever que um "brain-inspired LLM agent" mais completo pode aparecer a qualquer trimestre.
- **Corrigir metadados de venue antes de publicar** (Stachenfeld; Consciousness Prior) — munição para revisores.

### Confiança
**ALTA** no eixo de concorrentes/prior art e nas fundações neurocientíficas (100% das citações load-bearing verificadas reais; o paper-pilar — Theater of Mind — lido na íntegra; EMERALD 58,1% confirmado). **MÉDIA-ALTA** na bateria em si: Camada A usa benchmarks consagrados; Camada B deriva de princípios sólidos mas é construção original a implementar/validar.

---

## CALIBRAÇÃO ATUALIZADA

**Helix/RTC rebaixados explicitamente para confiança MÉDIA**, conforme pedido, com a base do rebaixamento:

- **Helix (Figure AI):** **fonte corporativa, blog, sem peer-review.** Todos os detalhes batem na página oficial (latente único contínuo, 35-DoF, async shared-memory, end-to-end S1→S2), mas a afirmação de *superioridade quantitativa* não tem avaliação neutra. → **MÉDIA** como "isto é o melhor caminho"; **ALTA** apenas como "é o que os líderes estão fazendo".
- **π0/π0.5, GR00T N1, Gemini Robotics 1.5:** **preprints próprios** (com código aberto em alguns casos). Existência e arquitetura confirmadas; superioridade não validada por terceiros → **MÉDIA**.
- **RTC (arXiv:2506.07339):** é o **único peer-reviewed (NeurIPS 2025)** da tríade, e seus números batem *exatos* contra a fonte primária. Ainda assim **rebaixado para MÉDIA** por dois motivos verificados: (1) **réplica independente do "+200 ms sem degradação" é fraca** — o número-âncora vem essencialmente do mesmo grupo (PI/Levine); (2) **incerteza de transferência** do regime PI (latência 76–97 ms) para a LLM-S2 congelada (potencialmente 200–1000+ ms), que pode empurrar `d` para fora da zona validada. A robustez de RTC é real *no regime testado*; a sua aplicabilidade ao regime 3a é hipótese a medir, não fato herdado.

Tudo que se apoia em Helix/RTC (Buraco 1 ponte, Buraco 3 Fase 3, Buraco 4 D1, Buraco 5 Camada B latência) herda este teto MÉDIO até medição no hardware-alvo.

**Honestidade adicional injetada (correção factual da verificação):** a tendência da literatura VLA de fronteira é **a favor de co-treino end-to-end com gradiente cruzado** (Helix retropropaga S1→S2; DIAL termina end-to-end com gradiente no LLM). A escolha da 3a (LLM congelada para sempre) é **minoritária / contracorrente**, justificada por **restrição de projeto** (LLM de fronteira inacessível ao gradiente), não por consenso. Apresentar assim — "vamos contra o SOTA porque temos uma restrição que o SOTA não tem" — fortalece a credibilidade, não a enfraquece.

---

## POSICIONAMENTO COMPETITIVO

A novidade da 3a **não é nenhum componente isolado** — cada peça existe e é citável (Doya, Rao-Ballard, Halassa, Stachenfeld, Cisek, Todorov-Jordan, DreamerV3, RTC, GW de VanRullen). **A originalidade é a integração específica** — LLM frontier *congelada* como System 2 + System 1 com world model em loop sub-segundo + tálamo-roteador por precisão + 4 neuromoduladores globais + gramática única predição/erro/precisão — e essa frase é **defensável hoje (jun/2026)**: nenhum concorrente público combina ≥4 dos 6 órgãos cérebro-fiéis em torno de uma LLM frontier, e o "gêmeo" mais próximo (Theater of Mind/GWA, abr/2026) preenche ~1,5 coluna e **não tem um único experimento** — *mas a defensabilidade tem prazo de validade curto e exige datar a varredura, porque um agente brain-inspired mais completo pode surgir a qualquer trimestre.*

---

## IMPACTO NA ORDEM DE CONSTRUÇÃO

As decisões **confirmam e detalham** a vontade do usuário de começar pela VISÃO na forma fiel — e impõem uma ordem específica pela regra "a LLM nunca recebe gradiente" + currículo subcortical→cortical (Knudsen; Fases 0→3 do Buraco 3). O MVP **não** começa pela LLM nem pelo loop motor completo; começa pela percepção e pela ponte.

**Ordem de construção recomendada (deriva direta das 5 resoluções):**

1. **VISÃO fiel sozinha (Fase 0) — É O MVP.** V1 preditivo (retina/LGN/V1, predição do próximo frame, auto-supervisão), **sem LLM, sem GW, sem motor**. Critério de aceite isolado: erro de predição cai e estabiliza. Casa com a preferência do usuário e com o currículo subcortical-primeiro. *(Áudio/A1 entra em paralelo ou logo depois, mesmo molde.)*

2. **Barramento + gramática única + timestamp-driven (Buraco 4 D2).** Construir o protocolo de mensagem `{conteúdo, erro, precisão, origem, t_origem, t_alvo, t_validade}` **antes** de plugar o segundo módulo — é a espinha que tudo usa. Watchdog/heartbeat no tálamo desde o início.

3. **Ponte GW (Fase 1)** — encoders/decoders com as 4 perdas, backbones congelados. Só depois de ≥2 módulos sensoriais estáveis. **NÃO contrastivo puro.**

4. **System 1 (Fase 2)** — world model + policy online (DayDreamer), condicionado por subobjetivo; cerebelo (forward model = alinhador temporal, Buraco 4 D3) e gânglios da base (Go/NoGo) entram aqui. Aqui aparece o primeiro **loop fechado percepção→ação** sem a LLM.

5. **Camadas motoras 1.5 (affordance/competição) + tálamo-roteador top-k + TRN** (Buraco 1) — sobre o System 1 já vivo.

6. **System 2 plugado (Fase 3)** — só agora a LLM congelada lê/escreve no GW via **RTC**; NMA de neuromodulação (Buraco 2) e ACC-EVC/gate de "acordar a LLM" (Buraco 1.5) entram juntos. **Medir aqui a latência real da LLM-S2 e recomputar `d`/`s` do RTC** — ponto de decisão que pode redefinir o papel da LLM para planejador lento.

7. **Consolidação CLS (replay→adapters)** e bateria de avaliação Camadas A/B/C — contínuos, a partir do momento em que há loop fechado.

**Mudança líquida vs. plano ingênuo:** o instinto de "plugar a LLM cedo" é explicitamente **adiado para a Fase 3** — a LLM é o *último* órgão a entrar no loop fechado, não o primeiro. A visão fiel não é só o ponto de partida preferido do usuário; é o ponto de partida **forçado** pela arquitetura. O risco-mestre (latência da LLM congelada quebrando a ponte RTC) só é testável no passo 6, então **derriscar cedo** medindo a latência da LLM-alvo em bancada isolada, em paralelo às Fases 0–2, antes de depender dela.