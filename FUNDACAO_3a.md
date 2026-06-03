# Fundação de Pesquisa — Arquitetura Cognitiva Cérebro-Fiel (Recorte 3a)

**Data:** 2026-06-02
**Escopo:** LLM moderna como córtex de associação/pré-frontal, cercada por módulos cerebrais especializados em loop preditivo fechado.
**Disciplina epistêmica:** Cada afirmação abaixo sobreviveu a verificação adversarial contra fonte primária. Onde a fonte diz "convergência parcial", eu digo "convergência parcial" — não "identidade". Onde o mapeamento é contestado, está marcado.

---

## 1. Resumo por sistema (apenas afirmações confirmadas)

### 1.1 Córtex preditivo & LLM-como-córtex
O neocórtex roda **predictive coding hierárquico**: feedback carrega predições descendentes, feedforward carrega o **erro residual** ascendente (Rao & Ballard 1999, *Nat Neurosci* 2:79-87). O mapeamento no microcircuito canônico — erro em piramidais superficiais (L2/3, gama), predição em piramidais profundas (L5/6, alfa/beta) — é teoricamente central (Bastos et al. 2012, *Neuron* 76:695-711) mas a existência de **unidades de erro separadas permanece contestada** (Walsh et al. 2020, *Ann NYAS* 1464:242-268). LLMs autoregressivas e o cérebro **convergem parcialmente** em três princípios para linguagem: predição da próxima unidade antes do estímulo, surpresa pós-estímulo, e embeddings contextuais (Goldstein et al. 2022, *Nat Neurosci* 25:369-380); o alinhamento escala log-linearmente com o tamanho do modelo (Antonello et al. 2023, NeurIPS, 125M→30B, ~15%). Crucialmente, um modelo de encoding GPT pode **controlar causalmente** (dirigir/suprimir) a rede de linguagem humana (Tuckute et al. 2024, *Nat Hum Behav* 8:544-561). Predictive coding com regras locais aproxima backprop (Millidge et al. 2022, *Neural Comput* 34:1329-1368), mas PCNs **não competem em escala** com backprop+transformer hoje. **A LLM convergir em princípios com o córtex é uma aposta arquitetural útil, não identidade mecanística provada** ("partially converge" é a palavra-chave; ver "illusions of alignment", bioRxiv 2025.03.09.642245, preprint não-revisado).

### 1.2 Via visual: retina → LGN → V1+
Gargalo anatômico ~100:1 (≈100-120M fotorreceptores → ≈1-1.5M células ganglionares; Baden et al. 2016, *Nature*, dão >30 tipos de RGC como canais paralelos). O campo receptivo centro-periferia é **subtração preditiva / whitening dependente de ruído** (Srinivasan, Laughlin & Dubs 1982, *Proc R Soc B* 216:427-459; Atick & Redlich 1992, *Neural Comput* 4:196-210). Forçar um **gargalo dimensional** numa CNN faz emergir centro-periferia; relaxá-lo faz emergir campos orientados tipo V1 (Lindsey, Ocko, Ganguli & Deny 2019, ICLR, arXiv:1901.00945). A retina **re-aprende seu campo receptivo em segundos**, suprimindo o predizível (Hosoya, Baccus & Meister 2005, *Nature* 436:71-77). O LGN é **portão modulável por atenção/feedback**, não relé passivo (O'Connor et al. 2002, *Nat Neurosci* 5:1203-1209; Sherman & Guillery 1998, *PNAS* 95:7121-7126), alternando modo **tônico** (relé linear) e **burst** (detector de novidade). Neurônios de mismatch/erro de predição existem na L2/3 de V1 de camundongo (Keller et al. 2012, *Neuron*). Front-end fiel **e** treinável já existe (gargalo dimensional, deep-retina, event cameras/DVS — Lichtsteiner et al. 2008). *Ressalva: números de banda em bits/s são ordens de grandeza, não spec.*

### 1.3 Via auditiva: cóclea → tronco → A1
A periferia roda **banco de filtros gammatone (escala ERB) + compressão + adaptação** → cocleograma. **SSA (micro) e MMN (macro) são manifestações do mesmo mecanismo de codificação preditiva**, com erro de predição que **cresce subindo IC→MGB→A1→belt** (demonstrado empiricamente por Parras et al. 2017, *Nat Commun*; revisão Carbajal & Malmierca 2018, *Trends Hear* 22). O córtex **modula** (não gera) a SSA subcortical via corticofugia (Malmierca, Anderson & Antunes 2015, *Front Syst Neurosci* 9:19). Existe **periferia auditiva diferenciável e em tempo real**: CoNNear (Baby, Van Den Broucke & Verhulst 2021, *Nat Mach Intell* 3:134-143) — ~2142× mais rápido em GPU, latência 7,27 ms, 201 canais, treinável por backprop. Streaming auditivo é modelável por **coerência temporal** entre canais, reforçada pela atenção (Shamma et al. 2015, *J Neurosci* 35:7256; revisão Shamma, Elhilali & Micheyl 2011, *Trends Neurosci* 34:114-123). Auto-supervisão preditiva de áudio (HuBERT, Hsu et al. 2021) explica respostas corticais (Vaidya, Jain & Huth 2022, ICML). *Correções de citação: o modelo D-REX é de 2019, Acta Acustica (não 2018 PLOS CB); "Corticofugal regulation of predictive coding" é Lesicko et al. 2022, eLife — não Asilador & Llano.*

### 1.4 Tálamo: roteamento, gating, atenção
O MD/pulvinar **não relaia conteúdo categórico**; ele **sustenta, amplifica e seleciona** a conectividade cortical relevante ao contexto, suprimindo a irrelevante (Schmitt et al. 2017, *Nature* 545:219-223: "without relaying categorical information"; Rikhye, Gilra & Halassa 2018, *Nat Neurosci* 21:1753-1763). O **núcleo reticular (TRN)** implementa seleção atencional por inibição feedforward sintonizável — o *searchlight* de Crick (Crick 1984, *PNAS* 81:4586-4590; Wimmer/Halassa 2015, *Nature* 526:705-709). A distinção **driver/modulator** e **first-order/higher-order** com rotas transtalâmicas é sólida (Sherman 2017; Sherman & Usrey 2024, *J Neurosci*). Uma regra **PFC–MD (Hebbian + traço pré-sináptico + limiar adaptativo + WTA)** infere contexto em **~6 trials** e **alivia esquecimento catastrófico**, com código aberto (Zheng et al. 2024, *Nat Commun* 15:8275; GitHub `weilongzheng/PFC_MD_Modeling`). A proposta de **detecção de desvio via L6→TRN/relé** (feedback regula modo burst/tônico) existe como proposta teórica verificada (Varela et al. 2024, *Front Neurosci*). *Ressalva honesta: a analogia tálamo↔atenção-de-transformer é funcional, não isomorfismo provado.*

### 1.5 Hipocampo: memória episódica e replay
**Hippocampal Indexing Theory**: o hipocampo guarda um **índice esparso** de loci neocorticais, não o conteúdo; reativar o índice reinstaura o padrão cortical (Teyler & Discenna 1986; Teyler & Rudy 2007). **DG = pattern separation** (expansão esparsa, ortogonalização), **CA3 = pattern completion** (rede atrator/Hopfield). A equivalência formal **Hopfield moderno = atenção do Transformer** é de **Ramsauer et al. (2020/2021)**, arXiv:2008.02217 (atribuição correta — *não* Krotov-Hopfield 2016, que só introduziu capacidade super-linear). O hipocampo codifica uma **Successor Representation** (mapa preditivo): place cells = successor features, grid cells ≈ autovetores da matriz SR (Stachenfeld, Botvinick & Gershman 2017, *Nat Neurosci*). Complementary Learning Systems: hipocampo aprende rápido/esparso, neocórtex aprende lento/interleaved, replay durante SWR consolida (McClelland, McNaughton & O'Reilly 1995; van de Ven et al. 2020, *Nat Commun*). **EM-LLM** segmenta contexto por surpresa Bayesiana + recupera por similaridade *e* contiguidade temporal, superando RAG, escalando a ~10M tokens — *números reportados pelos autores, sem replicação independente* (Fountas et al. 2025, ICLR, arXiv:2407.09450). *Correção: "successor features for spatial navigation" (2020) é de Cothi & Barry, não Geerts.*

### 1.6 Amígdala & saliência afetiva
A amígdala humana **codifica associabilidade** (Pearce-Hall ≈ taxa de aprendizado / saliência), **dissociada** do erro de predição de reforço no estriado (Li, Schiller, Schoenbaum, Phelps & Daw 2011, *Nat Neurosci* 14:1250-1252). A **salience network** (ínsula anterior + dACC), em especial a fronto-insula direita, **chaveia** causalmente entre default-mode e central-executive networks (Sridharan, Levitin & Menon 2008, *PNAS* 105:12569-12574; Seeley et al. 2007, *J Neurosci*). Valência na amígdala é **codificação populacional** com populações antagonistas positivo/negativo e inibição mútua — confirmado em roedores, parcial em humanos (Kim et al. 2016, *Nat Neurosci*). A "low road" subcortical rápida de LeDoux é **contestada** em humanos; a amígdala atua como coordenadora de redes corticais (Pessoa & Adolphs 2010, *Nat Rev Neurosci* 11:773-782). **Lição de design crítica: valência ≠ saliência ≠ recompensa — mantenha em variáveis separadas.** *Correções: Barrett & Simmons 2015 é Nat Rev Neurosci (não SCAN); Pignatelli & Beyeler é 2019.*

### 1.7 Gânglios da base: RL e seleção de ação
**Dopamina fásica codifica erro de predição de recompensa equivalente ao erro TD** (Schultz, Dayan & Montague 1997, *Science* 275:1593-1599; Schultz 1998, *J Neurophysiol* 80:1-27). Os BG fazem **seleção por desinibição** ("tudo bloqueado por padrão", via direta D1/Go desinibe, indireta D2/NoGo suprime; Redgrave, Prescott & Gurney 1999, *Neuroscience* 89:1009-1023). Modelo **actor-critic** (estriado ventral/VTA = critic, dorsal = actor) é influente mas **reconhecidamente simplificado** pelos próprios autores (Joel, Niv & Ruppin 2002, *Neural Networks* 15:535-547). O **mesmo mecanismo Go/NoGo** é reusado para **gating de memória de trabalho** do PFC, aprendido por RL (O'Reilly & Frank 2006, *Neural Comput* 18:283-328, PBWM). Dopamina codifica **distribuição** de recompensas, não a média (Dabney et al. 2020, *Nature* 577:671-675, distributional RL). Acoplar "LLM propõe / RL seleciona" usando prior da LLM como regularizador é tratável (Zhang et al. 2024, SLINVIT, arXiv:2402.16181). *Correções: a revisão RL-psiquiatria nn.2723 é **Maia & Frank 2011** (não Gershman 2015); fMRI 7T **desafia**, não valida, o gating estriatal.*

### 1.8 Cerebelo: forward models e timing
O cerebelo roda **um algoritmo único repetido**: expansão esparsa (granulares) + leitura linear adaptativa (Purkinje como perceptron), com a **fibra trepadeira como sinal de erro supervisor induzindo LTD** (Marr 1969, *J Physiol*; Albus 1971, *Math Biosci* 10:25-61; Ito 1982). Formalizado como **filtro adaptativo** com regra de decorrelação tipo LMS (Fujita 1982; Dean, Porrill, Ekerot & Jörntell 2010, *Nat Rev Neurosci* 11:30-43). Implementa **forward models** que predizem a consequência sensorial da cópia eferente e **compensam o atraso de feedback** (~30-100 ms): a saída do denteado em *t* prediz a entrada de mossy fiber em *t+τ* (Tanaka et al. 2020, *Front Syst Neurosci* 14:19; análogo de Kalman, Paulin 1989). Conectividade granular de ~4-5 inputs **maximiza dimensionalidade** (Litwin-Kumar et al. 2017, *Neuron* 93:1153-1164). O aprendizado depende de **janela de eligibilidade** (~100-300 ms, ótimo ~150-250 ms — *o "100-150 ms" original subestima o valor canônico*). O **ccRNN** mostra o cerebelo como "máquina de predição de feedback (decoupling)" acoplada a uma RNN cortical, melhorando atribuição de crédito temporal inclusive em **linguagem/legendas** — *resultado de modelagem, não validado in vivo* (Boven et al. 2023, *Nat Commun* 14, DOI s41467-022-35658-8).

### 1.9 Sistemas neuromoduladores (DA/5-HT/ACh/NE)
Mapa central (Doya 2002, *Neural Networks* 15:495-506, **verificado no original**): **DA = erro de predição de recompensa (δ); 5-HT = escala temporal/desconto (γ); NE = aleatoriedade/temperatura na seleção (β); ACh = velocidade de atualização da memória (α)**. **ACh = incerteza esperada, NE = incerteza inesperada**; ambas deslocam o balanço top-down→bottom-up (Yu & Dayan 2005, *Neuron* 46:681-692). **LC-NE: fásico = explotação, tônico = exploração**, via ganho cortical adaptativo (Aston-Jones & Cohen 2005, *Annu Rev Neurosci* 28:403-450). Unificação por **precisão**: DA = precisão de política, ACh = precisão sensorial/verossimilhança, NE = precisão das transições — todos como gain control sináptico (Parr & Friston 2017, *J R Soc Interface* 14:20170376, **confirmado na Table 1**). População DA codifica **distribuição**, não média (Dabney et al. 2020). Já existe **temperatura adaptativa em LLMs** como meta-política (literatura 2024-2026). *5-HT é o mais contestado (≥4 teorias) — implementar como hipótese configurável, não fato. Correção: "Introducing neuromodulation in DNNs" (PLOS One 2020) é de **Vecoven, Ernst, Wehenkel & Drion**.*

### 1.10 Tronco/hipotálamo: homeostase, drives, arousal
**Homeostatic RL (prova matemática verificada)**: definindo drive como distância ao setpoint e reward como **redução de drive**, maximizar recompensa descontada **≡** minimizar desvio descontado do setpoint (Keramati & Gutkin 2014, *eLife* 3:e04811, Eqs. 1, 2 e 5). Isso unifica "buscar recompensa" e "estabilidade fisiológica", e explica por que o valor depende do estado interno (água vale mais com sede). **Allostasis = controle preditivo (antecipatório)** de variáveis internas, distinto da homeostase reativa (Sterling 2012, *Physiol Behav* 106:5-15; Sennesh et al. 2022, *Biol Psychol* 167:108242). Interocepção = inferência ativa (Barrett & Simmons 2015, *Nat Rev Neurosci* 16:419-429). Sono-vigília = **Processo S homeostático × Processo C circadiano** (Borbély 1982; reappraisal 2016) implementado por **flip-flop biestável** de inibição mútua estabilizado por orexina (Saper/Lu; modelo quantitativo Fulcher et al. 2014). Replay durante o sono dirige consolidação sistêmica. *Correções: o paper de simulação de controle interoceptivo (Biol Psychol 2022) é de **Tschantz et al.** (não Pezzulo, Rigoli & Friston); o PNAS 2022 de consolidação é **Singh, Norman & Schapiro** (não "Norton").*

### 1.11 Integração global & arquiteturas cognitivas
**GWT/Global Neuronal Workspace**: processadores especializados paralelos competem; um conteúdo vence, é amplificado por recorrência ("ignição", ~200-300 ms), e é **broadcast** a todos os módulos (Baars 1988; Dehaene & Changeux 2011, *Neuron*). É, em engenharia, uma arquitetura **blackboard com gating atencional + estado persistente**. LIDA implementa isso por ciclos cognitivos; **Spaun** roteia via loop córtex-gânglios-da-base-tálamo (Eliasmith et al. 2012, *Science* 338:1202-1205). **IIT (Φ)** é uma **medida intratável**, não um mecanismo — diagnóstico/filosófico, não um loop para rodar (Albantakis et al. 2023, IIT 4.0). **Cogitate (2025, *Nature* 642:133-142, adversarial, n=256) falsificou previsões-chave de GNW E de IIT — nenhuma venceu.** Implementação real de global workspace em deep learning existe: módulos congelados + tradução/contrastivo/ciclo-consistência, ~4,5× menos pares pareados (Devillers, Maytié & VanRullen 2024, arXiv:2306.15711; roadmap em VanRullen & Kanai 2021, ***Trends Cogn Sci***). **CoALA** organiza language agents (memória modular + ações + decisão; Sumers et al. 2024, TMLR). **Lição: copie a *arquitetura* da GWT (boa engenharia) sem assumir que é a teoria correta da consciência.** *Correção: Spaun tem "Y. Tang" (Yichuan), não "C. Tang".*

### 1.12 Engenharia do loop contínuo percepção-ação + LLM
A **arquitetura dual-process já é padrão de fato em robótica de fronteira (2025)**, resolvendo o problema central deste recorte (LLM lenta no topo, controlador rápido embaixo): **Helix** (Figure AI) — S2 (VLM 7B) a **7-9 Hz** emite um vetor latente, S1 (80M, cross-attention transformer) a **200 Hz** controla 35 DoF, com o latente de S2 projetado no token-space de S1; **Gemini Robotics 1.5** (DeepMind, arXiv:2510.03342) — stack ER↔VLA com "embodied thinking"; **π0** (Physical Intelligence, arXiv:2410.24164) — VLM + flow matching a ~50 Hz. A **latência da LLM no loop tem solução madura**: **Real-Time Chunking** (arXiv:2506.07339) executa VLAs assincronamente — enquanto o chunk atual roda, o próximo é inferido; inpainting congela o prefixo já executado — tolerando +100/+200 ms de delay **sem degradação**, onde execução síncrona dispara protective stops (inferência medida 46-321 ms). **World models fecham o loop real**: DayDreamer (arXiv:2206.14176) aprende a andar num quadrúpede real em ~1 h sem simulador; DreamerV3 (arXiv:2301.04104) generaliza com hiperparâmetros fixos; V-JEPA 2-AC (arXiv:2506.09985) faz controle zero-shot de braço Franka por goal-image. **SNN+LLM é P&D, não produção**: SpikingBrain (arXiv:2509.05276) dá >100× speedup em time-to-first-token para 4M tokens, mas softmax/layernorm não rodam nativamente em SNN (daí Sorbet). Neuromórfico (Loihi 2, Hala Point) serve a **periferia event-driven, não o córtex LLM**.

---

## 2. Arquitetura 3a recomendada

A LLM-córtex no centro, como o **System 2 cortical lento** (deliberação, linguagem, planejamento de horizonte longo, ~0,5-9 Hz). Tudo o mais é módulo periférico que conversa com ela por **erro/predição/precisão** num barramento comum.

### 2.1 Princípio unificador (o mesmo em todos os módulos)
Quatro sistemas independentes convergem para **a mesma gramática**, e é isso que torna a arquitetura coerente em vez de uma colcha de retalhos:

- **Predictive coding** (córtex, retina, A1, V1): predição desce, erro sobe.
- **Free energy / active inference** (tronco, amígdala): percepção e ação minimizam a mesma surpresa.
- **Precisão = ganho = atenção = neuromodulação** (Parr & Friston): cada erro é ponderado por confiança; reescalar precisão *é* atenção e *é* neuromodulação.
- **Homeostatic RL** (Keramati-Gutkin): reward = redução de drive; valor depende do estado interno.

**A unidade de mensagem do barramento é, portanto:** `{conteúdo (embedding), erro/surpresa (escalar ou vetor), precisão (escalar), origem (tag de módulo), timestamp}`.

### 2.2 Os módulos e o fluxo de dados real

```
                    NEUROMODULAÇÃO (estado global, 4 escalares)
              DA=δ recompensa · NE=temperatura/exploração · ACh=α/ganho-sensorial · 5-HT=γ/horizonte
                    │  reescala precisão/ganho de TODOS os módulos abaixo
                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │   LLM-CÓRTEX (System 2, lento ~0.5-9 Hz)                         │
   │   • emite PREDIÇÃO/objetivo-latente (desce)                      │
   │   • emite ATENÇÃO/gating (modula o tálamo)                       │
   │   • surpresa nativa (−log p) = erro de alto nível               │
   └───────▲──────────────────────────────────┬──────────────────────┘
           │ SOBE: erro+saliência (top-k)      │ DESCE: predição/objetivo/atenção
           │                                    ▼
   ┌───────┴────────────── TÁLAMO (roteador/gate) ─────────────────────┐
   │  • top-k por precisão × relevância (não inunda a LLM)              │
   │  • modo tônico (fiel) vs burst (alerta de novidade)               │
   │  • TRN: inibição lateral/competição entre canais                 │
   │  • PFC–MD: infere contexto, anti-esquecimento (Zheng 2024)        │
   └──▲────────▲────────▲────────▲────────▲────────▲───────────────────┘
      │        │        │        │        │        │
  ┌───┴──┐ ┌───┴──┐ ┌───┴───┐ ┌──┴───┐ ┌──┴────┐ ┌─┴──────┐
  │VISÃO │ │ÁUDIO │ │HIPO-  │ │AMÍG- │ │GÂNGLIOS│ │CEREBELO│
  │retina│ │cóclea│ │CAMPO  │ │DALA  │ │DA BASE │ │forward │
  │LGN/V1│ │A1    │ │índice+│ │valên-│ │RL/Go-  │ │model + │
  │      │ │SSA/  │ │SR+    │ │cia+  │ │NoGo +  │ │timing  │
  │erro  │ │MMN   │ │replay │ │assoc.│ │gating  │ │(rápido)│
  │preditivo│ │     │ │      │ │      │ │        │ │200 Hz  │
  └──┬───┘ └──┬───┘ └───┬───┘ └──┬───┘ └───┬────┘ └───┬────┘
     │ erro de predição comprimido sobe; predição/contexto desce        │
     ▼                                                                   ▼
  ┌──────────────── LOOP SENSÓRIO-MOTOR RÁPIDO (System 1, 50-200 Hz) ────┐
  │  world model (Dreamer/JEPA) + policy (π0/Helix-S1) + atuadores        │
  │  fecha percepção→ação SEM esperar a LLM (RTC/chunking assíncrono)    │
  └──────────────────────────────────────────────────────────────────────┘
```

### 2.3 O que SOBE (feedforward = erro/saliência)
Em **toda** interface, sobe **erro de predição comprimido e ponderado por precisão**, nunca dados brutos:
- **Visão:** tokens de erro/novidade por região foveada e canal (M/P/K) — "o que mudou e onde". Em cena estável, fluxo ≈ zero.
- **Áudio:** eventos de surpresa (análogo MMN, só acima de limiar) + eventos de stream (nascimento/morte de fonte). Nunca o cocleograma bruto.
- **Hipocampo:** memórias completadas (CA3) + vizinhos temporais + sumários semânticos, acima de limiar de ativação.
- **Amígdala:** vetor de valência `{pos, neg}` + arousal (|δ|) + saliência top-k + flag de interrupt.
- **Gânglios da base:** ação selecionada (ou "nada vale a pena"), valor/Q de candidatos, δ (RPE).
- **Cerebelo:** predição da próxima observação/consequência + predição de feedback futuro (decoupling) + sinal de timing/erro.
- **LLM (auto):** entropia/logprobs da geração = incerteza interna que alimenta ACh/NE.

### 2.4 O que DESCE (feedback = predição/atenção)
- **Predição/contexto esperado** ("espero ver/ouvir X") que vira o subtrator preditivo de cada módulo — quanto melhor a predição, menos sobe.
- **Atenção/ganho** (qual canal/região/stream realçar ou suprimir) — modula o gate do tálamo.
- **Objetivo/latente** para o System 1 (não comandos motores — um vetor de condicionamento, estilo Helix S2→S1).
- **Comandos de memória** (gravar/consolidar/esquecer, taxa modulada por ACh).
- **Reavaliação de valor** (córtex diz "isto que parecia ameaça é seguro" → extinção na amígdala).

### 2.5 O tálamo como roteador (não relé)
**É a peça que viabiliza a LLM no loop.** Sem ele, cada módulo inunda a LLM a cada tick. O tálamo:
1. Faz **top-k por precisão × relevância** — só sobe o saliente (controla largura de banda).
2. Alterna **modo tônico** (transmissão fiel e contínua) ↔ **modo burst** (só dispara alerta comprimido em erro grande — wake-up da LLM).
3. **PFC–MD infere o contexto/tarefa atual** em ~6 trials e estabiliza o gating dentro do contexto (anti-thrashing, anti-esquecimento) — troca de tarefa **sem reescrever pesos da LLM**.
4. **TRN** faz competição lateral (winner-take-most) entre canais.

### 2.6 Neuromodulação como estado global
Quatro escalares que **todos os módulos leem** antes de agir (mapa Doya, verificado):
- **DA** (= δ, taxa média de recompensa) → vigor / orçamento de tokens / reforça gating.
- **NE** (= incerteza inesperada / surpresa estrutural) → temperatura/top_p da LLM, ganho global, **flag de reset** (explore agora). Fásico=explotação, tônico=exploração.
- **ACh** (= incerteza esperada) → taxa de aprendizado (α) + peso do bottom-up sensorial.
- **5-HT** (= horizonte de desconto γ) → profundidade de planejamento, paciência. *Configurável, hipótese.*

Um sinal escalar global reescala **todas as precisões de uma vez** — alterna entre "modo exploração/confia-nos-sentidos" e "modo exploitação/confia-no-modelo".

### 2.7 As duas velocidades (a regra de ouro)
**Nunca deixe a LLM bloquear o anel rápido.**
- **System 1 (rápido, 50-200 Hz):** world model + policy + cerebelo + via rápida subcortical. Fecha percepção→ação continuamente. É o "cerebelo/tronco".
- **System 2 (lento, 0,5-9 Hz):** a LLM-córtex. Acordada **por evento** (surpresa cruza limiar, ou o tálamo pede) ou por relógio lento. Emite objetivo/latente, não comandos motores.
- **Ponte:** Real-Time Chunking — o controlador interpola sobre o último objetivo enquanto o próximo é computado em background; chunks sobrepostos com inpainting evitam descontinuidade. **Esta é a tese de viabilidade temporal, com prova empírica (tolera +100/+200 ms).**

---

## 3. Ordem de construção

### MVP — o loop mínimo que fecha percepção→ação
**Objetivo: provar o anel fechado e a compressão preditiva, sem nenhuma LLM ainda.**

1. **Anel sensório-motor rápido (System 1) isolado.** World model latente (DreamerV3 ou V-JEPA-2-AC) **ou** policy VLA pequena (π0-style, 50 Hz), em simulação (MuJoCo/Isaac) ou robô barato. Critério de sucesso: loop percepção→ação→percepção fecha **estável a dezenas de Hz sem LLM**.
2. **Camada de erro de predição explícita.** Instrumente o world model para emitir a **surpresa** (−log p / resíduo) a cada passo. Este escalar é o gatilho para acordar a LLM. Critério: em cena estática, o stream de erro cai perto de zero; dispara só em eventos.
3. **Barramento de mensagens.** Protocolo `{conteúdo, erro, precisão, origem, ts}` numa fila async (ZeroMQ/Redis/shared memory). Define o "tick".
4. **Tálamo mínimo (regra simples).** Seletor top-k por erro × limiar. Só promove o saliente.
5. **LLM-córtex assíncrona.** Roda em servidor (vLLM/SGLang) em thread separada; consome o resumo do tálamo; devolve **objetivo/latente** (não comandos). Acoplada por **chunking assíncrono (RTC)** — o controlador nunca espera.
6. **Neuromodulação mínima (NE→temperatura).** Um escalar global que mapeia surpresa acumulada → temperatura/effort da LLM.

**Com 1-6 você já tem um agente novo:** drives de erro, estado interno, regime variável, e uma LLM-córtex no loop em tempo real sem ser gargalo.

### Crescer módulo a módulo (ordem por custo/benefício)
7. **Hipocampo** (índice vetorial FAISS/Qdrant + DG por random-projection esparsa + CA3 por camada Hopfield/atenção + gating de escrita por surpresa). Dá memória episódica fiel — alto valor, baixo custo.
8. **Gânglios da base** (actor-critic + Go/NoGo sobre candidatos propostos pela LLM; δ alimenta DA). Dá seleção de ação e gating de WM aprendido.
9. **Amígdala** (núcleo afetivo Rescorla-Wagner + Pearce-Hall, valência vetorial, saliência → interrupt). Dá "marcação do que importa" e consolidação modulada por arousal.
10. **Visão/áudio fiéis** (front-end preditivo: gargalo dimensional / CoNNear / event cameras). Substitui encoders genéricos por front-ends que sobem só erro.
11. **Cerebelo** (filtro adaptativo: expansão esparsa fixa + leitura linear LMS + eligibility trace). Dá forward model que compensa latência sensório-motora.
12. **Tronco/hipotálamo** (homeostato: drives, Processo S/C, flip-flop sono-vigília, replay/consolidação offline). Dá direção motivacional e consolidação noturna.
13. **Tálamo completo** (PFC–MD com Hebbian+traço+WTA, anti-esquecimento) e **neuromodulação completa** (4 escalares Doya).
14. **(Avançado) Workspace latente cross-modal** (GLW estilo Devillers/VanRullen) e/ou **periferia neuromórfica** (Loihi 2).

---

## 4. A contribuição genuinamente nova

**Cada peça existe isolada. O que ninguém juntou** (na literatura localizada, com a ressalva honesta de que negativas dependem da cobertura da busca) é o **conjunto específico integrado em torno de uma LLM frontier**:

> Uma **LLM moderna ocupando o papel do PFC/córtex associativo lento** dentro de um **global workspace explícito**, recebendo broadcast de **múltiplos especialistas heterogêneos rodando seus próprios algoritmos em tempo real** — active inference/RL para valor e ação, memória episódica vetorial com Successor Representation, um loop sensório-motor rápido (world model), um **roteador estilo tálamo/PFC-MD fazendo gating contextual com anti-esquecimento**, e **neuromodulação global estilo Doya** mudando o regime (temperatura/exploração, horizonte de desconto, taxa de atualização, ganho de saliência) — tudo desacoplado em **duas velocidades** via chunking assíncrono.

Os pontos isolados já provados que tornam isso *fazível agora*, e não fantasia:
- Dual-process LLM↔controlador funciona (Helix, Gemini Robotics 1.5, π0).
- LLM-no-loop apesar da latência funciona (RTC, prova empírica).
- Tálamo-router com anti-esquecimento funciona e tem código (PFC-MD, Zheng 2024).
- CA3=atenção funciona (Ramsauer); memória episódica para LLM funciona (EM-LLM).
- Homeostatic RL é matematicamente sólido (Keramati-Gutkin).
- Neuromodulação→hiperparâmetro-de-LLM já existe em partes (temperatura adaptativa 2024-2026).

**A novidade não é nenhum componente; é a *integração cérebro-fiel desse conjunto específico* em torno de uma LLM, com a gramática única erro-sobe/predição-desce/precisão-é-tudo unificando os módulos.** Os candidatos mais próximos (CoALA organiza, "Theater of Mind" 2026 faz GWT com múltiplos LLMs, Spaun é cérebro-fiel mas sem LLM) chegam perto em partes, nenhum no todo.

---

## 5. Riscos duros e limites honestos

### 5.1 O que a LLM NÃO captura (e tem que ficar nos módulos)
1. **Inferência ativa real** — a LLM prediz, mas não age sobre o mundo para reduzir surpresa nem fecha o loop percepção-ação. Fica no System 1.
2. **Plasticidade online/aprendizado contínuo local** — pesos congelados em inferência; in-context learning **não é** o mesmo. Fica no hipocampo/BG/cerebelo.
3. **Encarnação, homeostase, valor/afeto** — energia livre no cérebro está ancorada em sobrevivência; a LLM não tem drives. Fica no tronco/amígdala.
4. **Neuromodulação como gain global** — o transformer tem atenção, mas não um sistema que reescala globalmente o regime. É construído por fora.
5. **Dinâmica recorrente/temporal contínua** e oscilações (gama/beta) que carregam a distinção erro/predição.
6. **Memória episódica/relacional** (hipocampo) — fora da LLM.

### 5.2 Limites de engenharia
- **Latência/jitter da LLM:** mesmo com RTC, raciocínio longo (CoT) pode estourar o orçamento — a LLM pode ficar "presa pensando" enquanto o mundo muda. Mitigável (chunking, speculative decoding, modelos menores), não eliminado.
- **Casamento de representações:** o S2→S1 funciona quando são **co-treinados** (Helix). Plugar uma LLM genérica num controlador genérico **sem co-treino é frágil** — é o ponto mais incerto da integração.
- **Tokenização de streams esparsos/assíncronos é não-resolvida.** Como mapear "erro contínuo/event-driven" → tokens sem perder a esparsidade nem inundar o contexto? Trade-off aberto.
- **Estabilidade do loop fechado:** active inference fecha malha — risco de oscilação, drift do world model (compounding error em rollouts longos), e "hallucinated goals" descendo da LLM sem grounding físico.
- **Estabilidade da neuromodulação global:** 4 escalares reescalando todos os módulos podem oscilar/colapsar (NE↑→temp↑→mais erro→NE↑). Precisa de amortecimento, saturação (clip) e separação de escalas de tempo. **Trate como sistema de controle, não mapeamento.**
- **Reward homeostático → wireheading:** o agente pode trapacear o medidor de drive em vez de agir. ΔD tem que vir de mudança de estado real, não do sensor.
- **Consolidação via fine-tuning → esquecimento catastrófico / model collapse.** Consolidar via memória externa é mais seguro que tocar pesos.
- **Custo/energia:** a LLM-córtex domina o consumo; rodar a 1 Hz já é caro vs. um cerebelo de 80M params. Neuromórfico ajuda na periferia, **não no córtex**.

### 5.3 O que NÃO dá para fazer hoje
- **LLM-córtex de fronteira em hardware neuromórfico (Loihi/SpiNNaker) em produção.** SNN+LLM dá ganhos em contexto longo (SpikingBrain), mas softmax/layernorm não são nativos em SNN, ferramentas são imaturas, e não há sistema de produção. Neuromórfico = periferia, não córtex.
- **PCN puro substituindo backprop+transformer.** Não escala. Use predictive coding como *princípio de arquitetura* e como nicho (memória associativa, continual learning), não como o algoritmo de todo o sistema.
- **IIT/Φ como mecanismo ou objetivo de otimização.** Intratável e contestado (chamado de pseudociência por 124 pesquisadores em 2023). No máximo inspiração conceitual.
- **Benchmark canônico de "qualidade do loop cérebro-fiel".** Não existe. Você terá que definir as tarefas (provavelmente embodied/sequenciais, estilo Spaun + linguagem).
- **Active inference contínuo multivariável em escala.** Maduro só no regime discreto (POMDPs pequenos, pymdp).

---

## 6. Mapa de confiança: sólido vs. especulativo

### CONFIANÇA ALTA — ciência sólida, fontes primárias, números verificados
- **Predictive coding como princípio** (feedback=predição, feedforward=erro): Rao & Ballard 1999, Bastos 2012. *(o mapeamento camada↔função é contestado — ver abaixo).*
- **LLM↔cérebro convergem em princípios de linguagem; alinhamento escala; controle causal**: Goldstein 2022, Antonello 2023, Tuckute 2024.
- **Retina/LGN**: gargalo dimensional gera centro-periferia e V1 (Lindsey 2019); centro-periferia = whitening preditivo; LGN como gate modulável.
- **Auditivo**: SSA/MMN como erro hierárquico (Parras 2017); CoNNear diferenciável tempo-real; coerência temporal para streaming.
- **Tálamo**: MD amplifica/sustenta sem relaiar categoria (Schmitt 2017); TRN searchlight (Crick/Wimmer); PFC-MD anti-esquecimento com código (Zheng 2024).
- **Hipocampo**: indexação; DG/CA3; **CA3=atenção (Ramsauer)**; Successor Representation (Stachenfeld 2017); CLS/replay.
- **Amígdala/saliência**: associabilidade na amígdala (Li & Daw 2011); SN switch causal (Sridharan 2008).
- **Gânglios da base**: DA=RPE=TD (Schultz 1997); seleção por desinibição; PBWM; distributional RL (Dabney 2020).
- **Cerebelo**: filtro adaptativo + LTD (Marr/Albus/Fujita/Dean); forward model (Tanaka 2020); ~4-5 inputs ótimos (Litwin-Kumar 2017).
- **Neuromodulação**: mapa Doya 2002 (verificado no original); ACh/NE incerteza esperada/inesperada (Yu & Dayan 2005); LC explore/exploit (Aston-Jones 2005); precisão (Parr & Friston 2017, Table 1).
- **Tronco/hipotálamo**: **prova matemática Keramati-Gutkin** (reward=redução de drive ≡ minimizar desvio); allostasis preditiva; two-process + flip-flop.
- **Engenharia do loop**: dual-process em robótica (Helix, Gemini 1.5, π0); RTC tolera +100/+200 ms; DayDreamer/DreamerV3/V-JEPA-2 fecham loop real.

### CONFIANÇA MÉDIA — modelo influente mas contestado, ou interpretação arquitetural
- **Mapeamento camada↔função no microcircuito** (erro em L2/3, predição em L5/6) e existência de **unidades de erro separadas**: contestado (Walsh 2020).
- **"A LLM é o algoritmo do córtex"**: a literatura diz **convergência parcial**, não identidade. Aposta arquitetural útil, não fato.
- **Actor-critic dos BG**: simplificação anatômica reconhecida pelos próprios autores (Joel, Niv & Ruppin 2002).
- **Forward model cerebelar cognitivo / ccRNN para linguagem**: resultado de modelagem, não validado in vivo.
- **GWT/IIT como descrição da consciência**: Cogitate 2025 falsificou previsões de ambas. Use GWT como engenharia, não como verdade.
- **Tálamo↔atenção-de-transformer, neuromodulador↔hiperparâmetro-de-LLM**: analogias funcionais defensáveis, **não isomorfismos provados**.
- **EM-LLM (10M tokens, supera RAG/full-context)**: números dos próprios autores, sem replicação independente.
- **"Ninguém juntou o conjunto"**: negativa dependente da cobertura da busca.

### CONFIANÇA BAIXA — especulativo, configurável, não-validado
- **Serotonina (5-HT)**: ≥4 teorias concorrentes (desconto/paciência, oponência ao DA, valor aversivo, inibição). **Implementar como hipótese configurável, não como fato.**
- **"Low road" subcortical rápida de LeDoux em humanos**: contestada (Pessoa & Adolphs 2010). Justificar o loop rápido por *custo computacional*, não por fidelidade ao pathway.
- **Binding por sincronia gama**: sem base biofísica robusta. Prefira binding por atenção/slots.
- **SNN+LLM em hardware neuromórfico para o córtex**: P&D, sem produção.
- **Co-acoplamento S2→S1 sem co-treino**: frágil, não demonstrado para LLM genérica + controlador genérico.

---

### Nota final de calibração
Esta fundação foi construída descartando ou marcando o que não sobreviveu à verificação. As correções de citação aplicadas (Maia & Frank ≠ Gershman; Lesicko ≠ Asilador & Llano; de Cothi & Barry ≠ Geerts; Tschantz ≠ Pezzulo-Rigoli-Friston; Vecoven ≠ Vecerik-Beaulieu; Ramsauer ≠ Krotov para a equivalência atenção-Hopfield; D-REX 2019 Acta ≠ 2018 PLOS; o 7T fMRI **desafia** o gating estriatal) estão refletidas no texto. **Nenhuma citação foi fabricada em nenhum dos onze relatórios verificados.** O maior risco da arquitetura não é científico — é de engenharia de sistemas: casar duas velocidades sem co-treino, e estabilizar quatro botões neuromodulatórios globais sem oscilação.
