# CENÁRIO E LIÇÕES — projeto cérebro-fiel 3a

> Síntese de cinco clusters de pesquisa já verificados (arquiteturas cognitivas clássicas; LLM como arquitetura cognitiva; Global Workspace + LLM; world models + dual-process; active inference / predictive coding como agente; e o mapa de quem constrói isto agora) contra cada decisão de design do 3a.
>
> **Design 3a (referência):** LLM de fronteira CONGELADA como córtex (System 2 lento) + módulos cérebro-fiéis (visão/áudio preditivos, hipocampo episódico, amígdala afetiva, gânglios da base RL, cerebelo forward-model, tálamo-roteador) + loop sensório-motor rápido (System 1, world model) + neuromodulação global (4 escalares) + gramática única (predição desce, erro sobe, precisão=ganho=atenção) ancorada em energia livre / predictive coding. Duas velocidades em ponte por chunking assíncrono (RTC). Construção por etapas, começando pela VISÃO fiel.
>
> Data de referência: junho/2026. Cada item carrega o status de verificação do veredito de origem. Tratamos PR de empresa como PR, preprint como preprint, peer-review como peer-review.

---

## 1. O que CONFIRMA nossas escolhas

Validação externa, decisão por decisão. Quando grupos independentes, partindo de premissas diferentes, chegam ao mesmo lugar que nós, isso é sinal de que a escolha não é arbitrária.

### 1.1 O mapa de módulos por região cerebral — CONFIRMADO por convergência tripla independente

- **Leabra / O'Reilly, Hazy & Herd (Oxford Handbook of Cognitive Science, 2016/2017)** [CONFIRMADO] chega, por caminho puramente neurocientífico, a quase o mesmo inventário do 3a: córtex posterior preditivo, hipocampo episódico esparso, amígdala/OFC afetiva, gânglios da base RL com gating, PFC de manutenção ativa, cerebelo, dopamina como neuromodulador global.
- **Hong Jeong, "A Miniature Brain Transformer" (arXiv:2603.07217, mar/2026)** [CONFIRMADO] — módulos relé talâmico (gating), hipocampo lateralizado, saliência amigdaloide, buffer PFC, fast-path cerebelar. Quase isomórfico ao nosso, e é de 2026. **Ressalva:** ele não menciona gânglios da base explicitamente, e o cerebelo dele é "fast-path", não forward-model RL — diferença sutil a favor da nossa especificação mais completa.
- **AXIOM (Heins et al., arXiv:2505.24784, mai/2025)** [CONFIRMADO, números batem dígito a dígito] — coleção de módulos com papéis distintos (sMM = parsing de pixels em slots; iMM = tipos de objeto; tMM = protótipos de movimento; rMM = interações relacionais). Os "modos dinâmicos" do tMM são quase exatamente o papel forward-model do cerebelo no nosso design.
- **CoALA (Sumers et al., TMLR 2024)** [CONFIRMADO] + 4 sistemas (Princeton, Stanford, NVIDIA, BAAI) convergem em **separar memória episódica / semântica / procedural** — exatamente nossa separação hipocampo / conhecimento-de-mundo / gânglios-cerebelo.

**Veredito:** o particionamento funcional do 3a é o consenso emergente, peer-reviewed e replicado em 2024-2026. Não é especulação.

### 1.2 LLM congelada + módulos que aprendem ao redor — CONFIRMADO como viável

- **Voyager, Generative Agents, Reflexion** [todos CONFIRMADOS] provam que se extrai comportamento de longo horizonte de um LLM **congelado** escrevendo em memória externa, sem tocar pesos.
- **GWA / Shang (arXiv:2604.08206, abr/2026)** [CONFIRMADO que existe e que congela o LLM] e **toda a linhagem VanRullen congela os encoders** — três grupos distintos convergiram em "congele o componente caro e poderoso, aprenda o roteamento".
- **A primitiva exata do 3a está sendo construída em 2025-2026 com benchmarks:** PaceLLM (arXiv:2506.17310), **NeuroLoRA (arXiv:2603.12378)** — gate de neuromodulação aprendível sobre projeções congeladas, com benchmark MMLU/GSM8K/ScienceQA — e Brainstacks (arXiv:2604.01152). O paradigma tem nome: **"Frozen-LM Conditioning"**.

**Veredito:** congelar + módulos treináveis funciona e tem tração. (Mas ver §5: isto também derruba parte da nossa alegação de ineditismo.)

### 1.3 Duas velocidades (System 1 rápido / System 2 lento) — CONFIRMADO como consenso de 2025

Padrão recorrente verificado: **lento ~1–10 Hz / rápido ~30–200 Hz**.

| Sistema | S2 (lento) | S1 (rápido) | Status |
|---|---|---|---|
| Helix (Figure, 2025) | VLM 7–9 Hz | política 200 Hz | [CONFIRMADO] |
| GR00T N1 (NVIDIA, 2025) | VLM 10 Hz | DiT 120 Hz | [CONFIRMADO] |
| DuoCore-FS (Astribot, 2025) | 1–3 Hz | 25–30 Hz | [CONFIRMADO] |
| Talker-Reasoner (DeepMind, 2024) | Reasoner | Talker (Flash) | [CONFIRMADO] |

Cinco labs independentes (Figure, NVIDIA, DeepMind, PI, acadêmicos) convergiram nisso. Nosso envelope de frequências está correto — não inventamos uma razão impossível.

### 1.4 Ponte por vetor latente + chunking assíncrono (RTC) — CONFIRMADO e peer-reviewed

- **RTC (Black, Galliker, Levine, arXiv:2506.07339, NeurIPS 2025)** [CONFIRMADO] — a RTC que o 3a referencia é um algoritmo real, training-free, com código em LeRobot. Tolera **+200–300 ms** de latência sem quebrar o controle (verbatim da PI: "performance remains completely unchanged up to +200ms").
- **Helix** [CONFIRMADO] passa **um único vetor latente contínuo** de S2 para S1 — valida o tálamo-roteador passando um latente, não um plano em texto.
- **DuoCore bridge buffer** [CONFIRMADO] — S2 atualiza buffer de latentes a 1–3 Hz, S1 lê o mais recente sem bloquear. Receita exata do tálamo-roteador.
- **Talker-Reasoner** [CONFIRMADO] — memória compartilhada de belief states, Talker lê o último disponível (possivelmente velho), Reasoner atualiza assíncrono. Blueprint pronto do RTC.

### 1.5 Gramática única ancorada em energia livre — CONFIRMADA como família, funciona como agente

- **AXIOM** [CONFIRMADO] roda minimização de **energia livre esperada** (utilidade + ganho de informação) como única regra de ação/percepção/aprendizado, e **bateu DreamerV3 (420M) e BBF em Gameworld 10k com 0,3–1,6M params e 18 ms/step** (update) vs 221 ms + 823 ms (planejamento) do Dreamer. A lei única **funciona como agente eficiente em regime de baixos dados e estrutura forte**.
- **Predictive coding ≈ backprop é teorema** (Millidge et al., Neural Computation 2022) [CONFIRMADO] — cobertura teórica para usar erro/precisão como mecanismo de aprendizado local nos módulos plásticos.
- **O'Reilly et al. (J. Cog. Neuroscience 2021)** [CONFIRMADO] — predição top-down no pulvinar, resultado bottom-up, sinal de diferença temporal, plasticidade local, **sem população de erro dedicada**. Predictive coding biologicamente plausível, já implementado.
- **Consciousness Prior (Bengio 2017)** [CONFIRMADO] define o estado consciente via **dip na função de energia** com fator-grafo esparso — mesma família matemática da nossa free-energy.
- **JEPA → V-JEPA 2** [CONFIRMADO] é literalmente "predição em espaço de embedding minimizando uma função de energia goal-conditioned", e V-JEPA 2-AC **planeja num robô real**.

> ⚠️ **Ressalva honesta (do próprio veredito):** o mapeamento de XCAL / O'Reilly-2021 à **energia livre de Friston** é **análogo, não identidade matemática provada**. O'Reilly chama de "temporal-difference / error-driven", não deriva da free-energy. Tratar como "mesma família", não "exatamente a mesma gramática". O salto retórico de "análogo a predictive coding" para "exatamente a nossa gramática ancorada em energia livre" é exagero leve — calibrar a linguagem.

### 1.6 Neuromodulação por 4 escalares globais — CONFIRMADA, com pedigree de 20+ anos

- **Doya, "Metalearning and neuromodulation" (Neural Networks, 2002)** [CONFIRMADO, mapa exato]: **dopamina = erro de predição de recompensa; serotonina = escala temporal / desconto; noradrenalina = exploração vs execução; acetilcolina = velocidade de atualização de memória.** É literalmente o mapa dos nossos 4 escalares.
- **Feldman & Friston (Frontiers Human Neuroscience, 2010)** [CONFIRMADO, verbatim]: "precisão é codificada pelo ganho sináptico das unidades que reportam erro de predição" = nossa equação **precisão = ganho = atenção**.
- **PVLV / Hazy, Frank & O'Reilly (2010)** [CONFIRMADO] — dopamina como sinal de aprendizado/saliência broadcast global, não "prazer".
- **NeuroLoRA (arXiv:2603.12378, 2026)** [CONFIRMADO] — neuromodulação escalar **implementada com benchmark** sobre projeções congeladas.
- **LILITH (arXiv:2507.04575, 2025)** [CONFIRMADO, conceitual] — tokens escalares tipo neuromodulador (broadcast, não atenção densa).

### 1.7 "Precisão = ganho = atenção" — CONFIRMADO como princípio, com análogos implementados

- **kWTA / inibição GABA da Leabra** = competição enviesada (biased competition de Desimone & Duncan) — mesmo papel funcional da nossa precisão como ganho.
- **Goyal et al. (Shared Workspace, ICLR 2022)** [CONFIRMADO verbatim] — gargalo de capacidade limitada "encourages specialization and compositionality and facilitates synchronization". Nosso tálamo-roteador é esse gargalo.

### 1.8 Começar pela VISÃO fiel — CONFIRMADO como a sequência certa

AXIOM (object-centric vision primeiro), RGM (pixels→planning), linhagem VanRullen (Simple Shapes → imagens) e LeCun/V-JEPA **todos começam pela percepção visual** como módulo fundante. Quem fez agente que age começou pela visão.

### 1.9 Gargalo serial deliberado / world model em imaginação latente

- **Common Model of Cognition (Laird, Lebiere & Rosenbloom, AI Magazine 2017)** [CONFIRMADO] formaliza o gargalo serial de decisão sobre processamento paralelo — mesma assimetria do nosso S2 lento sobre S1 rápido.
- **DreamerV3 (Nature 2025)** [CONFIRMADO] — controlador rápido treina rolando-se em imaginação no espaço latente: exatamente o papel do nosso forward-model cerebelar.

---

## 2. O que DESAFIA / nos faz repensar

As divergências mais valiosas. Onde a evidência empurra contra uma decisão nossa.

### 2.1 ⚠️ O MAIOR DESAFIO: congelar o System 2 é contracorrente em robótica — todos co-treinam

Esta é a divergência número um, e a mais cara.

- **Helix** [CONFIRMADO verbatim] retropropaga gradiente de S1 **para dentro** de S2 através do vetor latente. **GR00T** é "tightly coupled, jointly trained end-to-end". **FiS-VLA** compartilha parâmetros. **Nenhum congela o córtex.**
- O ganho de generalização-em-velocidade vem justamente de deixar o gradiente do controlador **moldar** as representações do modelo lento.
- **Resultado negativo direto:** Coda-Forno et al. (arXiv:2510.00494) [CONFIRMADO] — coprocessador S2 comunicando latentes com base S1 **não bate** modelo único de orçamento equivalente; latentes ocupam subespaços redundantes. Em texto puro, a separação dois-sistemas com comunicação latente **não rende sozinha**.

**O que isto significa para o 3a:** a separação rápido/lento só compensa quando há **assimetria física real** (200 Hz de controle vs 9 Hz de visão) forçando especialização. Sem essa pressão, dois módulos viram um só ineficiente.

**Decisão a repensar (não abandonar):** mantemos a LLM congelada, **mas** o ônus da prova é nosso. Precisamos de:
1. Uma **camada adaptadora rica e treinável** (o tálamo-roteador) absorvendo o que o co-treino faria — caso contrário S1 herda representações não-alinhadas à tarefa de controle.
2. Garantir **assimetria física genuína** (sensório-motor a 100+ Hz). Se o nosso ganho depender de elegância arquitetural em vez de pressão física, ele evapora — esse é o aviso do resultado negativo dos LLMs.
3. **Ablação que ninguém fez ainda:** "congelado + adaptador treinável" vs "co-treinado". É a aposta de viabilidade do 3a inteiro e está sem evidência direta a favor **ou** contra.

### 2.2 ⚠️ Não confiar no córtex-LLM como verificador de si mesmo

- **Huang et al. (ICLR 2024)** [CONFIRMADO] — auto-correção **intrínseca** (LLM revendo a própria resposta sem sinal externo) frequentemente **piora** o desempenho.
- **Kambhampati (Position, ICML 2024)** [CONFIRMADO] — LLMs **não planejam** de forma robusta (PlanBench).
- **Reflexion** [CONFIRMADO; o número "~70%" é NÃO-VERIFICÁVEL] — auto-avaliação de sucesso é frágil sem ground-truth.

**Lição forte:** nossa gramática "erro sobe" precisa de um sinal de erro que **não venha do próprio LLM**. Tem que vir do **cerebelo forward-model (predição vs realidade)** ou de um verificador físico/ambiental. É exatamente o que o nosso loop sensório-motor pode fornecer e os agentes-prosa não têm. Adotar o padrão **LLM-Modulo**: nunca deixar o córtex emitir sem um crítico externo.

### 2.3 Geração de tokens pode ser o substrato errado para o world model

- **LeCun (JEPA/V-JEPA 2)** [CONFIRMADO] rejeita explicitamente LLMs autoregressivos como núcleo de world model — prediz embeddings, não pixels/tokens, e argumenta que predição latente generaliza melhor e é mais barata.

**Decisão a repensar:** se nosso loop rápido for um world model **gerativo**, isso colide com a evidência. Considerar um núcleo sensório-motor **JEPA-like (predição latente)** em vez de gerativo para a etapa de visão.

### 2.4 World-model-as-planner é lento — é System 2, não System 1

- **V-JEPA 2-AC** [CONFIRMADO verbatim, Tabela 3] leva **16 s por passo de ação** (MPC+CEM em latente). Isso é deliberação, não reflexo.

**Lição:** não confundir papéis. O cerebelo forward-model precisa ser um **preditor amortizado (uma passada)**, não um otimizador iterativo. World-model-planner estilo Dreamer/JEPA é o "imaginador" lento, não o controlador reativo.

### 2.5 Contrastivo puro não basta para a ponte — precisa de broadcast + cycle-consistency

- **Maytié et al. (2024)** [CONFIRMADO verbatim] — variantes tipo CLIP "did not display the same generalization abilities".
- **Devillers et al. (2024)** [CONFIRMADO] — ablação mostra que **tanto o workspace compartilhado quanto o cycle-consistency são críticos**.

**Lição:** se a ponte RTC for só contrastiva, provavelmente falha. Precisa do laço de reconstrução bidirecional — que, convenientemente, é **exatamente o que predictive coding já dá** (erro de reconstrução sobe). **Ressalva:** validado só em Simple Shapes / RL simples; incerto se escala para visão de alta dimensão.

### 2.6 Aprendizado não pode ser hand-engineered — risco de herdar a fragilidade do Spaun

- **Princípio 5 da Leabra ("Learning is critical")** critica nominalmente a NEF de Eliasmith ("processo puramente de engenharia, não representa aprendizado dirigido por experiência") e o ACT-R ("embutir conhecimento inicial considerável").
- **Spaun** [CONFIRMADO] não aprende conectividade central (pesos por mínimos quadrados); repertório de 10 numerais + poucos símbolos; não aprende tarefa nova.
- **AXIOM venceu jogando fora a generalidade neural** [CONFIRMADO verbatim] — priors object-centric **codificados à mão** (slots, distância de interação τ hand-tuned, matrizes A/B/C). Os autores admitem: "applications of active inference have typically been confined to small-scale tasks with carefully designed priors".

**Lição dupla:**
1. Se nossos módulos cérebro-fiéis tiverem pesos hand-engineered demais, herdamos a fragilidade. Precisa de **aprendizado online genuíno** em pelo menos alguns módulos.
2. A aposta do 3a é que **a LLM congelada fornece os priors gerais que o AXIOM teve que codificar à mão**. Esse é precisamente o trade-off que precisamos provar — não assumir.

### 2.7 Predictive coding NÃO escala como mecanismo de treino profundo

- **Pinchetti et al. (2024)** [CONFIRMADO; correção: usa VGG e Tiny-ImageNet/CelebA, NÃO ResNet18/ImageNet] — PC iguala backprop em CNNs rasas mas trava ao escalar; o porquê permanece **sem explicação**.
- **μPC (NeurIPS 2025)** [CONFIRMADO] — 128 camadas, mas **só em MNIST/Fashion-MNIST**.
- **Crítica em Neural Computation 2023 (Zahid, Guo & Fountas)** [CONFIRMADO] — variantes práticas de PC têm complexidade temporal **pior** que backprop mesmo paralelizadas.

**Lição:** usar PC/energia livre como **gramática conceitual** e para **dinâmica e controle** é seguro. Usá-lo como **mecanismo de treino profundo em escala** ainda não é viável. Isto **reforça** manter a LLM (treinada por backprop) congelada — não tente substituir o córtex inteiro por uma torre de PC do zero.

### 2.8 Estabilidade representacional vs. roteamento dinâmico (Princípio 17 da Leabra)

O significado está no **padrão de atividade da população, não em mensagens individuais**; representações precisam ser **estáveis no tempo**. Isto desafia roteadores que remapeiam módulos "on the fly".

**Lição para o tálamo-roteador:** o roteamento deve mudar **quem fala com quem** e o **ganho**, mas o **vocabulário de cada módulo precisa ser estável** — senão a ponte assíncrona (RTC) fica sem semântica fixa para transportar.

### 2.9 Outros achados que mudam detalhes (não a arquitetura)

- **Cross-attention a camada intermediária bate a final** (GR00T, ~12ª camada) [CONFIRMADO verbatim] — "faster inference speed and higher policy success rate". Conectar S1 ao córtex congelado por camada do meio, não a final. (Camada ótima provavelmente difere num córtex de fronteira de escala diferente.)
- **Congelar o encoder sensorial causa "wash-out"** (literatura frozen-VLM) [CONFIRMADO como fenômeno] — pode impedir recuperação de detalhe pixel-level. Se a visão preditiva for congelada **e** a ponte for estreita, perdemos detalhe fino. Considerar **visão parcialmente plástica** ou ponte mais larga para sinal sensório.
- **PBWM trava sem sistema de progresso/curiosidade** (O'Reilly admite) — o gating "continua a escolher ações subótimas sem fazer progresso". **Nossa amígdala + neuromodulação precisa incluir um sinal de progresso/curiosidade, não só valor**, senão fica preso em ótimos locais. (O "ganho de informação" do AXIOM já é esse escalar — mas o próprio AXIOM admite que ganho de informação **às vezes piora a performance**; não é grátis, precisa regulação.)
- **Ganglios/EFE explodem em horizonte longo** (|A|^h) — nosso S2 lento que planeja em chunks pode contornar, mas só se a ponte for bem-feita; senão herdamos a maldição do horizonte.

---

## 3. O que ADOTAR concretamente

Técnicas, código aberto e achados usáveis já. Com fonte e status.

### 3.1 Para a ponte de duas velocidades (nosso maior risco)

| Adotar | Fonte | Status |
|---|---|---|
| **RTC literal** (freeze-N-ações-garantidas + inpaint-o-resto + partial-attention; orçamento +200–300 ms) | `huggingface.co/docs/lerobot/rtc`; arXiv:2506.07339 | Peer-reviewed NeurIPS 2025 |
| **Bridge buffer** (S2 escreve latentes a 1–3 Hz, S1 lê sem bloquear) + cross-timescale sampling no treino | DuoCore-FS, arXiv:2512.20188 | Preprint, números batem |
| **Cross-attention à camada ~12 (intermediária) do córtex** | GR00T N1, arXiv:2503.14734 | Preprint, modelo aberto |
| **Offset temporal no treino** (treinar S1 com observação defasada Δ em relação a S2) | Helix (Figure) | Anúncio técnico, sem código |
| **Padrão Talker-Reasoner / Fast Talker-Slow Thinker** (belief state em memória compartilhada, override só quando a tarefa exige) | arXiv:2410.08328; VoiceAgentRAG arXiv:2603.02206 (speedup 316×) | Preprint/eng. |

> ⚠️ RTC foi medido em delays de **centenas de ms** (manipulação de mesa), **não nos múltiplos segundos** que uma LLM de fronteira pode impor. Verificar no nosso gap antes de comprometer.

### 3.2 Para a gramática única / aprendizado dos módulos

| Adotar | Fonte | Status |
|---|---|---|
| **XCAL** como regra de "erro = expectativa − resultado" num limiar flutuante de dois tempos | ccnbook.colorado.edu (O'Reilly et al. 2012) | Peer-reviewed |
| **kWTA** como atenção/precisão esparsa barata (alternativa a softmax) | Leabra, Princípio 9/12 | Peer-reviewed |
| **Decomposição da EFE do AXIOM** (utilidade + ganho de informação, Eq. 10) — ganho de informação = um dos 4 escalares (curiosidade/precisão epistêmica) | AXIOM, arXiv:2505.24784 | Preprint |
| **Mapa de Doya** como semântica pronta dos 4 escalares (DA→reward error, 5HT→horizonte, NA→exploração/ganho, ACh→taxa de atualização) | Doya, Neural Networks 2002 | Peer-reviewed |
| **Matriz C de preferências em linguagem natural** — LLM congelada injeta objetivos/priors de alto nível no loop AIF | Bo Wen / IBM, arXiv:2508.05619 | Preprint conceitual |
| **Cycle-consistency + contrastiva + broadcast loss** para a ponte aprendida entre espaços (4–7× menos dados pareados) | Devillers/Maytié/VanRullen, arXiv:2306.15711 / 2403.04588 | Peer-reviewed (IEEE TNNLS / RLC) |

### 3.3 Código aberto reusável já

| Componente do 3a | Asset | Fonte | Status |
|---|---|---|---|
| Visão preditiva object-centric + world model rápido | **AXIOM** (sMM/iMM/tMM/rMM; estudar o Bayesian Model Reduction p/ crescer/podar slots) | github.com/VersesTech/axiom (JAX) | Preprint + código |
| Visão preditiva (encoder+predictor self-supervised, energia goal-conditioned) | **V-JEPA 2** (pesos abertos) | ai.meta.com/research/vjepa | Pesos abertos |
| Loop rápido de baixa latência on-device | **LFM2** (350M–2.6B + MoE 8B-A1B) | Liquid AI, arXiv:2511.23404 | Produto/checkpoints abertos |
| World model / forward-model (robustez symlog+twohot+norm) | **DreamerV3** | github.com/danijar/dreamerv3 | Peer-reviewed (Nature) |
| Backbone VLA + flow-matching p/ System 1 | **π0 / openpi** | github.com/Physical-Intelligence/openpi | Tech report + código |
| Córtex sensório-motor / reference frames (etapa visão, sem LLM) | **Monty** (Thousand Brains) | github.com/thousandbrainsproject/tbp.monty | MIT, patentes non-assert |
| Esqueleto de seleção de ação (tálamo/gânglios como POMDP discreto + EFE) | **pymdp** (JAX-first) | arXiv:2201.03904 | Peer-reviewed (JOSS) |
| Ponte assíncrona reativa (message passing a streams, taxas distintas) | **RxInfer.jl** | arXiv:2112.13251 | Peer-reviewed |
| Módulos plásticos backprop-free (amígdala, ajustes de precisão) | **ngc-learn** (ANGC/ActPC) | NACLab (JAX) | Peer-reviewed |
| Orquestração de módulos-LLM (monitor de conflito, preditor, avaliador, decompositor, orquestrador) | **MAP** | arXiv:2310.00194 | Peer-reviewed (Nature Comms) |
| Banco de testes de integração de todos os módulos cérebro-fiéis | **emergent / leabra** | github.com/emer (Go) | Peer-reviewed |

### 3.4 Para o hipocampo episódico

- **Complementary Learning Systems** (McClelland/McNaughton/O'Reilly 1995; Ketz/Morkonda/O'Reilly, PLoS Comp Biol 2013) [CONFIRMADO] — hipocampo esparso (DG/CA3, pattern separation via inibição alta) para one-shot, córtex lento para semântica; o de 2013 usa error-driven baseado em fases-θ (mais capacidade que Hebbian puro).
- **Função de recuperação dos Generative Agents** [CONFIRMADO, números exatos] como default barato: `score = α_rec·recency + α_imp·importance + α_rel·relevance`, recency = decaimento 0,995/hora, importance = nota 1–10 na codificação, relevance = cosseno de embedding; min-max para [0,1]. **Nossa neuromodulação (NE/ACh) pode modular dinamicamente os α** (eles deixaram fixos em 1). Reflexão por limiar de importância acumulada (Σ importance > 150) = gatilho de consolidação (replay/sono). Código: github.com/joonspk-research/generative_agents. ⚠️ São hiperparâmetros de um sandbox (gpt-3.5-turbo); testar antes de hardcodar.

### 3.5 Para a memória procedural (gânglios / cerebelo)

- **Skill library** (Voyager/Cradle) — comportamentos como código executável recuperável por embedding, composicional. ⚠️ **Exige unlearning/poda + sinal de sucesso EXTERNO** (ver §4, Error Fossilization).
- **PBWM** (PFC-BG-tálamo) como blueprint do roteador de duas velocidades — Go/NoGo gating modulado por dopamina, RPE aprendendo **quando** abrir o gate.
- **Cerebelo forward-model**: Tanaka et al. (Frontiers Sys Neurosci 2020) + configuração tandem forward+inverse (Honda et al., PNAS 2018) [ambos CONFIRMADOS].

### 3.6 Protocolos de validação a adotar

- **Ablação workspace-vs-sem-workspace e cycle-vs-sem-cycle** (Devillers mostra que ambos são necessários).
- **TrueSkill / ranking pareado** para medir contribuição de cada módulo sem ground-truth (metodologia padrão-ouro dos Generative Agents: ablação com d=8,16, p<0,001).
- **Instrumentar desde o dia 1** — o maior "não funcionou" do GWA foi a ausência de avaliação.

---

## 4. Onde travaram — armadilhas a evitar (todas verificadas)

- **Error Fossilization (Voyager):** sucesso auto-declarado contamina a skill library **para sempre**; o erro propaga e amplifica. → Memória procedural precisa de poda/unlearning + sinal de sucesso externo.
- **Auto-correção intrínseca degrada (Huang, ICLR 2024):** não construir o loop "erro sobe" dentro do LLM.
- **FAST tokenizer → inference collapse (0,95 Hz no DuoCore com 81+ tokens):** o canal do S2 tem que ser **curto** (resolveram com RVQ-VAE de ~36 tokens).
- **Temporal ensembling falha a partir de +100 ms:** não usar ensembling ingênuo para suavizar a ponte; usar RTC.
- **Comunicação latente entre dois LLMs não escala (Coda-Forno):** mais canal latente ≠ mais inteligência.
- **PC profundo trava ~VGG/ResNet-pequeno; falha em datasets reais grandes:** não treinar o córtex de PC do zero.
- **RGM evitou benchmark deliberadamente:** quando um método foge da comparação dura, ceticismo. Nós **vamos** benchmarkar.
- **Sakana CTM: 72,47% top-1 ImageNet** — dinâmicas temporais cérebro-fiéis ainda pagam custo de acurácia sozinhas.
- **Hong Jeong (achado negativo útil):** inibição entre módulos **sozinha nunca** lateraliza — só com PFC/working-memory acoplado. Não esperar especialização de conectividade inibitória sem buffer de WM.
- **Parcial-funcionalidade (Leabra):** arquiteturas com módulos muito interdependentes "não funcionam de jeito nenhum num estado parcialmente aprendido (como um chip de CPU pela metade)". → **Construção por etapas é certa SE cada etapa for robusta sozinha.** Evitar acoplamentos que só funcionam com tudo pronto.
- **Hype vs peer-review (Verses):** over-claim destrói credibilidade. Não comunicar resultado antes de benchmark reproduzível.

---

## 5. Quem observar de perto (3–5, datados)

1. **VERSES / AXIOM (Heins, Friston et al.) — arXiv:2505.24784, mai/2025; código aberto, JAX.**
   O mais próximo no eixo "gramática única (energia livre) como agente + módulos especializados + visão primeiro". Provou que a lei única funciona como agente eficiente. **O que vigiar:** se escalam além de jogos object-centric com priors hand-tuned, e se algum dia casam com um LLM. É o nosso espelho no caminho-sem-LLM. Acompanhar o OpenReview (em revisão) e replicar os números no código antes de citar como fato.

2. **VanRullen / Toulouse (Global Latent Workspace) — linhagem 2021–2026, peer-reviewed, com código.**
   O mais maduro no eixo "ponte aprendida entre espaços latentes por cycle-consistency". **Multimodal Dreaming** (arXiv:2502.21142) já adiciona world model ao GW; **Chateau-Laurent & VanRullen** (arXiv:2503.01906) roteia informação por um workspace e bate Transformer em OOD com menos params; **An Attention Mechanism for Robust Multimodal Integration** (arXiv:2602.08597, 2026) ataca exatamente a lacuna precisão/ganho. É a fonte de receita mais valiosa para a nossa ponte.

3. **Physical Intelligence (π0 + RTC) + NVIDIA (GR00T N1) — 2024–2025, RTC peer-reviewed (NeurIPS 2025), código aberto.**
   Donos da evidência mais forte sobre a ponte rápida↔lenta. RTC é literalmente a nossa ponte. **O que vigiar:** Hi Robot (ponte simbólica/linguagem da própria PI) e qualquer extensão de RTC para gaps de latência maiores (nosso caso).

4. **Hong Jeong (Miniature Brain Transformer) — arXiv:2603.07217, mar/2026.**
   Inventário de módulos quase isomórfico ao nosso, com ablação controlada. **O que vigiar:** se adiciona gânglios da base/RL e roteamento explícito, e se algum dia troca o transformer-treinado-do-zero por LLM congelado. É o gêmeo acadêmico mais recente.

5. **Linha "Frozen-LM Conditioning" — NeuroLoRA (arXiv:2603.12378), PaceLLM (arXiv:2506.17310), Brainstacks (arXiv:2604.01152), 2025–2026.**
   A primitiva exata do 3a (LLM congelado + módulos neuromodulados treináveis) sendo construída **com benchmarks**. **O que vigiar de perto:** porque é aqui que alguém pode publicar a síntese antes de nós, e porque NeuroLoRA já demonstra neuromodulação escalar funcionando — o que precisamos integrar e superar, não reinventar.

> Menção: **LeCun / Meta FAIR (JEPA)** — blueprint objective-driven (configurator, perception, world model, cost, **actor**, short-term memory) quase espelha o nosso. Observar como referência conceitual e fonte de V-JEPA 2.

---

## 6. Leitura honesta de posicionamento (sem ego)

### Onde o 3a é genuinamente diferente

- **A síntese fechada não existe ainda.** Ninguém publicou **LLM de fronteira congelada como córtex System 2 + conjunto completo de módulos cérebro-fiéis (incl. gânglios da base RL + cerebelo forward-model + amígdala + hipocampo + tálamo-roteador) + duas velocidades por RTC + 4 escalares + gramática de energia livre**, tudo num só sistema. Cada peça existe e foi validada **isoladamente**; a integração completa é espaço em branco real. [Confirmado pelos cinco vereditos.]
- **A combinação específica "congelado + duas velocidades físicas + gramática de energia livre" é incomum.** Robótica co-treina e não usa energia livre; active inference não congela LLM nem usa duas velocidades físicas; agentes-prosa não têm gramática contínua nem loop rápido físico. Estamos na **interseção** de três comunidades que não conversam.

### Onde somos mais um na multidão

- **Modularidade por região cerebral:** consenso. Leabra, Hong Jeong, MAP, WBAI, CoALA. Não é diferencial.
- **Duas velocidades:** consenso de 2025 (5 labs). Não é diferencial.
- **LLM congelado + módulos treináveis:** paradigma nomeado ("Frozen-LM Conditioning") com benchmarks. **NÃO é mais espaço em branco** — a alegação de ineditismo aqui precisa ser **rebaixada** (PaceLLM, NeuroLoRA, Brainstacks já estão lá).
- **Neuromodulação por 4 escalares:** Doya descreveu há 20+ anos; LILITH e NeuroLoRA implementam. Somos mais um.
- **Gramática de energia livre / predictive coding:** VERSES, Friston, Bengio, LeCun. Família lotada.
- **Começar pela visão:** todos fazem.

### Risco real de sobreposição

- **Risco alto e imediato:** a linha **Frozen-LM Conditioning** (NeuroLoRA etc.) pode chegar primeiro à primitiva "congelado + neuromodulação + working-memory com benchmark". Se nossa proposta de valor for **a primitiva**, já fomos batidos. Nossa defesa só funciona se o valor estiver na **síntese cérebro-completa integrada** + **duas velocidades físicas** + **gramática unificada** — não em nenhuma peça isolada.
- **Risco médio:** VanRullen (ponte aprendida) e VERSES (gramática única como agente) estão maduros e ativos. Eles podem fechar a integração no lado deles (LeCun-blueprint completo; AXIOM + LLM).
- **Risco de execução (o nosso maior):** a ponte congelado↔rápido com adaptador treinável **nunca foi testada diretamente contra co-treino**. Se o adaptador não recuperar o gap do co-treino, o congelamento é peso morto e o design não fecha. Esta é a aposta que sustenta tudo.

### Síntese honesta

O 3a **não é revolucionário em nenhuma peça** — é uma **aposta de integração** num ponto que ninguém ocupou porque exige atravessar três comunidades. O valor é real **se e somente se** (a) a síntese completa render mais que a soma das partes, e (b) congelar + adaptar não custar o ganho do co-treino. Ambas são hipóteses não testadas. Vender como "primeiro a congelar LLM como córtex" é **falso** e vai destruir credibilidade (lição da Verses). Vender como "primeira síntese cérebro-completa integrada com gramática única e duas velocidades físicas" é **defensável** — mas ainda é uma aposta, não um resultado.

---

## 7. Ação recomendada

O que muda (ou não) no plano, em ordem de prioridade.

### Não muda (confirmado externamente — seguir)

1. **Manter o mapa de módulos por região.** Convergência tripla independente o valida.
2. **Manter as duas velocidades e a ponte por RTC.** Consenso de 2025, RTC peer-reviewed.
3. **Manter começar pela visão fiel** — mas como **etapa robusta sozinha** (Leabra: chip pela metade não funciona).
4. **Manter os 4 escalares com a semântica de Doya.**

### Muda / decisão a tomar agora

5. **PRIORIDADE MÁXIMA — desenhar o experimento decisivo: "congelado + tálamo-roteador adaptador treinável" vs "co-treino end-to-end".** É a aposta que sustenta o 3a inteiro e ninguém a testou. Fazer isso cedo, num domínio mínimo (visão→ação simples), antes de investir na arquitetura completa. Se o adaptador não recuperar o gap, repensar o congelamento.
6. **Decidir o substrato da visão: JEPA-like (predição latente) em vez de gerativo.** A evidência (LeCun, V-JEPA 2, Maytié contrastivo-não-basta) empurra forte para latente + cycle-consistency. Adotar V-JEPA 2 como ponto de partida.
7. **Fixar que o sinal de erro vem do cerebelo/sensorium, NUNCA do LLM se auto-julgando** (Huang, Kambhampati). Codificar o padrão LLM-Modulo: córtex nunca emite sem crítico externo.
8. **Adicionar um sinal de progresso/curiosidade à neuromodulação** (não só valor) — senão prende em ótimos locais (PBWM admite isso). Usar o "ganho de informação" da EFE do AXIOM, **com regulação** (ele às vezes piora a performance).
9. **Especificar que o vocabulário de cada módulo é estável; só o roteamento e o ganho mudam** (Princípio 17). O RTC transporta semântica fixa.
10. **Memória procedural com unlearning/poda + sinal de sucesso externo desde o início** (Error Fossilization). Nada de sucesso auto-declarado.
11. **Considerar visão parcialmente plástica ou ponte mais larga** se a visão congelada + ponte estreita causar wash-out de detalhe fino.

### Muda na comunicação/posicionamento

12. **Rebaixar a alegação de ineditismo "LLM congelado como córtex".** Não é espaço em branco; é "Frozen-LM Conditioning". Reposicionar o diferencial na **síntese integrada completa + duas velocidades físicas + gramática unificada**, e sempre como **aposta a provar**, não resultado.
13. **Calibrar a linguagem da gramática:** "mesma família que predictive coding / energia livre", não "exatamente a mesma gramática ancorada em energia livre de Friston" (o mapeamento é análogo, não identidade provada).
14. **Instrumentar e benchmarkar desde o dia 1** (lição GWA + RGM + Verses). Adotar ablação TrueSkill e workspace/cycle.

### Lacunas de cobertura a fechar antes de decisões grandes

15. Ler as fontes primárias ainda não verificadas ao caractere: derivação completa do XCAL↔free-energy; rótulos System 1/2 do LLM-ACTR; equações exatas do GWA. Incorporar **LCB ("Latent Codes as Bridges")** e **Hi Robot** (atacam diretamente o trade-off latente-vs-linguagem na ponte) e a linha **transformer-world-models (TD-MPC2, IRIS/STORM)** como alternativa ao RSSM para o forward-model cerebelar.

---

*Documento baseado em cinco clusters de pesquisa com verificação adversarial. Status por afirmação: [CONFIRMADO] = checado em fonte primária; [NÃO-VERIFICÁVEL] = não corroborável no abstract/fonte acessível; [EXAGERADO] = existe mas com imprecisão sinalizada. Confiabilidade geral dos clusters de origem: ALTA (zero citações inventadas em ~80 referências; correções pontuais de atribuição registradas no texto).*
