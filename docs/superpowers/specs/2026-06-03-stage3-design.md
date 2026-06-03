I now have the complete picture. Here is the recommended design.

---

# Estágio 3 — DESIGN RECOMENDADO: a ponte assíncrona LLM-lenta ↔ controlador-rápido

> Documento pronto para virar plano de implementação. Funde os três candidatos: a espinha minimalista do **Candidato 1**, as três camadas defensivas + portabilidade servidor↔Mac do **Candidato 2**, e o pré-qualificador integrado + a honestidade sobre neuromodulação do **Candidato 3**. Trade-offs marcados onde existem.

---

## 1. Resumo executivo

O estágio 3 fecha o **menor loop de duas velocidades honesto**: uma LLM pequena REAL e congelada (Qwen2.5-0.5B) atua como córtex lento (System 2, ~2–7 Hz) rodando em **thread de fundo**, emitindo um objetivo-latente contínuo único que um `Controller` rápido (System 1, 50–100 Hz) consome **todo passo** lendo a versão mais recente disponível — possivelmente velha — por um buffer last-write-wins, com **blend exponencial estilo RTC** na troca de latente e **fallback zero-order-hold** no warmup. O risco nº1 — já de-riscado para *representação* (toy 5 seeds + ResNet18 R²~0.95), mas **não** para *engenharia temporal* — é a latência wall-clock real da LLM disputando a GPU integrada com o controlador. O entregável científico é uma única curva, **sucesso × latência**, sob três condições (`oracle_sync` / `naive_async` / `rtc_blend`) varrendo o atraso `d` em passos, que decide o design 3a: se RTC segura o teto até `d ~ H/2` → a LLM é **co-controlador** viável; se quebra em `d` pequeno → a LLM precisa ser rebaixada a **planejador lento** (sub-objetivos esparsos), bifurcação já prevista em `RESOLUCOES_3a.md` (Buraco 4). Reuso quase total do repo: `ContextualReach`, `Controller`, `Adapter`, `linear_probe_r2`, `train_freeze_adapt`, e o padrão de carga congelada de `stage2_probe.py`.

---

## 2. Escolhas concretas travadas

### 2.1 LLM-córtex
- **Modelo:** `Qwen/Qwen2.5-0.5B` — Apache 2.0, **não-gated** (zero atrito de download), 24 camadas, hidden **896**, contexto 32k, arquitetura padrão (RoPE/SwiGLU/RMSNorm/GQA/tied embeddings) 100% suportada em `transformers`. fp16 ≈ 1 GB; cabe folgado nos ~29 GB.
- **Secundário (robustez/latência maior):** `Qwen/Qwen2.5-1.5B` (28 camadas, hidden 1536, ~3.1 GB) — troca de **uma string**, para subir a latência e estressar `d` maior.
- **Descartados** (verificado): Llama-3.2-1B/3B (gated + licença custom), Gemma-2-2B (gated + soft-capping de logits → exige `attn_implementation="eager"`, risco numérico em MPS). SmolLM2-1.7B é alternativa Apache neutra, **mas** treinado em contexto 2048 — pior se a `scene` serializada ficar longa.
- **Backend — decisão travada com trade-off explícito:** carregar via `transformers` (stack do repo). **Não geramos texto** — fazemos **um único forward de prefill com `output_hidden_states=True`, nunca `.generate()`**. O gargalo medido do PyTorch/MPS (decode ~7–9 tok/s, ~25–30× mais lento que MLX — arXiv:2511.05502, Mac Studio M2 Ultra) é de *decode autoregressivo* e **não se aplica** ao nosso caso; a métrica relevante é TTFT/prefill. A lentidão do MPS é, aqui, *desejável* (gera a latência honesta de córtex). **Caveat de honestidade:** se a latência real medida no Mac-alvo ficar inviável (ex. >2 s de forma instável), a ponte é **agnóstica ao backend** — trocar a LLM-córtex por MLX é um plano de contingência (Seção 6), não a escolha default.

Carga congelada (espelha `stage2_probe.py`):
```python
NAME = "Qwen/Qwen2.5-0.5B"
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    NAME, torch_dtype=torch.float16, output_hidden_states=True,
).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)            # CONGELADO
```
Notas: fp16 default; **fallback fp32** se aparecer NaN em MPS (RAM sobra). **bf16 em MPS é frágil — evitar.** Cap de 4 GB/tensor do MPS **não dispara** (0.5–1.5B, prompt curto < 2k tokens) — por isso **não** escalar para 3B nem prompts longos no MVP.

### 2.2 Camada / pooling do latente (ablação, não dogma)
Precedente GR00T N1 (NVIDIA 2025): *"using middle-layer instead of final-layer LLM embeddings resulted in both faster inference speed and higher downstream policy success rate ... we use the representations from the 12th layer"*. Logo:
- **Camada inicial:** `L = n_layers // 2` (= **12** no 0.5B, 14 no 1.5B). Confirmar com sweep da sonda — vencedora é empírica.
- **Pooling inicial:** **último token** (decoder-only causal vê todo o contexto). **Tratar mean-pool como ablação genuína** — a literatura de embeddings NÃO garante último-token > mean (recency bias; MTEB frequentemente favorece mean). A sonda decide por R².

```python
out = model(**enc)                         # forward único (prefill), SEM .generate()
L = len(out.hidden_states) // 2            # camada do meio
h = out.hidden_states[L]                   # (1, T, 896)
idx = enc["attention_mask"].sum(1) - 1     # índice do último token real
h = h[0, idx]                              # (896,)
h = layernorm(h)                          # estabilização (ver 2.3)
```

### 2.3 Adaptador (tálamo-roteador) — receita LLaVA-1.5 + anti-anisotropia
```
LayerNorm(H) → Linear(H, 64) → GELU → Linear(64, Z_DIM=8) → L2-normalize
```
- **LayerNorm(H) antes do MLP — obrigatório.** Hidden states de LLM são dominados por poucas *outlier dimensions* que desestabilizam o treino do adaptador (LLM.int8, Dettmers 2022: ~150k outliers em 6 dims **num 6.7B**). **Calibração honesta:** num 0.5B o efeito é **mais brando** que num 6.7B — normalizar continua barato e correto, mas não é o desastre de um modelo grande.
- **L2-normalize no `z'` de saída** — limita o quão fora-de-distribuição um latente velho fica quando a ponte o transporta defasado (a favor direto da robustez à latência).
- É **literalmente** o `Adapter` atual (`models.py`) com a entrada trocada de `Z_DIM=8` para `H=896`, mais LayerNorm/L2 nas pontas. `Controller` **inalterado** (`(pos, z') → ação`).

### 2.4 Controlador (System 1)
`Controller` do repo, sem mudança de forma: `Linear(POS_DIM + Z_DIM, 64) → ReLU → Linear(64, 2)`. Roda em torch/MPS; tensores minúsculos, **cap de 4 GB não os afeta**.

### 2.5 Tarefa
**`ContextualReach` estendida — NÃO trocar por pêndulo/reacher** (adicionaria dinâmica instável e tuning de controle sem testar a *ponte*). O env já é o estressor certo: `step()` **troca o objetivo no meio do episódio** a cada `switch_every` passos (`contextual_reach.py:47-48`) — quando o objetivo muda, o `z` velho fica errado, exatamente onde a tolerância à latência é testada.
- **Serialização da `scene` (16-D) → texto curto, estruturado:** quantizar em poucos tokens, ex. `"scene: 12 -3 45 ... ; pos: 30 -20"` (inteiros = poucos tokens, prefill barato). É a opção mais simples e honesta para LLM real; embeddings contínuos à la Frozen exigem projetor treinado → **estágio 4**. **Risco gatekeepado pela sonda** (Seção 4).
- **Calibração do estressor:** medir latência real `L` (ms) → `d = L·fps` passos → setar `switch_every ≈ 1–3 × d` para que o controlador passe frações significativas do tempo sobre objetivo obsoleto. (`switch_every=20` atual pode ser fácil demais; calibrar para `naive_async` realmente quebrar, senão o experimento não tem poder.)

### 2.6 Ponte assíncrona (mecânica exata)
**Single-process, duas threads** (sem gRPC/ZeroMQ/Redis — overkill). Invariante: **o loop rápido NUNCA bloqueia esperando o córtex.**

**Buffer = `LWWRegister`** (last-write-wins, 1 slot, `threading.Lock`; ordem de chegada não importa — semilattice, padrão DRTC). Carrega `{z_novo, z_antigo, t_chegada}`:
```python
class LWWRegister:
    def __init__(self):
        self._lock = threading.Lock()
        self._z = self._z_prev = self._t_arrival = None
    def publish(self, z, ctrl_step):
        with self._lock:
            self._z_prev = self._z if self._z is not None else z
            self._z, self._t_arrival = z, ctrl_step
    def read(self, ctrl_step):
        with self._lock:
            if self._z is None: return None
            return self._z, self._z_prev, ctrl_step - self._t_arrival
```

**Thread de fundo (córtex):** `while running:` pega a `scene` atual → `cortex_llm(scene)` (`no_grad`, 1 forward) → `Adapter` → `z'_novo`; **mede `Δt_inf` wall-clock**; `publish`. Disparo *gated* por **cooldown** = `⌈Δt_inf·fps⌉ + ε` passos (não re-disparar antes da inferência anterior terminar; evita backlog).

**Loop rápido (main):** todo passo → `read(step)` → se `None`, **ZOH** do último `z'` válido (ou `z=0` no warmup) → **blend RTC** → `Controller(pos, z_efetivo)` → `env.step`.

**Blend RTC no espaço latente — adaptação declarada, NÃO RTC literal.** O paper RTC (Black, Galliker & Levine, *"Real-Time Execution of Action Chunking Flow Policies"*, NeurIPS 2025, arXiv:2506.07339) faz inpainting com pseudoinverse guidance sobre chunks de **ações** de flow-matching. Aqui o córtex emite um **latente-objetivo**, não chunk de ações — então aplicamos a *mesma intuição* (congelar o prefixo já comprometido + transição suave do resto) como blend exponencial em `z`. **Sem pseudoinverse guidance** (só importaria se o córtex gerasse trajetórias). Soft-mask exponencial direto do paper:
```python
def rtc_weight(i, d, L):          # i = passos desde a chegada de z_novo
    if i < d: return 1.0          # prefixo congelado: usa z_antigo ("freeze the prefix")
    c = (L - i) / max(L - d + 1, 1)
    return c * (math.exp(c) - 1) / (math.e - 1)
z_efetivo = (1 - w) * z_novo + w * z_antigo
```
Parâmetros iniciais (regime do paper, reescalado): `fps=50` (dt do env=0.1 → reescalar passo de controle p/ ~20 ms como no paper), `H≈50`, `L_blend=execution_horizon=10`, `d`=atraso medido. **Restrição matemática herdada do paper:** `d ≤ s ≤ H−d`, logo **`d < H/2`** — fronteira que o estágio cruza de propósito.

**Threading, não multiprocessing (no MVP), com fallback explícito.** O forward PyTorch/MPS *geralmente* libera o GIL durante kernels, mas **não há garantia** para MPS (e há exceções conhecidas, ex. `torch.compile` segura o GIL — pytorch#163061). É o **risco de engenharia nº1** → medir `fps` com/sem thread; fallback documentado: `multiprocessing.Process` + `Queue` (serializar vetor 896-D é trivial).

**Neuromodulação (pitada de NE) — OPCIONAL, marcada como extrapolação.** D4 do RESOLUCOES não tem âncora citada. Como o córtex é usado em **extração (prefill), não sampling**, "NE → temperatura de sampling" **não se aplica literalmente**. Tradução honesta ao stack: NE alto (alta surpresa = `|Δreward|`) → encurta a `t_validade` do latente e afia o pooling (`softmax(logits_pool/τ)` com τ caindo). **Fora do MVP mínimo**; só incluir se o loop básico fechar e sobrar apetite. Manter como pitada honesta, não mecanismo validado.

---

## 3. Componentes / arquivos a construir (reusando o repo)

Três novos, três estendidos, **nada reescrito**.

| Arquivo | Ação | O que faz |
|---|---|---|
| `brain/cortex_llm.py` | **novo** | `LLMCortex`: carrega Qwen2.5-0.5B congelado (MPS fp16→fallback fp32), `scene_to_prompt(scene)` (quantiza 16-D→texto), forward único → `hidden_states[L]` → pooling → `h`; mede e expõe `lat_ms`. Flag `backend={"mps","mlx","stub"}` (portabilidade). Padrão de carga de `stage2_probe.py`. |
| `brain/latent_bus.py` | **novo** | `LWWRegister` + `CortexThread` (produtor de fundo com cooldown). Coração da assincronia. |
| `brain/bridge_async.py` | **novo** | `eval_policy_async(...)`: produtor/consumidor com thread real; substitui o `if step % K == 0` síncrono de `bridge.py` por leitura do LWWRegister; inclui blend RTC + ZOH + as 3 condições (`oracle`/`naive`/`rtc`). Generaliza `eval_policy`. |
| `stage3/stage3_loop.py` | **novo (runner)** | Orquestra: mede latência (isolada + sob carga), pré-qualifica com sonda, treina `Adapter`+`Controller` com latente defasado, varre atraso, gera JSON+MD no padrão de `study.py`. |
| `brain/models.py` | **estender** | `H_DIM=896`; `AdapterLLM` (= `Adapter` com entrada `H_DIM`, + `LayerNorm(H)` na entrada e L2-norm na saída). `Controller`/`Cortex`/`Monolith` inalterados. |
| `brain/train.py` | **estender** | `train_freeze_adapt` ganha `delay_steps`: serve ao controlador um `z'` defasado de até `d` passos durante o treino (truque do Helix — robustez por construção). |
| `brain/probe.py` | **estender** | `sweep_layer_pooling(cortex_llm, data)`: varre `L ∈ {n//4, n//2, 3n//4, n-1} × {last_token, mean}`, devolve melhor R². Reusa `linear_probe_r2`. |
| `requirements.txt` | **editar** | adicionar `transformers`. (`mlx-lm` só no Mac, fora do requirements testável no servidor.) |

**Portabilidade servidor↔Mac (do Candidato 2 — vale a pena):** `cortex_llm.py` com 3 backends. `stub` (córtex MLP instantâneo + `sleep` calibrado) roda testes/CI no servidor CPU **sem baixar pesos** e varre atraso injetado *além* do que a LLM entrega; `mps`/`mlx` rodam a LLM real no Mac. O experimento honesto exige `mps`/`mlx`; `stub` é só CI e sweep de injeção.

**Reusar sem tocar:** `brain/envs/contextual_reach.py`, `brain/data.py` (`collect_oracle_dataset`), `stage2/stage2_probe.py` (template de carga congelada).

---

## 4. O experimento e as métricas

**Princípio anti-"fake asynchrony":** a latência tem de ser **wall-clock de LLM real disputando recursos**. `sleep()` só é legítimo (a) no backend `stub` de CI e (b) **somado** à latência real, no sweep de injeção — nunca substituindo a LLM no caminho principal.

**Passo 0 — Gatekeeper (sonda, antes de gastar compute).** `sweep_layer_pooling` sobre o `h` da LLM real prevendo `goal` da `scene`. Exigir **R²(goal) > limiar** (alvo ≥ 0.7, por analogia ao 0.95 da ResNet18). Se baixo → trocar camada/pooling/encoding/modelo **antes** de fechar o loop. Garante que falha posterior seja atribuível à **latência**, não à perda de info na tokenização. Reuso direto de `linear_probe_r2`.

**Métrica nº0 (bloqueante) — LATÊNCIA-SOB-CARGA.** `L50`/`L95` do forward da LLM **isolada** E **sob carga** do FastLoop rodando (contenção da GPU integrada piora a latência — é o número honesto). Reportar ambos. *Nenhum número herdado é confiável: o ~1015 ms do Qwen-0.5B que circula é **AMD Ryzen/x86, não M3 Ultra**, e era saída longa; os tok/s de MPS são **M2 Ultra 192 GB**, não Mac base — só o ranking relativo transfere.* Medir no Mac-alvo é o passo 1 real.

**Métrica nº0b — `fps` efetivo** do FastLoop com/sem CortexThread ativa (valida o paralelismo via GIL; se cai → multiprocessing).

**Experimento principal — degradação sucesso/retorno × atraso, 3 condições:**

| Condição | `z'` efetivo | Papel |
|---|---|---|
| `oracle_sync` | sempre fresco (`d=0`, equivale a `K=1`) | **teto** |
| `naive_async` | último `z'` cru, troca abrupta (zero-order hold) | **baseline** que deve oscilar/cair |
| `rtc_blend` | soft-mask exponencial + freeze-`d` | testa a recuperação |

Note que `naive_async` ≈ o `bridge.eval_policy` atual com K alto; `oracle_sync` ≈ K=1. O delta científico do estágio 3 é o `rtc_blend` e a latência **wall-clock** no lugar de K.

**Sweep de atraso** `d ∈ {0, 2, 5, 10, 15, 20, 30, 50}` passos, gerado de **duas formas reportadas separadamente**: (i) **real** = TTFT medido; (ii) **injetado** = `+100/+200/+400/+800 ms` (stub) para varrer além do hardware e achar o ponto de quebra. Sempre marcar qual ponto é medido e qual é injetado.

**Baselines do enunciado, explícitos:**
- **síncrono-bloqueante** = loop que *espera* o córtex a cada recomputação (o FastLoop trava `Δt_inf` por chamada) → mostra o colapso de `fps` que justifica a assincronia.
- **assíncrono-RTC** = a ponte recomendada (`rtc_blend`).
- **assíncrono-ingênuo** (`naive_async`) = o meio-termo que isola o ganho do blend.

**Métricas primárias** (já em `eval_policy`): **taxa de sucesso** (`-r ≤ success_dist` em algum passo) e **retorno médio**; mais o **throughput** do paper RTC (*proporção da tarefa completada / duração*).

**Regime de leitura (do `study.py`):** rodar **HOLD** (`switch_every` grande → isola qualidade das features, deve ser insensível ao atraso) e **SWITCH** (objetivo troca na escala da latência → isola a ponte). A diferença HOLD−SWITCH é o efeito puro da latência.

**Critério de vitória do RTC:** `rtc_blend` mantém **≥90% de `oracle_sync`** num regime de atraso onde `naive_async` cai **abaixo de ~50%**.

**Entregável científico:** a **curva sucesso × atraso** para as 3 condições. Decide a bifurcação do design (Seção 6).

---

## 5. Ordem de construção por etapas (cada uma robusta sozinha)

Cada etapa entrega algo verificável e não quebra o que veio antes.

1. **`AdapterLLM` + plumbing no servidor (CPU/stub), sem LLM.** Estender `models.py` (`H_DIM`, `AdapterLLM` com LayerNorm/L2). Backend `stub` em `cortex_llm.py` (MLP instantâneo emulando hidden 896 + `sleep` calibrado). **Robusto sozinho:** `train_freeze_adapt` e `eval_policy` rodam com o adaptador novo; testes verdes no servidor. *Sem dependência de Mac/pesos.*
2. **Ponte assíncrona com córtex `stub`.** Construir `latent_bus.py` (`LWWRegister`, `CortexThread`) e `bridge_async.py` (ZOH + blend RTC + 3 condições). **Robusto sozinho:** valida produtor/consumidor, blend e as 3 condições com latência *injetada* via `sleep`, no servidor. Teste-âncora: `oracle_sync ≥ naive_async`, e `rtc_blend ≥ naive_async` em atraso alto.
3. **`delay_steps` no treino.** Estender `train_freeze_adapt` para servir `z'` defasado. **Robusto sozinho:** comparar controlador treinado com vs sem defasagem na ponte stub — espera-se robustez maior a `d>0` com o truque do Helix.
4. **LLM real no Mac — extração + gatekeeper.** Backend `mps` em `cortex_llm.py` (carga Qwen congelada, `scene_to_prompt`, forward+pooling). Rodar `sweep_layer_pooling`. **Robusto sozinho:** produz R² por camada×pooling; só prossegue se passar o limiar. *Primeiro contato com o Mac; nada do loop depende disso ter rodado antes.*
5. **Medição de latência-sob-carga + `fps`.** No Mac: `L50/L95` isolado e sob carga; `fps` com/sem thread. **Robusto sozinho:** números que fixam `d`, `switch_every` e decidem threading vs multiprocessing. *Métrica nº0 bloqueante — gate para a etapa 6.*
6. **Loop fechado real + curva sucesso×latência.** Plugar `LLMCortex(backend="mps")` em `bridge_async.py`; treinar (etapa 3) com `d` da etapa 5; rodar `stage3_loop.py` (3 condições + sweep real e injetado). **Robusto sozinho:** o entregável — JSON+MD com a curva e o veredito.
7. **(Opcional) Qwen-1.5B + pitada de NE.** Subir latência (1.5B) para estressar `d` maior; NE→τ/janela só se o loop básico fechar. *Aditivo, não no caminho crítico.*

---

## 6. Riscos duros e plano B

- **Transferência do regime RTC (76–97 ms, π₀.₅) → regime LLM (200–1000+ ms) pode dar NEGATIVO.** É *feature*, não bug — o estágio existe para descobrir isso. A matemática exige `d < H/2`; a 50 Hz, 1 s = `d=50`, que **estoura H=50**. (`RESOLUCOES_3a.md` linha 212 já nomeia este como o MAIOR RISCO REAL.)
  - **PLANO B — LLM vira planejador lento.** Se RTC quebra em `d` pequeno, a LLM não é co-controlador contínuo: rebaixá-la a **planejador** que emite **sub-objetivos esparsos** (um `z` por "macro-passo", o controlador interpola/persegue localmente entre emissões), não steering contínuo. Reduz a frequência exigida do córtex e dilui a latência — bifurcação já prevista no Buraco 4. **Operacionalização concreta:** aumentar `switch_every` para muito acima de `d`, reduzir a taxa-alvo do córtex, e medir se o sucesso volta. A curva da Seção 4 *diz qual ramo tomar* — o estágio não falha; ele decide.
- **LLM lenta demais / instável em transformers+MPS.** Plano B de backend: a ponte é **agnóstica** — `cortex_llm.py backend="mlx"` (Qwen 4-bit) é mais rápido no decode (irrelevante p/ prefill) mas é o caminho recomendado pelo benchmark para Apple Silicon. Custo: dependência extra (`mlx-lm`) e runtime fora do torch. Só acionar se a métrica nº0 mostrar latência inviável.
- **GIL não libera → `fps` despenca.** Mitigação obrigatória: medir (etapa 5); fallback `multiprocessing.Process` + `Queue`.
- **Tokenização da `scene` destrói info → tarefa falha pelo motivo errado.** A sonda (Passo 0) pré-qualifica e isola "info destruída" de "latência". Se R² baixo: refinar quantização, trocar camada/pooling, ou (estágio 4) embeddings contínuos.
- **`switch_every`/`success_dist` fáceis demais → `naive` não quebra, experimento sem poder.** Calibrar para `naive_async` realmente cair em atraso alto (Seção 2.5).
- **fp16 em MPS com NaN** → fallback fp32 (RAM sobra). bf16 frágil — evitar.
- **Adaptação ≠ RTC literal** (sem pseudoinverse guidance) — declarar no relatório; defensável, mas não é o método do paper.
- **Não testa robô físico** (intencional): experimento de *tarefa*, não de hardware; dinâmica de contato fica fora.

---

## 7. Critérios de aceite

**Engenharia (a ponte funciona):**
- [ ] A LLM-córtex carrega congelada em MPS e produz um forward de prefill sem NaN (fp16 ou fallback fp32).
- [ ] `LWWRegister` + `CortexThread` rodam single-process; o FastLoop **nunca bloqueia** (verificado: `fps` do FastLoop com thread ativa ≥ ~90% do `fps` sem thread, OU fallback multiprocessing acionado e documentado).
- [ ] Latência-sob-carga `L50/L95` medida no Mac-alvo, isolada e sob carga, **reportada** (não herdada). `d` e `switch_every` derivados dela.
- [ ] Suíte de testes (servidor, backend `stub`) verde: blend RTC correto (`w=1` para `i<d`, decai exp depois), ZOH no warmup, 3 condições distintas.

**Ciência (a ponte aguenta a latência — ou diz que não):**
- [ ] **Gatekeeper passou:** R²(goal) da sonda sobre o `h` da LLM **acima do limiar** (alvo ≥ 0.7), com camada×pooling escolhidos por sweep. Se não passou, encoding/camada/modelo ajustados **antes** do loop.
- [ ] Curva **sucesso × atraso** produzida para `oracle_sync` / `naive_async` / `rtc_blend`, com pontos **medidos** e **injetados** marcados separadamente, **+ baseline síncrono-bloqueante** mostrando o colapso de `fps`.
- [ ] **Veredito explícito e fundamentado** numa das duas saídas:
  - **(A) co-controlador viável:** `rtc_blend` mantém ≥90% de `oracle_sync` num regime onde `naive_async` cai abaixo de ~50%, e degrada graciosamente até `d ~ H/2`; **ou**
  - **(B) planejador lento:** `rtc_blend` quebra em `d` pequeno → acionar Plano B e medir se o ramo planejador recupera o sucesso.
- [ ] Relatório (JSON + MD, padrão `study.py`) com: números de latência, curva, veredito, e **declaração honesta** de que o blend é *adaptação* de RTC (não RTC literal) e de quais latências foram injetadas vs medidas.

---

### Apêndice — verificação dos ativos do repo (todos confirmados por leitura direta)
- `bridge.py:26-27` — `eval_policy` recomputa `cortex_fn(obs["scene"])` só a cada `K` passos; `policy_fn` roda todo passo com `z` velho. ✔ ponto de integração exato.
- `models.py` — `Cortex` (scene→z), `Adapter` (z→z', "tálamo-roteador"), `Controller` ((pos,z)→ação, "System 1"); `Z_DIM=8`. ✔
- `probe.py` — `linear_probe_r2(frozen_cortex, data)` (R² out-of-sample do goal). ✔ gatekeeper.
- `train.py:47` — `train_freeze_adapt(data, frozen_cortex, ...)` congela córtex, treina só adapter+controller (`no_grad` no córtex). ✔ ponto p/ `delay_steps`.
- `study.py` — decomposição HOLD (`switch_every=200`) vs SWITCH (`switch_every=20`, sweep K), agregação multi-seed, relatório MD/JSON. ✔ template.
- `envs/contextual_reach.py:47-48` — **goal troca mid-episode** a cada `switch_every` (não por-reset). ✔ estressor já correto. **Correção aos candidatos:** não é preciso "adicionar troca de alvo", só calibrar `switch_every`.
- `stage2/stage2_probe.py` — ResNet18 congelada, MPS, `requires_grad_(False)`, extração em `no_grad`. ✔ template de carga congelada.
- `requirements.txt` — só `torch/numpy/pytest`; `transformers` é dependência nova. ✔

**Arquivos a criar:** `/home/savino/projects/brain/brain/cortex_llm.py`, `/home/savino/projects/brain/brain/latent_bus.py`, `/home/savino/projects/brain/brain/bridge_async.py`, `/home/savino/projects/brain/stage3/stage3_loop.py`.
**A estender:** `/home/savino/projects/brain/brain/models.py`, `/home/savino/projects/brain/brain/train.py`, `/home/savino/projects/brain/brain/probe.py`, `/home/savino/projects/brain/requirements.txt`.
**A reusar sem tocar:** `/home/savino/projects/brain/brain/envs/contextual_reach.py`, `/home/savino/projects/brain/brain/data.py`, `/home/savino/projects/brain/stage2/stage2_probe.py`.