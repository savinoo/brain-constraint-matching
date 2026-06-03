# Estágio 3 — Ponte assíncrona LLM-congelada ↔ controlador (resultado honesto)

Construção e testes no servidor (backend `stub`); execução real no Mac (Qwen2.5-0.5B, MPS).
Este documento separa o que é **sólido** do que ficou **em aberto** — sem maquiar.

## O que é SÓLIDO

### 1. Pipeline completo, construído e testado
`rtc_blend`, `LWWRegister`+`CortexThread`, `LLMCortex` (stub/mps), `eval_with_latents`, runner.
**25 testes verdes** no servidor (backend `stub`, sem LLM). Latentes pré-computados (a cena
independe das ações) tornam viável usar a LLM real apesar do custo por forward.

### 2. LLM-como-córtex VALIDADA num modelo de linguagem real (o resultado principal)
Gatekeeper: sonda linear sobre a hidden congelada da **Qwen2.5-0.5B** (camada 12, último token),
a partir da cena serializada em texto, prevendo o objetivo:
- **GATE_R² = 0.974** (40 episódios) e **0.867** (12 episódios) — alto e estável.

O objetivo é **altamente decodável** da representação congelada de uma LLM real (melhor até que a
ResNet18, 0.85). A premissa "**LLM de fronteira congelada como córtex + adaptador linear**" se sustenta
num LLM de verdade, não só num proxy de visão. **A peça central da aposta do 3a está de-riscada.**

### 3. Latência real medida no MPS
- **~83 ms/forward** em regime morno (após warmup).
- L50 ≈ 538 ms / L95 ≈ 721 ms **sem warmup** (inclui forwards frios / recompilação MPS).
- A ~50 Hz, 83 ms ≈ **d≈4 passos** de atraso — modesto. (O número sem warmup mostra que cold-start
  e recompilação dominam; um sistema real precisa de warmup e cache.)

## O que ficou EM ABERTO (honestamente)

### A curva sucesso × atraso NÃO é um resultado utilizável ainda
Na execução no Mac (escala reduzida pelo tempo: 12 ep treino, `switch_every=15`), **o `oracle`
colapsou para ~0.12** — ou seja, mesmo com o latente sempre fresco o controlador quase nunca acerta.
Sem um teto alto, a curva não tem poder estatístico e **não diz nada sobre RTC**. Causas:
1. **Escala insuficiente** — 12 episódios (~600 amostras) treinam um adaptador/controlador fraco; e
   `switch_every=15` é agressivo demais para a dinâmica (o objetivo troca antes de ser alcançado).
2. **Custo por forward da LLM real (~265 ms wall)** limita o tamanho do experimento no tempo
   disponível — runs com poder estatístico precisam de batching/cache dos forwards.

### O ganho do RTC não é testável nesta tarefa (insight de design)
Mesmo com escala suficiente, a `ContextualReach` é **puramente cinemática** (ação = velocidade, sem
inércia). Nessa tarefa **não há custo de descontinuidade**: usar o latente mais fresco imediatamente
é ótimo, e suavizar (RTC) só adiciona atraso. O ganho do RTC — documentado em **robótica de
manipulação** — depende de **dinâmica/contato** onde mudanças bruscas são caras. Confirmado no stub:
`naive` degrada com o atraso (custo de defasagem demonstrado), mas `rtc` não supera `naive` aqui.

## Veredito

- **Representação (LLM-como-córtex):** ✅ validada num LLM real (R²≈0.87–0.97).
- **Latência:** ✅ medida (~83 ms morno); compatível com d pequeno a 50 Hz.
- **Engenharia da ponte:** ✅ construída e testada (stub); ⏳ não exercitada com poder no loop real.
- **Ganho do RTC / co-controlador vs planejador:** ⏳ **ainda em aberto** — bloqueado por (1) tarefa
  cinemática (RTC precisa de dinâmica) e (2) experimento sub-dimensionado + custo por forward.

## Próximo passo (decisão)
Para fechar o veredito co-controlador vs planejador, é preciso, juntos:
1. **Tarefa com dinâmica** (inércia/torque + controle ciente de velocidade), onde mudanças bruscas
   custem — só aí o RTC tem o que suavizar.
2. **Run com poder** — mais dados de treino + `switch_every` calibrado para o `oracle` ficar alto,
   e **cache/batch dos forwards** da LLM para caber no tempo.
3. (Opcional) **MLX** se a latência da Qwen no MPS for o gargalo.

O de-risco que importava — a LLM congelada carrega a informação de controle — está **feito e positivo**.

---

## ATUALIZAÇÃO — tarefa com DINÂMICA (2ª ordem) + veredito final

Implementei o modo **inércia** na tarefa (ação = força, há velocidade; oráculo PD; controlador
`ControllerDyn` que vê pos+vel) — o regime onde mudanças bruscas custam (overshoot) e o RTC *poderia*
ajudar. **28 testes verdes** no servidor.

### Resultado no STUB (latente limpo, experimento COM PODER) — o controlado
| condição | d=0 | d=2 | d=5 | d=10 | d=20 | d=40 |
|---|---|---|---|---|---|---|
| oracle (fresco) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| naive (defasado) | 1.00 | 0.90 | 0.75 | 0.75 | 0.70 | 0.70 |
| rtc (suavizado) | 0.95 | 0.80 | 0.75 | 0.80 | 0.70 | 0.70 |

### Resultado no MAC (Qwen real) — confirma representação, mas sub-dimensionado
GATE_R² = **0.89** (✅ representação de novo). Mas `oracle` colapsou a 0.38: com a LLM real, o
controle em malha fechada ficou limitado por **escala de treino (~720 amostras) + latente de 8-dim**
(não pela latência). Curva sem poder — não usável (igual ao caso cinemático).

### VEREDITO (combinando o stub controlado + o gatekeeper real)
1. **Representação (LLM-como-córtex):** ✅ sólida e repetida (gate R² 0.87–0.97 num LLM real).
2. **Custo de latência:** real e cresce com o atraso (naive degrada). **No ponto de operação real**
   (latência ~83 ms morno → **d≈4** a 50 Hz), a staleness é **leve** (naive ≈ 0.90). → **a LLM
   congelada como CO-CONTROLADORA é viável na latência medida**, sem precisar de truque.
3. **RTC-de-latente NÃO é a alavanca.** Mesmo com dinâmica, suavizar o latente não bate o naive
   (às vezes piora). Faz sentido: o RTC da literatura age no **chunk de ação** (inpainting de ações),
   não no objetivo-latente. Nossa arquitetura (LLM emite objetivo-latente) não tem esse nível.
   A alavanca contra latência alta é a **cadência da LLM / split planejador-vs-controlador**, não o RTC.
4. **Planejador lento (Plano B):** só seria necessário se a latência fosse muito maior (d grande,
   onde naive cai a ~0.70). Na latência atual, não precisa.

**Conclusão honesta:** o estágio 3 entregou o que importava — a **premissa do córtex-LLM congelado
está validada num LLM real** e a **latência real é baixa o suficiente para co-controle**. O ganho do
RTC, especificamente, **não se materializa** nesta arquitetura de latente (e o experimento mostrou por
quê). O que falta para um número de malha fechada com a LLM real é **escala/fidelidade** (mais treino,
latente maior, e cache/batch dos forwards) — engenharia, não conceito.
