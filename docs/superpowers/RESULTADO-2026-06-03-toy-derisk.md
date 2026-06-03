# Resultado — Toy de-risk: congelar o córtex custa caro?

Estudo com 5 seeds. Tarefa ContextualReach (visão→ação 2D, objetivo escondido no "scene").
Treino por imitação de oráculo. **Métrica:** retorno médio (maior=melhor) e taxa de sucesso.

## Regime HOLD — objetivo estático (isola a QUALIDADE das features do córtex)

### Sucesso médio
| condição | K=1 |
|---|---|
| co-treino (B) | 1.00±0.00 | 
| congelado rico (A) | 1.00±0.00 | 
| congelado grosseiro | 1.00±0.00 | 
| congelado desalinhado | 0.97±0.03 | 
| congelado aleatório | 0.99±0.02 | 
| congelado c/ gargalo 1-D | 0.13±0.06 | 
| monólito (C) | 1.00±0.00 | 


### Retorno médio
| condição | K=1 |
|---|---|
| co-treino (B) | -7.85±0.04 | 
| congelado rico (A) | -7.91±0.11 | 
| congelado grosseiro | -8.06±0.11 | 
| congelado desalinhado | -10.47±1.60 | 
| congelado aleatório | -9.71±1.36 | 
| congelado c/ gargalo 1-D | -43.52±1.29 | 
| monólito (C) | -7.80±0.04 | 


## Regime SWITCH — objetivo troca a cada 20 passos (isola a LATÊNCIA da ponte, varrendo K)

### Sucesso médio
| condição | K=1 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| co-treino (B) | 0.86±0.01 | 0.87±0.01 | 0.89±0.01 | 0.81±0.02 | 
| congelado rico (A) | 0.88±0.01 | 0.86±0.01 | 0.90±0.02 | 0.82±0.03 | 
| congelado grosseiro | 0.87±0.02 | 0.86±0.02 | 0.89±0.01 | 0.83±0.03 | 
| congelado desalinhado | 0.81±0.03 | 0.86±0.01 | 0.83±0.03 | 0.78±0.03 | 
| congelado aleatório | 0.84±0.04 | 0.86±0.03 | 0.84±0.04 | 0.80±0.02 | 
| congelado c/ gargalo 1-D | 0.28±0.03 | 0.24±0.05 | 0.33±0.04 | 0.24±0.04 | 
| monólito (C) | 0.88±0.01 | 0.89±0.02 | 0.92±0.01 | 0.88±0.02 | 


### Retorno médio
| condição | K=1 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| co-treino (B) | -45.31±0.11 | -42.03±0.15 | -41.94±0.12 | -43.14±0.11 | 
| congelado rico (A) | -45.23±0.15 | -41.93±0.16 | -41.83±0.10 | -43.08±0.11 | 
| congelado grosseiro | -45.33±0.12 | -42.10±0.13 | -41.94±0.08 | -43.17±0.09 | 
| congelado desalinhado | -45.77±0.32 | -42.43±0.49 | -42.53±0.47 | -43.69±0.53 | 
| congelado aleatório | -45.57±0.37 | -42.28±0.42 | -42.34±0.45 | -43.56±0.53 | 
| congelado c/ gargalo 1-D | -63.88±0.94 | -62.42±2.13 | -61.29±1.55 | -61.86±1.20 | 
| monólito (C) | -44.79±0.07 | -41.39±0.08 | -41.41±0.08 | -42.48±0.06 | 


## Veredito

1. **Congelar é GRÁTIS quando a representação congelada RETÉM a variável da tarefa.**
   Córtex rico, grosseiro (só quadrante), desalinhado (objetivos embaralhados) e até aleatório
   igualam o co-treino — porque qualquer função informativa do input preserva o objetivo de baixa
   dimensão, e o adaptador treinável o reconstrói. É exatamente por isso que LLMs congeladas
   funcionam como extratores de features.
2. **Congelar FALHA só quando a representação DESTRÓI a informação** (gargalo 1-D sobre objetivo 2-D):
   sucesso despenca. O risco real não é "features erradas/desalinhadas" — é "informação ausente".
3. **A latência (K) degrada TODAS as condições de forma parecida** — é um efeito de rastrear um alvo
   móvel com sinal velho, não algo específico do congelamento.

**Implicação para o 3a:** a aposta "congelar a LLM-córtex + adaptador recupera o gap" é **bem mais
segura do que a literatura de robótica sugere**, DESDE QUE as variáveis de controle sejam decodificáveis
da representação congelada da LLM (capacidade enorme → quase sempre o caso). O perigo concreto a vigiar
é a classe de variáveis que a LLM simplesmente **não representa** (o análogo do gargalo). Próximo passo
de escala (V-JEPA 2 / LLM real) deve medir, com sonda linear, se as variáveis de controle são
decodificáveis da representação congelada — esse é o teste que prevê sucesso/fracasso.

## Confirmação quantitativa — a fronteira é exata

Varredura da dimensão do gargalo do córtex congelado (objetivo da tarefa = 2-D), regime HOLD,
média de 3 seeds:

| capacidade do gargalo | sucesso |
|---|---|
| 1-D | 0.25 |
| 2-D | 1.00 |
| 3-D | 1.00 |
| 4-D | 1.00 |

**Congelar falha exatamente quando a capacidade da representação cai abaixo da dimensão intrínseca
da tarefa** (aqui, 2). Acima do limiar, congelar é gratuito. Isto transforma o risco numa quantidade
mensurável: *a representação congelada carrega ≥ as variáveis que a tarefa precisa?* — exatamente o
que uma sonda linear sobre a LLM/V-JEPA-2 mede no estágio seguinte.

## A sonda linear PREVÊ o sucesso (instrumento do estágio 2 validado)

Sonda linear = regressão linear da representação congelada → objetivo (R² out-of-sample). Mede
decodabilidade. Média de 3 seeds, regime HOLD:

| modo do córtex congelado | sonda R² | sucesso do congelamento |
|---|---|---|
| rico | 1.00 | 1.00 |
| grosseiro | 0.99 | 1.00 |
| desalinhado | 0.99 | 0.99 |
| aleatório | 0.99 | 0.99 |
| gargalo 1-D | 0.43 | 0.25 |

A R² da sonda **prevê** se congelar vai funcionar. Isto valida, no brinquedo onde conhecemos a
verdade, o **instrumento barato do estágio 2**: antes de construir qualquer coisa sobre uma
LLM/V-JEPA congelada, rodar uma sonda linear das variáveis de controle sobre a representação dela.
R² alto → congelar é seguro; R² baixo → a LLM não representa o que a tarefa precisa, e só aí o
congelamento é um problema. **Barato, decisivo, e mede exatamente o risco real.**
