# Estágio 2 — Sonda linear num modelo de visão REAL congelado

Rodado no Mac do Lucas (Apple Silicon, MPS). Modelo congelado: **ResNet18 pré-treinada** (ImageNet).
Tarefa: imagens sintéticas de um quadrado com **posição (x,y), tamanho e cor (R,G,B)** conhecidos.
Métrica: **R² out-of-sample** de decodificar cada variável das features congeladas via regressão linear.

## Resultados

### Frozen pré-treinado (rico)
| variável | x | y | tamanho | R | G | B | **médio** |
|---|---|---|---|---|---|---|---|
| R² | 0.95 | 0.96 | 0.95 | 0.73 | 0.76 | 0.76 | **0.85** |

### Frozen aleatório (rede não-treinada)
| variável | x | y | tamanho | R | G | B | **médio** |
|---|---|---|---|---|---|---|---|
| R² | −0.66 | −1.41 | 0.37 | −0.37 | −0.36 | −0.20 | **−0.44** |

### Curva de capacidade (features pré-treinadas projetadas para d dims)
| d | 1 | 2 | 4 | 8 | 16 | 32 | full (4608) |
|---|---|---|---|---|---|---|---|
| R² médio | 0.04 | 0.07 | 0.16 | 0.28 | 0.45 | 0.64 | 0.85 |

## Leitura

1. **CONFIRMA a aposta num modelo real.** A representação congelada de um modelo **pré-treinado**
   carrega as variáveis de controle — posição quase perfeita (R²≈0.95), tamanho idem, cor decente.
   Congelar + adaptador linear é viável: a informação está lá, decodável linearmente.

2. **CORRIGE o otimismo do brinquedo (nuance importante).** No brinquedo raso, até um córtex
   **aleatório** funcionava. Aqui, numa rede **profunda** aleatória (não pré-treinada), as variáveis
   **não** são linearmente decodáveis (R² negativo — pior que prever a média). No regime real/profundo,
   **a qualidade do pré-treino importa.** Tradução honesta: congelar é seguro **se** o modelo congelado
   for um modelo pré-treinado de verdade cuja representação codifica as variáveis — não "qualquer rede
   congelada". Uma LLM/foundation model de fronteira qualifica; uma rede ao acaso, não.

3. **CONFIRMA a lei de capacidade.** R² sobe monotônico com o número de dimensões disponíveis —
   mesma fronteira de "informação presente" vista no brinquedo, agora num modelo real.

4. **A SONDA LINEAR é o instrumento certo.** Ela disse, antes de construir nada: pré-treinado = bom,
   aleatório = ruim. É o pré-qualificador barato do estágio seguinte.

## Implicação para o 3a

Usar a representação de uma **LLM/foundation model real** (não aleatória) como córtex congelado é
**defensável** — as variáveis de controle tendem a estar decodificáveis nela. E ganhamos um protocolo:
**antes de investir em qualquer tarefa nova, rodar a sonda linear** das variáveis de controle sobre a
representação congelada. R² alto → seguir; R² baixo → ou a tarefa precisa de variáveis que o modelo não
representa, ou falta capacidade/treino — e aí ajustar antes de construir.

*Ambiente: ResNet18 é um modelo pequeno; o próximo refino seria repetir com um modelo mais forte
(DINOv2 / V-JEPA 2) e com variáveis de controle mais realistas, mas o mecanismo já está demonstrado.*
