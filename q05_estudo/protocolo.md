# Protocolo anterior aos experimentos — Q5

Registrado em 13/10/2026 (America/Recife), antes da execução da referência e das variantes.

## Pergunta e hipótese

Em Adult, uma camada estrela com largura h compra mais acurácia por FLOP que uma
camada densa ReLU com largura 2h? O resultado muda com a largura?

**Hipótese principal, anterior aos resultados:** para h=32 e h=64, estrela-ReLU
superará o denso-2h em pelo menos 0,3 ponto percentual de acurácia média de teste,
com separação dos intervalos média ± desvio-padrão amostral.
**Hipótese mecanística secundária:** a vantagem estrela−denso será maior com
somente indicadores categóricos que com somente atributos contínuos.
Nenhum resultado é pressuposto; ausência de separação será inconclusiva, não equivalência.

## Desenho fixado

- Sementes 0, 1, 2; h em 16, 32, 64, 128; 100 épocas full-batch; Adam lr=0,01,
  betas=(0,9;0,999), epsilon=1e-8, sem decaimento de pesos; float64.
- Referência original denso-ReLU 64; seed=0 precisa arredondar a 0,8543 e produzir
  exatamente 84.227.782.200 FLOPs de treino. Interromper se falhar.
- Dados completos: denso-ReLU 2h, estrela-ReLU h, soma de dois ramos h,
  GLU h, GEGLU h e SwiGLU h, todos com saída Linear(...,2).
- Ablações: repetir estrela e denso-2h em contínuos (6 entradas) e categóricos
  (102 entradas). Não escolher largura com base no teste.
- Controle de teto: em cada bloco e h, denso com a maior largura inteira cuja
  inferência não exceda a da estrela. Isso não é igualdade exata de custo usado.
- Mesma divisão treino/validação por semente em todas as arquiteturas; nenhuma
  seleção de checkpoint, early stopping ou ajuste de hiperparâmetros pelo teste.
- Usar `research.benchmark.run_adult` sem modificar o instrumento. Seleção de
  atributos dentro do modelo. Todos os resultados da época 100 serão publicados.
- Não retunar a taxa de aprendizado: o estimando é a troca de arquitetura sob
  o protocolo da referência, não o melhor desempenho possível de cada família.

## Medição e interpretação

Publicar treino e inferência separadamente; a contagem de treino inclui forward
de validação e exclui Adam e backward de operações elementares. Reportar custo
medido do motor, não alegar velocidade proporcional aos FLOPs. Tempo em processos
concorrentes é apenas custo de reprodução, não benchmark de latência.

Estrela e soma têm exatamente o mesmo custo contado e os mesmos parâmetros.
Para d entradas, uma inferência por exemplo custa (2d+7)m+4 no denso de largura m
e (4d+10)h+4 na estrela/soma. Para dados completos, 442h+4 versus 446h+4.

Relatar média ± DP (ddof=1), diferenças pareadas por semente e intervalo t de 95%
dessas diferenças (3 sementes, 2 graus de liberdade). O intervalo t é descritivo
sob aproximação de normalidade; não corrige comparações múltiplas nem incerteza
da população de teste. Para declarar vantagem clara, exigir também ausência de
sobreposição de média±DP, conforme a regra pedagógica do enunciado. Essa regra
não é, isoladamente, teste estatístico. Fronteira de Pareto das médias é descritiva.

## Verificações e entregas

Validar forward contra NumPy independente, gradientes de entradas/pesos/vieses
por diferenças finitas, contagem de FLOPs e reprodução da referência. Guardar
resultados individuais, curvas, pesos finais, hashes dos dados/código, ambiente,
comandos, figuras, tabela `summarize`, figuras `pareto` e relatório em nove seções.
O pré-processamento original usa todo o treino oficial antes da divisão interna:
declarar a influência das features da validação e preservar a comparação histórica.
