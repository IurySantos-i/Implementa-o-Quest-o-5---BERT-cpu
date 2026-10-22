# Diagnóstico e correção do consumo de memória

Os eventos do journal em 13/10/2026 às 10:32, 10:36 e 10:38 (America/Recife) registraram encerramento dos escopos do VS Code pelo systemd-oomd. A ocupação global de RAM/swap ultrapassou 90%, e os escopos mortos tinham mais de 10 GB de swap.

## Operações responsáveis

1. Cada Tensor guarda valor e gradiente NumPy, inclusive vários nós constantes. Isso amplia a memória do grafo em relação ao tamanho apenas das ativações.
2. `_backward` referencia o nó de saída e seus pais; `_prev` guarda as arestas. Esses ciclos requerem coleta cíclica, não apenas contagem de referências.
3. `run_adult` cria grafos completos de treino e validação em cada época. O executor inicial fazia coleta explícita somente depois do treino inteiro.
4. Três trabalhadores full-batch multiplicavam a retenção. O pool era limitado a três; não foi identificada criação ilimitada de processos. Faltavam teto de RAM, swap e monitoramento.
5. Não foram encontrados trabalhadores antigos ainda executando após os encerramentos. Nenhum processo alheio ao estudo foi encerrado nesta correção.

## Evidência controlada

O diagnóstico sintético usa 8 entradas, 128 exemplos, GLU de largura 16 e 60 épocas; limite de espaço virtual de 768 MiB. O coletor automático permaneceu habilitado.

- Máximo de objetos Tensor sem coleta explícita por época: **634**.
- Com coleta explícita por época: **37**.
- Pesos e perdas equivalentes com tolerância 1e-12.
- Pico RSS do próprio diagnóstico: **46.41 MiB**.

Esses números demonstram retenção de objetos; não estimam diretamente os GB usados em Adult. A validação protegida em Adult reproduziu acurácia **0.8543087034**, **84227782200 FLOPs de treino**, **232427556 FLOPs de inferência** e curvas equivalentes. Seu pico de cgroup foi **0.556 GiB**.

## Proteções em vigor

O executor padrão agora chama executar_seguro.py. Um único trabalhador, uma thread BLAS, cgroup próprio com MemoryHigh=2G, MemoryMax=3G, MemorySwapMax=0, TasksMax=16 e limite temporal de dez minutos. O supervisor e o trabalhador verificam a margem de RAM; o filho verifica se o supervisor ainda existe. Trava de arquivo impede duas retomadas.

O início exige 7 GiB disponíveis; a execução é encerrada preventivamente abaixo de 4,5 GiB ou acima de 9 GB no conjunto monitorado. Há checagens de PSI e saída global para swap. Os limites aplicam-se somente ao filho de cálculo, não ao editor. O serviço systemd libera todo o grupo ao terminar.

Não foi reduzida a precisão, o lote, as épocas, a grade nem o número de sementes. A coleta usa o hook existente do benchmark; o motor e o instrumento originais foram preservados. Não há offload para a Radeon porque o motor é NumPy/CPU e não implementa backend GPU.

## Registros

- `resultados/diagnostico_grafos.json`: diagnóstico pequeno e equivalência.
- `resultados/equivalencia_memoria.json`: validação da referência sob coleta explícita.
- `resultados/monitoramento_memoria.jsonl`: amostras a cada dois segundos.
- `resultados/sessoes_seguras.jsonl`: tempo e consumo de cada serviço concluído.
- `resultados/resumo_recursos.json`: resumo final gerado pela análise.
- `adendo_recursos.md`: alteração operacional e proveniência dos resultados.
- `test_recursos.py`: quatro verificações de bloqueio/grade, todas aprovadas.

## Retomada com 73 resultados preservados

Em 13/10/2026 às 19:21:26 UTC, a proteção interrompeu `completo_glu_h128_s1`
quando `psi_full10` atingiu 1,1%, acima do limite de 1%. Havia 9.110.331.392
bytes disponíveis e o contador global de saída para swap permaneceu em
23.737.007 páginas durante essa tentativa. Esses registros indicam o gatilho
observado, mas não identificam a origem da pressão global. O controlador encerrou
somente seu serviço; os 73 resultados completos permaneceram intactos.

A retomada ocorreu após nova inspeção mostrar PSI zero, 9.782.312.960 bytes
disponíveis e o mesmo contador de swap. Nenhum limite foi relaxado. A configuração
pendente foi então concluída com as 100 épocas originais. O tempo parcial da
tentativa abortada não integra `sessoes_seguras.jsonl`, que contém apenas serviços
concluídos; as amostras da tentativa permanecem no monitoramento.

## Conclusão da execução protegida

A grade terminou com **156 resultados completos**: 51 históricos preservados e
105 treinos pelo executor protegido. O último foi `categorico_denso_teto_h128_s2`.
A auditoria final conferiu os 156 arquivos de pesos, os custos e três inferências
independentes, sem repetir treinamentos.

O maior `memory.peak` dos cgroups de treino foi **1.996.980.224 bytes (1,86 GiB)**.
A menor memória física disponível nas amostras foi **7.397.675.008 bytes (6,89 GiB)**;
o pico amostrado do conjunto trabalhador + processos monitorados do VS Code foi
**2.793.996.288 bytes (2,60 GiB)**. São grandezas diferentes: o pico do cgroup inclui
alocações entre amostras, enquanto os outros números vêm do monitor periódico.
O contador global de swap teve crescimento no período; não se atribui esse
movimento ao filho, cujo cgroup manteve `MemorySwapMax=0`.

Na inspeção posterior à geração dos documentos, havia **9.118.777.344 bytes
(8,49 GiB)** disponíveis, PSI de memória igual a zero e nenhum serviço
`bert-q05-calculo-*` carregado. Os dados completos estão em
`resultados/resumo_recursos.json` e nos registros de monitoramento.
