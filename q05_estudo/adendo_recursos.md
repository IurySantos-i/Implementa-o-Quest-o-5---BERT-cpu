# Adendo operacional — 13/10/2026

Este adendo preserva o protocolo científico e os 51 treinos concluídos. A alteração
foi exigida após três eventos de systemd-oomd às 10:32, 10:36 e 10:38 (Recife).
O journal registra mais de 10 GB de swap no escopo do VS Code e ocupação global
de RAM/swap superior a 90%. As retomadas anteriores foram interrompidas por OOM;
não se deve atribuir essas interrupções ao fechamento voluntário do editor.

O motor Tensor armazena closures `_backward` que referenciam nós, inclusive o nó
de saída. Isso forma ciclos. `run_adult` cria grafos de treino e validação a cada
época, sem coleta explícita; o executor inicial só chamava gc.collect depois de
100 épocas. O coletor conta objetos Python, não o volume dos buffers NumPy.
Três processos full-batch multiplicaram esse custo transitório. Não havia uma
criação ilimitada de trabalhadores (o pool era de três), mas faltavam limites de
memória, monitoramento e controle da duração dos grafos.

## Mudanças permitidas, sem alterar o experimento

- Um trabalhador por vez, em serviço systemd de usuário separado do VS Code.
- MemoryHigh=2G; MemoryMax=3G; MemorySwapMax=0; TasksMax=16; CPUQuota=100%.
- Monitorar a cada dois segundos: MemAvailable, PSI, entradas/saídas de swap,
  memória do trabalhador e do VS Code. Interromper o filho ao ultrapassar margens.
- Antes de iniciar, exigir pelo menos 7 GiB disponíveis; durante o treino,
  interromper abaixo de 4,5 GiB disponíveis ou ao atingir 9 GB conjuntos.
- Coletar ciclos ao final de toda época pelo hook `on_epoch` já existente, e após
  cada treino. Essa coleta não muda as operações numéricas, dados ou FLOPs.
- Um processo novo por configuração evita retenção de arenas entre treinos.
- Preservar float64, full-batch, 100 épocas, Adam 0,01, seeds 0/1/2 e toda a grade.
- Nenhum backend GPU é usado: o código disponível é NumPy/CPU; não há Cupy,
  PyTorch ou JAX no ambiente inspecionado. Transferir esse motor para GPU seria
  uma alteração de implementação e de instrumento, não uma opção transparente.

`historico/executar_original.py` conserva o hash do executor das primeiras 51
execuções. `ambiente.json` permanece histórico e imutável. As execuções restantes
recebem um manifesto operacional novo. A auditoria valida ambas as versões e
exige equivalência da referência com o hook de coleta antes de retomar a grade.
Tempos de execução antigos concorrentes e novos sequenciais serão discriminados.
