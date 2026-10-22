# Restrições persistentes de recursos deste projeto

- Máquina: CachyOS, 16 GB de RAM; GPU AMD RX 9060 XT com 16 GB de VRAM.
- Prioridade: preservar VS Code e o sistema. Meta conjunta editor/ambiente de desenvolvimento/cálculos
  <=10 GB de RAM; manter >=4 GB de memória física disponível e evitar swapping.
- Antes de cálculo pesado, medir MemAvailable, swap/PSI e memória dos processos.
- Execução sequencial por padrão; no máximo dois trabalhadores, apenas após medir
  segurança. Fixar BLAS/OMP/MKL/NumExpr em uma thread. Não lançar treinos em paralelo.
- Treinos Q5 somente pelo executor protegido `q05_estudo/executar_seguro.py`:
  cgroup separado do VS Code, MemoryMax/MemorySwapMax/TasksMax e monitoramento.
  Nunca executar o script histórico desprotegido de `q05_estudo/historico/`.
- O executor protegido interrompe o filho se a memória disponível cai abaixo da
  margem ou há pressão/swap. Não contornar esse bloqueio para terminar mais rápido.
- Liberar grafos autograd entre épocas; manter float64, sementes, épocas e
  resultados científicos, documentando qualquer alteração de execução.
- VRAM não substitui RAM. Este motor NumPy não tem backend GPU; não prometer
  offload sem implementação, validação e benefício comprovados.
- Preservar resultados completos e retomar somente pendências. Não encerrar
  processos alheios ao projeto nem aplicar limites ao cgroup inteiro do VS Code.
