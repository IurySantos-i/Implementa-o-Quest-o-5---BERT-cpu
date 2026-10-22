# Q5 — Estrela versus camada densa em Adult

Esta entrega responde à Q5 de `01_research_questions.ipynb`: uma camada estrela de largura `h` oferece melhor acurácia por FLOP que uma camada densa ReLU de largura `2h`, e a resposta depende da largura? O relatório, o notebook, os comentários e as análises estão em português brasileiro.

**Entrega concluída:** 156 treinamentos auditados, 52 configurações com três sementes, relatório de nove páginas e notebook com saídas reais. No protocolo fixado, a comparação principal é inconclusiva em `h=16` e `h=32` e favorece o denso em `h=64` e `h=128`. As tabelas do relatório publicam as acurácias, dispersões e os FLOPs correspondentes.

Para retomar uma sessão, consulte primeiro o [CHECKPOINT.md](CHECKPOINT.md), com a última etapa concluída, contagem por bloco e checklist da entrega. Atualize esse retrato sem executar cálculos com `python q05_estudo/atualizar_checkpoint.py`. Não há treinamento pendente nesta entrega.

Comece por [relatorio.pdf](relatorio.pdf) ou [relatorio.md](relatorio.md). O [notebook](estudo_q05.ipynb) serve para leitura e análise dos resultados salvos; por padrão, ele não dispara os 156 treinamentos. O [protocolo](protocolo.md) contém as hipóteses e as regras registradas antes dos experimentos.

## Conteúdo da entrega

| Arquivo ou pasta | Finalidade |
|---|---|
| [relatorio.pdf](relatorio.pdf) | Relatório diagramado nas nove seções exigidas. |
| [relatorio.md](relatorio.md) | Versão textual do relatório. |
| [estudo_q05.ipynb](estudo_q05.ipynb) | Leitura guiada, derivação e análise dos resultados existentes. |
| [modelos.py](modelos.py) | Camada densa, estrela, soma e variantes GLU; seleção dos blocos de atributos e custos teóricos. |
| [executar.py](executar.py) | Entrada padrão para o executor protegido. |
| [executar_seguro.py](executar_seguro.py) | Execução sequencial, cgroups isolados, coleta de grafos e monitoramento. |
| [analisar.py](analisar.py) | Tabelas, diferenças pareadas, intervalos e figuras `pareto()`. |
| [gerar_relatorio.py](gerar_relatorio.py) | Geração do relatório a partir das análises. |
| [test_modelos.py](test_modelos.py) | Verificações de forward, gradientes, parâmetros e FLOPs. |
| [verificar_resultados.py](verificar_resultados.py) | Auditoria dos resultados e artefatos salvos. |
| [verificar_entrega.py](verificar_entrega.py) | Conferência do PDF, notebook, seções, tabela e hashes da entrega. |
| [CHECKPOINT.md](CHECKPOINT.md) | Ponto de retomada e checklist, atualizado por `atualizar_checkpoint.py`. |
| [protocolo.md](protocolo.md) | Pergunta, hipóteses, orçamento e interpretação fixados previamente. |
| [resultados/](resultados/) | Registros individuais, curvas, pesos, ambiente, hashes e tabelas. |
| [figuras/](figuras/) | Figuras exportadas para consulta e inclusão no relatório. |
| [literatura.md](literatura.md) | Discussão das quatro leituras e previsões para este contexto. |
| [referencias.bib](referencias.bib) | Referências bibliográficas em BibTeX. |

## Desenho experimental

São três sementes (`0`, `1`, `2`), quatro larguras estrela (`16`, `32`, `64`, `128`) e 100 épocas em lote completo por treinamento. Mantêm-se Adam com taxa `0,01`, precisão `float64` e o instrumento original `pesquisa.benchmark.run_adult`. O orçamento fixado é de épocas; os FLOPs de treino e inferência são medidos e publicados separadamente.

A grade tem **156 treinamentos**, executados **sequencialmente**, com uma thread BLAS por processo:

- Dados completos: denso `2h`, estrela `h`, soma de dois ramos `h`, GLU, GEGLU, SwiGLU e denso sob o teto de inferência da estrela. São 84 treinamentos.
- Ablações com apenas os seis atributos contínuos ou os 102 indicadores categóricos: denso `2h`, estrela `h` e denso sob o teto, nos mesmos pares de larguras e sementes. São 72 treinamentos.

A estrela-ReLU já corresponde à ReGLU; ela não aparece duplicada como outra variante. A reprodução da referência original é uma verificação adicional à grade. O controle de teto usa a maior largura densa inteira que não excede a inferência da estrela; não significa que treino, parâmetros e tempo sejam exatamente iguais.

## Segurança de memória — requisito obrigatório

Consulte o [diagnóstico de recursos](diagnostico_recursos.md), com os registros de OOM, a causa da retenção e as verificações de equivalência.

A máquina tem 16 GB de RAM. As primeiras tentativas com três trabalhadores sofreram encerramentos por `systemd-oomd`; **não execute o arquivo histórico desprotegido**. Os 51 resultados completos foram preservados. O [adendo operacional](adendo_recursos.md) documenta a mudança sem alterar sementes, dados, precisão, lote ou épocas.

O executor atual exige Linux com `systemd --user` e cria **um serviço filho separado do VS Code** para cada configuração: `MemoryHigh=2G`, `MemoryMax=3G`, `MemorySwapMax=0`, `TasksMax=16`, `CPUQuota=100%`, uma thread BLAS e no máximo dez minutos por configuração. O filho verifica a existência do supervisor; o controlador impede duas retomadas simultâneas por trava de arquivo.

Antes de iniciar exige 7 GiB disponíveis. Durante o treino monitora a cada dois segundos e interrompe somente o serviço de cálculo se MemAvailable cair abaixo de 4,5 GiB, o conjunto monitorado exceder 9 GB, houver pressão de memória ou crescimento expressivo de swap. Não limita nem encerra o VS Code. Se a margem não existir, aguarde a liberação natural de recursos; não contorne a proteção.

A coleta explícita de ciclos ao final de cada época evita acumular grafos NumPy. A equivalência foi verificada contra a referência nas curvas, acurácia e FLOPs. A VRAM da Radeon não é usada: este motor não implementa backend GPU, e portar operações alteraria a implementação científica.

`monitoramento_memoria.jsonl`, `sessoes_seguras.jsonl` e os picos de cgroup nos JSONs registram o consumo real. Execute os comandos de treino em um terminal de usuário com acesso ao `systemd --user`; em um sandbox sem esse acesso, o executor não deve ser contornado.

```bash
python q05_estudo/executar.py --inspecionar
python q05_estudo/test_recursos.py
```

## Dependências

Use Python com `numpy` e `matplotlib`, além dos arquivos locais do repositório. Não é necessário Pipenv. Em um ambiente Python destinado ao estudo:

```bash
python -m pip install numpy matplotlib
```

Para gerar o PDF, disponibilize também o executável `xelatex` de uma instalação LaTeX. Para abrir o notebook interativamente, use JupyterLab ou outro leitor de `.ipynb`; Jupyter não é necessário para executar os scripts. As versões efetivamente usadas na execução estão registradas em `resultados/ambiente.json`.

A verificação final usa `pdfinfo` e `pdftotext`, do Poppler: `python q05_estudo/verificar_entrega.py`. Ela confere, entre outros itens, as 6–10 páginas e as nove seções obrigatórias. O checkpoint é um registro mutável de progresso e fica fora do manifesto final de hashes.

## Organização e proveniência

O nome atual do pacote é `pesquisa`; os imports usam `pesquisa.benchmark` e `pesquisa.q05_estudo`. As subpastas `resultados`, `figuras` e `historico` mantêm seus nomes em português. As restrições obrigatórias de memória estão em [RECURSOS.md](../RECURSOS.md).

Os caminhos e hashes registrados durante os experimentos permanecem nos manifestos históricos. O arquivo [historico/migracao_estrutura.json](historico/migracao_estrutura.json) relaciona os arquivos anteriores às versões atuais; o checkpoint histórico conserva o conteúdo original para conferência. As alterações de organização não modificam os resultados, pesos, curvas ou fórmulas dos modelos. `proveniencia.py` exige um hash exato ou essa correspondência documentada e rejeita mudanças posteriores não registradas.

Para conferir a integridade dos dados e a organização atual, sem treinar:

```bash
python q05_estudo/verificar_estrutura.py
python q05_estudo/verificar_entrega.py
```

Os arquivos `datasets/adult/adult.data` e `datasets/adult/adult.test` devem continuar presentes no repositório. Execute a entrega dentro da cópia completa de `BERT-cpu`, pois ela importa o motor, os carregadores e o instrumento de medição desse projeto.

## Reproduzir a análise sem treinar novamente

A partir de `/home/iurygds/Documents/BERT-cpu/research`:

```bash
python q05_estudo/test_modelos.py -v
python q05_estudo/verificar_resultados.py
python q05_estudo/analisar.py
python q05_estudo/gerar_relatorio.py
```

Os scripts de análise e relatório usam os resultados salvos. `analisar.py` exige a grade completa e informa qual execução está faltando quando o estudo está incompleto. As verificações numéricas acima não substituem os resultados dos treinamentos.

Os comandos equivalentes a partir da raiz `/home/iurygds/Documents/BERT-cpu` são:

```bash
python pesquisa/q05_estudo/test_modelos.py -v
python pesquisa/q05_estudo/verificar_resultados.py
python pesquisa/q05_estudo/analisar.py
python pesquisa/q05_estudo/gerar_relatorio.py
```

## Executar ou retomar os treinamentos

A partir de `pesquisa/`:

```bash
python q05_estudo/test_modelos.py --json -v
python q05_estudo/executar.py --processos 1
python q05_estudo/verificar_resultados.py
python q05_estudo/analisar.py
python q05_estudo/gerar_relatorio.py
```

A partir da raiz do repositório:

```bash
python pesquisa/q05_estudo/test_modelos.py --json -v
python pesquisa/q05_estudo/executar.py --processos 1
python pesquisa/q05_estudo/verificar_resultados.py
python pesquisa/q05_estudo/analisar.py
python pesquisa/q05_estudo/gerar_relatorio.py
```

`executar.py` verifica a referência original antes das variantes e retoma automaticamente os treinamentos ainda não salvos. Uma execução interrompida pode ser retomada repetindo o mesmo comando. Com todos os registros presentes, o comando não treina novamente a grade. A opção `--somente-referencia` executa apenas a equivalência da referência com a coleta de grafos. O executor aceita somente `--processos 1`; tentativas de usar três processos são rejeitadas.

O programa verifica os hashes científicos e os do executor protegido. A versão inicial do executor é preservada em `historico/`, com seu manifesto, para validar a proveniência das primeiras 51 execuções sem misturá-la silenciosamente com a versão protegida. Essa checagem preserva a identificação do experimento. Os registros de sessão e os tempos de cada treinamento permitem consultar o custo real de reprodução; tempos com processos concorrentes não são uma comparação de latência entre arquiteturas.

## Repetir o estudo do início preservando a entrega

Para uma execução independente ou após alterações no código, copie o repositório inteiro e mova os resultados existentes para um backup **na cópia**. Use uma pasta em disco com espaço disponível; não copie o estudo para um `tmpfs`, pois ele consome RAM. O exemplo abaixo cria uma pasta exclusiva em `Documents` e preserva a entrega original:

```bash
df -h /home/iurygds/Documents
pasta_reproducao="$(mktemp -d /home/iurygds/Documents/bert-cpu-q05.XXXXXX)"
cp -a --reflink=auto /home/iurygds/Documents/BERT-cpu "$pasta_reproducao/BERT-cpu"
mv "$pasta_reproducao/BERT-cpu/pesquisa/q05_estudo/resultados" "$pasta_reproducao/resultados_da_entrega"
cd "$pasta_reproducao/BERT-cpu/research"
python q05_estudo/test_modelos.py --json -v
python q05_estudo/executar.py --processos 1
python q05_estudo/verificar_resultados.py
python q05_estudo/analisar.py
python q05_estudo/gerar_relatorio.py
```

O backup fica ao lado da cópia de `BERT-cpu`, dentro de `pasta_reproducao`. Os novos resultados recebem outro manifesto e novos tempos. A opção `--reflink=auto` compartilha os blocos dos arquivos quando o sistema de arquivos permite; a execução permanece sujeita às mesmas proteções de memória.

## Preparação para a prova

1. Expanda o produto de dois ramos afins usando o viés em `[1; x]` e identifique os termos quadráticos.
2. Derive o gradiente pelas duas rotas do produto e explique o comportamento nos dois lados da ReLU.
3. Confira as dimensões das matrizes e compare o custo da rede inteira nos pares `h` e `2h`.
4. Trace entropia cruzada, retropropagação e uma atualização dos pesos em um exemplo pequeno.
5. Leia uma diferença pareada e sua incerteza antes de formular uma conclusão a partir de uma curva de Pareto.
