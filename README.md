# Pesquisa experimental com BERT-CPU

Este diretório reúne as perguntas de pesquisa, o instrumento de medição e o estudo experimental da questão 5: **uma camada estrela oferece melhor acurácia por FLOP que uma camada densa mais larga em Adult?**

O estudo está concluído: **156 treinamentos, 52 configurações e três sementes**. A comparação principal é inconclusiva nas larguras estrela 16 e 32 e favorece o denso nas larguras 64 e 128, segundo os critérios registrados. A conclusão se restringe aos dados e ao protocolo avaliados.

## Leitura

1. [Relatório em PDF](q05_estudo/relatorio.pdf) ou [versão textual](q05_estudo/relatorio.md).
2. [Notebook com resultados executados](q05_estudo/estudo_q05.ipynb).
3. [Protocolo experimental](q05_estudo/protocolo.md) e [trabalhos relacionados](q05_estudo/literatura.md).
4. [Código, dados derivados e instruções de reprodução](q05_estudo/README.md).

## Organização

| Caminho | Conteúdo |
|---|---|
| [01_research_questions.ipynb](01_research_questions.ipynb) | Enunciado das quinze perguntas e requisitos da entrega. |
| [benchmark.py](benchmark.py) | Instrumento de medição de desempenho e FLOPs. |
| [q05_estudo/](q05_estudo/) | Implementação, relatório, análises e documentação da questão 5. |
| [q05_estudo/resultados/](q05_estudo/resultados/) | Métricas, curvas, pesos, tabelas e auditorias. |
| [q05_estudo/figuras/](q05_estudo/figuras/) | Figuras do relatório. |
| [q05_estudo/historico/](q05_estudo/historico/) | Registros de proveniência e da reorganização dos arquivos. |
| [RECURSOS.md](RECURSOS.md) | Limites obrigatórios de memória e execução. |
| [q05_estudo/CHECKPOINT.md](q05_estudo/CHECKPOINT.md) | Estado da entrega e ponto de continuidade. |

O pacote Python chama-se `pesquisa`. Os nomes originais presentes no protocolo e nos registros históricos identificam o ambiente da execução e foram preservados. O mapeamento e os hashes das versões estão documentados em [historico/migracao_estrutura.json](q05_estudo/historico/migracao_estrutura.json).

Para consultar os resultados existentes, não é necessário repetir os treinamentos. Qualquer nova execução deve respeitar [RECURSOS.md](RECURSOS.md) e utilizar o executor protegido descrito no estudo.

## Reprodução a partir deste repositório

Este repositório publica o módulo `pesquisa` do BERT-CPU. O motor `bert_cpu`, os carregadores `datasets` e os dados Adult pertencem ao [repositório BERT-CPU](https://github.com/luisfilipeap/BERT-cpu) e são dependências externas. Para executar o estudo, clone esta publicação como a pasta `pesquisa` dentro de uma cópia completa daquele projeto e siga as instruções de [reprodução](q05_estudo/README.md). O relatório e os resultados salvos podem ser consultados diretamente neste repositório.
