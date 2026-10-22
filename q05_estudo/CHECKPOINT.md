# Checkpoint — entrega da Q5

Atualizado em 2026-10-22T22:06:42+00:00 (UTC). Este retrato é gerado por `python q05_estudo/atualizar_checkpoint.py`. O programa lê arquivos pequenos, sem treinar nem importar NumPy.

## Última etapa concluída

**Organização acadêmica e migração do pacote para pesquisa verificadas.**

Último treinamento salvo: **`categorico_denso_teto_h128_s2`**, concluído em 2026-10-13T20:27:49.662338+00:00.

**156 de 156 concluídos; 0 pendentes; 0 inconsistentes.** Os resultados completos devem ser preservados. A inspeção deste checkpoint confere presença, configuração e tamanho das curvas; a validação científica completa cabe a `verificar_resultados.py`.

| Bloco | Concluídos | Planejados |
|---|---:|---:|
| completo | 84 | 84 |
| continuo | 36 | 36 |
| categorico | 36 | 36 |

## Resposta já medida para a pergunta principal

Comparação de estrela h com denso 2h, usando as 24 execuções completas (quatro larguras × duas famílias × três sementes). A tabela abaixo é recalculada diretamente dos JSONs salvos. As extensões e a auditoria final têm seu próprio status no checklist.

| h | Estrela: teste (%) ± DP | Denso: teste (%) ± DP | Δ estrela−denso e IC t 95% (p.p.) | Treino GFLOPs (estrela / denso) | Inferência MFLOPs (estrela / denso) | Leitura no protocolo |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 85,490 ± 0,124 | 85,462 ± 0,132 | 0,029 [-0,424; 0,481] | 41,612 / 42,154 | 115,204 / 116,246 | Inconclusivo |
| 32 | 85,095 ± 0,199 | 85,472 ± 0,158 | -0,377 [-0,903; 0,149] | 83,144 / 84,228 | 230,344 / 232,428 | Inconclusivo |
| 64 | 84,808 ± 0,302 | 85,374 ± 0,079 | -0,565 [-1,121; -0,009] | 166,209 / 168,376 | 460,622 / 464,790 | Denso favorecido |
| 128 | 84,270 ± 0,089 | 85,204 ± 0,103 | -0,934 [-1,407; -0,460] | 332,338 / 336,673 | 921,179 / 929,515 | Denso favorecido |

A regra de leitura exige IC que exclua zero e barras média±DP separadas. Sobreposição não demonstra equivalência. O pareamento tem custos próximos, não exatamente iguais. O controle adicional de teto e as ablações permitem examinar essa limitação; a conclusão se restringe a Adult, 100 épocas e Adam 0,01.

## Organização dos arquivos

A pasta raiz chama-se `pesquisa`; as demais pastas já usam nomes em português. Os imports atuais usam `pesquisa.benchmark` e `pesquisa.q05_estudo`. As regras obrigatórias de memória estão em `RECURSOS.md`. O mapeamento de caminhos e hashes está em `historico/migracao_estrutura.json`; o checkpoint histórico conserva as versões anteriores para auditoria. Os treinamentos, pesos, curvas e figuras científicas foram preservados.

## Checklist

- [x] Protocolo, hipóteses e grade registrados antes dos experimentos.
- [x] Referência original reproduzida; sete testes numéricos aprovados (333 coordenadas).
- [x] Causa dos OOMs investigada; executor protegido e quatro testes de recursos concluídos.
- [x] Equivalência da coleta de grafos confirmada: curvas, acurácia e FLOPs preservados.
- [x] Modelos, literatura e geradores de análise, relatório e notebook implementados.
- [x] Grade completa: 156/156 treinamentos salvos com pesos e 100 épocas.
- [x] Auditoria final de resultados, hashes, custos e inferências.
- [x] Tabelas, gráficos e contrastes das hipóteses gerados.
- [x] Relatório em português e notebook com saídas reais gerados.
- [x] Interpretação final revisada; PDF de 6–10 páginas e notebook verificados.
- [x] Organização acadêmica e migração do pacote para pesquisa verificadas.

## Próxima ação e retomada

**Entrega concluída e verificada.** Não há etapa pendente neste retrato. Em alterações futuras, retome apenas as verificações e os artefatos afetados.

1. Leia `RECURSOS.md` e este checkpoint. Atualize-o para obter a contagem atual.
2. Confira se já existe supervisor `executar_seguro.py` ou serviço `bert-q05-calculo-*`. Se existir, acompanhe-o; não lance outro. Um identificador de sessão de ferramenta perdido não prova que o cálculo terminou.
3. Antes de retomar, consulte `python q05_estudo/executar.py --inspecionar`: MemAvailable, swap, PSI e processos. O executor exige a margem de segurança.
4. Somente se faltarem treinos e não houver execução ativa, use `python -u q05_estudo/executar_seguro.py`. Ele valida hashes e pula todos os resultados completos. Se a proteção impedir a execução, resolva a pressão de recursos; não contorne os limites.
5. Ao completar a grade, execute somente os passos ainda pendentes abaixo. Audite antes de interpretar; revise H2 e variantes com as tabelas finais.

```bash
python q05_estudo/verificar_resultados.py
python q05_estudo/analisar.py
python q05_estudo/gerar_relatorio.py
python q05_estudo/verificar_entrega.py
python q05_estudo/atualizar_checkpoint.py
```

Antes de marcar a revisão final, inspecione visualmente as páginas do PDF e confira a interpretação dos contrastes. O verificador automático complementa essa revisão; não a substitui.

## Restrições que continuam valendo

Um treinamento por vez, uma thread, cgroup próprio de até 3 GiB e swap zero. Preservar ≥4 GiB disponíveis e a meta conjunta de RAM do projeto/VS Code. Não executar `historico/executar_original.py`, não alterar os arquivos científicos ou de segurança com hashes enquanto a grade estiver em execução e não reduzir épocas, precisão ou sementes. Não repetir testes já aprovados sem uma mudança relevante. A GPU não é utilizada pelo motor NumPy deste estudo.

O monitoramento está em `resultados/monitoramento_memoria.jsonl`; as sessões, em `resultados/sessoes_seguras.jsonl`. Os JSONs individuais registram métricas, curvas, horário de conclusão e, na versão protegida, pico real do cgroup. Os 51 resultados históricos válidos permanecem preservados.
