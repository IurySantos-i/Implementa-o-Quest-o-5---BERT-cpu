"""Atualiza o ponto de retomada lendo os artefatos; não inicia cálculos."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

PASTA = Path(__file__).resolve().parent
RESULTADOS = PASTA / "resultados"


def identificador(c):
    return f'{c["bloco"]}_{c["familia"]}_h{c["h"]}_s{c["semente"]}'


def entrega_verificada():
    """Não mantém a marca final se um artefato mudou após sua verificação."""
    try:
        verificacao=json.loads((RESULTADOS / 'verificacao_entrega.json').read_text())
        manifesto=(RESULTADOS / 'hashes_entrega.json').read_bytes()
        return (verificacao['aprovado']
                and verificacao['sha256_manifesto_entrega'] == hashlib.sha256(manifesto).hexdigest()
                and all(hashlib.sha256((PASTA / nome).read_bytes()).hexdigest() == esperado
                        for nome, esperado in json.loads(manifesto).items()))
    except (OSError, ValueError, KeyError):
        return False


def resposta_principal(concluidos):
    """Resumo legível das comparações já completas, sem depender das ablações."""
    registros = dict(concluidos)
    nomes = [f"completo_{f}_h{h}_s{s}" for h in (16, 32, 64, 128)
             for f in ("estrela", "denso") for s in (0, 1, 2)]
    if not all(n in registros for n in nomes):
        return "As comparações principais ainda não estão todas concluídas.\n"
    linhas = []
    for h in (16, 32, 64, 128):
        grupos = [[registros[f"completo_{f}_h{h}_s{s}"] for s in (0, 1, 2)]
                  for f in ("estrela", "denso")]
        medias = [100 * statistics.mean(r['score'] for r in g) for g in grupos]
        dps = [100 * statistics.stdev(r['score'] for r in g) for g in grupos]
        deltas = [100 * (a['score'] - b['score']) for a, b in zip(*grupos)]
        delta = statistics.mean(deltas)
        margem = 4.302652729911275 * statistics.stdev(deltas) / 3**.5
        separadas = abs(medias[0] - medias[1]) > sum(dps)
        leitura = ("Estrela favorecida" if separadas and delta - margem > 0 else
                   "Denso favorecido" if separadas and delta + margem < 0 else
                   "Inconclusivo")
        n = lambda x: f"{x:.3f}".replace('.', ',')
        treino = ' / '.join(n(g[0]['train_flops'] / 1e9) for g in grupos)
        inferencia = ' / '.join(n(g[0]['infer_flops'] / 1e6) for g in grupos)
        linhas.append(f"| {h} | {n(medias[0])} ± {n(dps[0])} | {n(medias[1])} ± {n(dps[1])} | "
                      f"{n(delta)} [{n(delta-margem)}; {n(delta+margem)}] | {treino} | {inferencia} | {leitura} |")
    return ("## Resposta já medida para a pergunta principal\n\n"
            "Comparação de estrela h com denso 2h, usando as 24 execuções completas "
            "(quatro larguras × duas famílias × três sementes). A tabela abaixo é "
            "recalculada diretamente dos JSONs salvos. As extensões e a auditoria final "
            "têm seu próprio status no checklist.\n\n"
            "| h | Estrela: teste (%) ± DP | Denso: teste (%) ± DP | Δ estrela−denso e IC t 95% (p.p.) | "
            "Treino GFLOPs (estrela / denso) | Inferência MFLOPs (estrela / denso) | Leitura no protocolo |\n"
            "|---:|---:|---:|---:|---:|---:|---|\n" + '\n'.join(linhas) + "\n\n"
            "A regra de leitura exige IC que exclua zero e barras média±DP separadas. "
            "Sobreposição não demonstra equivalência. O pareamento tem custos próximos, "
            "não exatamente iguais. O controle adicional de teto e as ablações permitem "
            "examinar essa limitação; a conclusão se restringe a Adult, 100 épocas e Adam 0,01.\n")


def main():
    grade = json.loads((RESULTADOS / "grade_planejada.json").read_text())
    concluidos, pendentes, inconsistentes = [], [], []
    for c in grade:
        nome = identificador(c)
        caminho = RESULTADOS / f"{nome}.json"
        if not caminho.exists():
            pendentes.append(nome)
            continue
        try:
            r = json.loads(caminho.read_text())
            completo = (r["configuracao"] == c
                        and len(r["extra"]["train_hist"]) == 100
                        and len(r["extra"]["val_hist"]) == 100
                        and (RESULTADOS / f"{nome}_pesos.npz").is_file())
        except (ValueError, KeyError):
            completo = False
        if completo:
            concluidos.append((nome, r))
        else:
            inconsistentes.append(nome)

    ultimo = max(concluidos, key=lambda par: par[1]["concluido_utc"], default=None)
    contagem = Counter(r["configuracao"]["bloco"] for _, r in concluidos)
    esperados = Counter(c["bloco"] for c in grade)
    finalizada = len(concluidos) == len(grade) and not inconsistentes
    itens = [
        (True, "Protocolo, hipóteses e grade registrados antes dos experimentos."),
        (True, "Referência original reproduzida; sete testes numéricos aprovados (333 coordenadas)."),
        (True, "Causa dos OOMs investigada; executor protegido e quatro testes de recursos concluídos."),
        (True, "Equivalência da coleta de grafos confirmada: curvas, acurácia e FLOPs preservados."),
        (True, "Modelos, literatura e geradores de análise, relatório e notebook implementados."),
        (finalizada, f"Grade completa: {len(concluidos)}/{len(grade)} treinamentos salvos com pesos e 100 épocas."),
        ((RESULTADOS / "auditoria_final.json").exists(), "Auditoria final de resultados, hashes, custos e inferências."),
        ((RESULTADOS / "analise.json").exists(), "Tabelas, gráficos e contrastes das hipóteses gerados."),
        ((PASTA / "relatorio.pdf").exists() and (PASTA / "estudo_q05.ipynb").exists(),
         "Relatório em português e notebook com saídas reais gerados."),
        (entrega_verificada(),
         "Interpretação final revisada; PDF de 6–10 páginas e notebook verificados."),
        ((RESULTADOS / "verificacao_estrutura.json").exists(),
         "Organização acadêmica e migração do pacote para pesquisa verificadas."),
    ]
    ultima_etapa=next(item for pronto, item in reversed(itens) if pronto)
    texto = [
        "# Checkpoint — entrega da Q5\n",
        f"Atualizado em {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC). "
        "Este retrato é gerado por `python q05_estudo/atualizar_checkpoint.py`. "
        "O programa lê arquivos pequenos, sem treinar nem importar NumPy.\n",
        "## Última etapa concluída\n",
        f"**{ultima_etapa}**\n",
        (f"Último treinamento salvo: **`{ultimo[0]}`**, concluído em "
         f"{ultimo[1]['concluido_utc']}.\n" if ultimo else "Nenhum treinamento completo.\n"),
        f"**{len(concluidos)} de {len(grade)} concluídos; {len(pendentes)} pendentes; "
        f"{len(inconsistentes)} inconsistentes.** Os resultados completos devem ser preservados. "
        "A inspeção deste checkpoint confere presença, configuração e tamanho das curvas; "
        "a validação científica completa cabe a `verificar_resultados.py`.\n",
        "| Bloco | Concluídos | Planejados |\n|---|---:|---:|\n" + "\n".join(
            f"| {b} | {contagem[b]} | {n} |" for b, n in esperados.items()) + "\n",
        resposta_principal(concluidos),
        "## Organização dos arquivos\n",
        "A pasta raiz chama-se `pesquisa`; as demais pastas já usam nomes em português. "
        "Os imports atuais usam `pesquisa.benchmark` e `pesquisa.q05_estudo`. "
        "As regras obrigatórias de memória estão em `RECURSOS.md`. "
        "O mapeamento de caminhos e hashes está em `historico/migracao_estrutura.json`; "
        "o checkpoint histórico conserva as versões anteriores para auditoria. "
        "Os treinamentos, pesos, curvas e figuras científicas foram preservados.\n",
        "## Checklist\n",
        "\n".join(f"- [{'x' if pronto else ' '}] {item}" for pronto, item in itens) + "\n",
        "## Próxima ação e retomada\n",
        (f"A primeira configuração ainda sem resultado é **`{pendentes[0]}`**. "
         "Ela pode estar em execução: a existência do arquivo é a confirmação de conclusão.\n"
         if pendentes else
         "**Entrega concluída e verificada.** Não há etapa pendente neste retrato. "
         "Em alterações futuras, retome apenas as verificações e os artefatos afetados.\n"
         if all(pronto for pronto, _ in itens) else
         "A grade está salva; prossiga com as etapas de análise ainda desmarcadas.\n"),
        "1. Leia `RECURSOS.md` e este checkpoint. Atualize-o para obter a contagem atual.\n"
        "2. Confira se já existe supervisor `executar_seguro.py` ou serviço "
        "`bert-q05-calculo-*`. Se existir, acompanhe-o; não lance outro. "
        "Um identificador de sessão de ferramenta perdido não prova que o cálculo terminou.\n"
        "3. Antes de retomar, consulte `python q05_estudo/executar.py --inspecionar`: "
        "MemAvailable, swap, PSI e processos. O executor exige a margem de segurança.\n"
        "4. Somente se faltarem treinos e não houver execução ativa, use "
        "`python -u q05_estudo/executar_seguro.py`. "
        "Ele valida hashes e pula todos os resultados completos. Se a proteção impedir a "
        "execução, resolva a pressão de recursos; não contorne os limites.\n"
        "5. Ao completar a grade, execute somente os passos ainda pendentes abaixo. "
        "Audite antes de interpretar; revise H2 e variantes com as tabelas finais.\n",
        "```bash\npython q05_estudo/verificar_resultados.py\n"
        "python q05_estudo/analisar.py\npython q05_estudo/gerar_relatorio.py\n"
        "python q05_estudo/verificar_entrega.py\n"
        "python q05_estudo/atualizar_checkpoint.py\n```\n",
        "Antes de marcar a revisão final, inspecione visualmente as páginas do PDF e "
        "confira a interpretação dos contrastes. O verificador automático complementa "
        "essa revisão; não a substitui.\n",
        "## Restrições que continuam valendo\n",
        "Um treinamento por vez, uma thread, cgroup próprio de até 3 GiB e swap zero. "
        "Preservar ≥4 GiB disponíveis e a meta conjunta de RAM do projeto/VS Code. "
        "Não executar `historico/executar_original.py`, não alterar os arquivos científicos "
        "ou de segurança com hashes enquanto a grade estiver em execução e não reduzir "
        "épocas, precisão ou sementes. Não repetir testes já aprovados sem uma mudança "
        "relevante. A GPU não é utilizada pelo motor NumPy deste estudo.\n",
        "O monitoramento está em `resultados/monitoramento_memoria.jsonl`; as sessões, "
        "em `resultados/sessoes_seguras.jsonl`. Os JSONs individuais registram métricas, "
        "curvas, horário de conclusão e, na versão protegida, pico real do cgroup. "
        "Os 51 resultados históricos válidos permanecem preservados.\n",
    ]
    if inconsistentes:
        texto.append("**Resolver antes de continuar:** " + ", ".join(inconsistentes) + "\n")
    (PASTA / "CHECKPOINT.md").write_text("\n".join(texto), encoding="utf-8")
    print(f"Checkpoint atualizado: {len(concluidos)}/{len(grade)} completos; "
          f"{len(pendentes)} pendentes; {len(inconsistentes)} inconsistentes.")


if __name__ == "__main__":
    main()
