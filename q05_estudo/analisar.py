"""Gera tabelas, diferenças pareadas e figuras a partir dos resultados reais."""
import os
for _nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_nome] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/q05_matplotlib")
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parents[1]))
from pesquisa.benchmark import Result, summarize, pareto

NOMES = {"denso": "Denso 2h", "estrela": "Estrela", "soma": "Soma",
         "glu": "GLU", "geglu": "GEGLU", "swiglu": "SwiGLU",
         "denso_teto": "Denso no teto"}
CORES = dict(zip(NOMES, ["#315e9e", "#d05736", "#408060", "#8b62a8",
                        "#b18a26", "#279a9c", "#707070"]))
BLOCOS = {"completo": "Todos os atributos", "continuo": "Somente contínuos",
          "categorico": "Somente categóricos"}


def gravar_csv(caminho, linhas):
    with Path(caminho).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def carregar():
    grade = json.loads((PASTA / "resultados/grade_planejada.json").read_text())
    resultados = []
    for c in grade:
        ident = f'{c["bloco"]}_{c["familia"]}_h{c["h"]}_s{c["semente"]}'
        arquivo = PASTA / "resultados" / f"{ident}.json"
        if not arquivo.exists():
            raise RuntimeError(f"Estudo incompleto: falta {ident}")
        r = json.loads(arquivo.read_text())
        assert r["configuracao"] == c
        resultados.append(r)
    return resultados


def agregar(resultados):
    grupos = defaultdict(list)
    for r in resultados:
        c = r["configuracao"]
        grupos[(c["bloco"], c["familia"], c["h"])].append(r)
    tabela = []
    for (bloco, familia, h), rs in grupos.items():
        rs.sort(key=lambda r: r["seed"])
        assert [r["seed"] for r in rs] == [0, 1, 2]
        assert len({r["train_flops"] for r in rs}) == 1
        assert len({r["infer_flops"] for r in rs}) == 1
        media = lambda campo: statistics.mean(r[campo] for r in rs)
        tabela.append(dict(bloco=bloco, familia=familia, h=h,
                           largura=rs[0]["configuracao"]["largura"], n=3,
                           acuracia_media=media("score"),
                           acuracia_dp=statistics.stdev(r["score"] for r in rs),
                           acuracia_treino=statistics.mean(r["extra"]["acc_train"] for r in rs),
                           acuracia_validacao=statistics.mean(r["extra"]["acc_val"] for r in rs),
                           treino_gflops=media("train_flops")/1e9,
                           inferencia_mflops=media("infer_flops")/1e6,
                           parametros=rs[0]["extra"]["n_params"], segundos_media=media("seconds")))
    return tabela, grupos


def comparar(tabela, grupos):
    lookup = {(r["bloco"], r["familia"], r["h"]): r for r in tabela}
    comparacoes = []
    for bloco in BLOCOS:
        for h in [16, 32, 64, 128]:
            for controle in (["denso", "denso_teto", "soma"] if bloco == "completo" else ["denso", "denso_teto"]):
                a = lookup[(bloco, "estrela", h)]
                b = lookup[(bloco, controle, h)]
                delta = [100*(ra["score"]-rb["score"]) for ra, rb in
                         zip(grupos[(bloco,"estrela",h)], grupos[(bloco,controle,h)])]
                media, dp = statistics.mean(delta), statistics.stdev(delta)
                margem = 4.302652729911275 * dp / np.sqrt(3)
                sobreposicao = not (a["acuracia_media"]-a["acuracia_dp"] > b["acuracia_media"]+b["acuracia_dp"] or b["acuracia_media"]-b["acuracia_dp"] > a["acuracia_media"]+a["acuracia_dp"])
                if not sobreposicao and media-margem > 0:
                    leitura = "vantagem clara da estrela no protocolo"
                elif not sobreposicao and media+margem < 0:
                    leitura = "vantagem clara do controle no protocolo"
                else:
                    leitura = "inconclusivo com três sementes"
                comparacoes.append(dict(bloco=bloco, h=h, controle=controle,
                    delta_s0_pp=delta[0], delta_s1_pp=delta[1], delta_s2_pp=delta[2],
                    delta_media_pp=media, delta_dp_pp=dp, ic95_inferior_pp=media-margem,
                    ic95_superior_pp=media+margem, barras_sobrepostas=sobreposicao,
                    razao_treino=a["treino_gflops"]/b["treino_gflops"],
                    razao_inferencia=a["inferencia_mflops"]/b["inferencia_mflops"],
                    interpretacao=leitura))
    return comparacoes


def analisar_hipoteses(comparacoes):
    """Aplica a regra registrada sem trocar o critério após observar os dados."""
    primarias=[r for r in comparacoes if r["bloco"]=="completo" and
               r["controle"]=="denso" and r["h"] in (32,64)]
    h1=all(r["delta_media_pp"] >= .3 and not r["barras_sobrepostas"] for r in primarias)
    mecanisticas=[]
    for h in [16,32,64,128]:
        for controle in ["denso","denso_teto"]:
            cat=next(r for r in comparacoes if r["bloco"]=="categorico" and r["h"]==h and r["controle"]==controle)
            cont=next(r for r in comparacoes if r["bloco"]=="continuo" and r["h"]==h and r["controle"]==controle)
            dif=[cat[f"delta_s{s}_pp"]-cont[f"delta_s{s}_pp"] for s in [0,1,2]]
            media,dp=statistics.mean(dif),statistics.stdev(dif)
            margem=4.302652729911275*dp/np.sqrt(3)
            mecanisticas.append(dict(h=h,controle=controle,diferenca_das_diferencas_pp=media,
                dp_pp=dp,ic95_inferior_pp=media-margem,ic95_superior_pp=media+margem,
                s0_pp=dif[0],s1_pp=dif[1],s2_pp=dif[2]))
    # Contraste exploratório, acrescentado após observar a grade principal.
    # Não modifica a hipótese H1 nem seu critério registrado.
    extremos=[next(r for r in comparacoes if r['bloco']=='completo' and
                   r['controle']=='denso' and r['h']==h) for h in (16,128)]
    mudancas=[extremos[1][f'delta_s{s}_pp']-extremos[0][f'delta_s{s}_pp'] for s in (0,1,2)]
    media,dp=statistics.mean(mudancas),statistics.stdev(mudancas)
    margem=4.302652729911275*dp/np.sqrt(3)
    largura=dict(exploratorio=True,h_menor=16,h_maior=128,media_pp=media,dp_pp=dp,
                 ic95_inferior_pp=media-margem,ic95_superior_pp=media+margem,
                 s0_pp=mudancas[0],s1_pp=mudancas[1],s2_pp=mudancas[2])
    return dict(h1_atendida_pela_regra_registrada=h1, contrastes_primarios=primarias,
                h2_contrastes=mecanisticas, mudanca_entre_larguras_extremas=largura)


def salvar_figura(fig, nome):
    fig.savefig(PASTA / "figuras" / f"{nome}.png", dpi=180, bbox_inches="tight")
    fig.savefig(PASTA / "figuras" / f"{nome}.pdf", bbox_inches="tight")
    plt.close(fig)


def gerar_figuras(tabela, grupos, comparacoes):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for custo, unidade in [("infer_flops", "Inferência no teste (FLOPs; escala log)"),
                           ("train_flops", "Treino com validação (FLOPs; escala log)")]:
        registros = []
        for (bloco, familia, h), rs in grupos.items():
            if bloco != "completo" or familia not in ("denso", "estrela"):
                continue
            for r in rs:
                registros.append(Result(name=f'{NOMES[familia].replace(" 2h", "")} {r["configuracao"]["largura"]}',
                     seed=r["seed"], score=100*r["score"], train_flops=r["train_flops"], infer_flops=r["infer_flops"]))
        ax = pareto(registros, cost=custo, title="Adult: acurácia e custo — média ± DP", annotate=True)
        ax.set_xlabel(unidade)
        ax.set_ylabel("Acurácia no teste (%)")
        for anotacao in ax.texts:
            anotacao.set_position((7,-15) if anotacao.get_text().startswith('Estrela') else (7,10))
        for conjunto in ax.containers:
            cor=CORES['estrela'] if conjunto.get_label().startswith('Estrela') else CORES['denso']
            linha,tampas,barras=conjunto.lines
            linha.set_color(cor)
            for artista in (*tampas,*barras):artista.set_color(cor)
            for tampa in tampas:
                tampa.set_markeredgecolor(cor)
        # A fronteira é calculada sobre médias; não indica dominância estatística.
        pontos = sorted({(getattr(r,custo), statistics.mean(x.score for x in registros if x.name==r.name)) for r in registros})
        fronteira, melhor = [], -float("inf")
        for x, y in pontos:
            if y > melhor:
                fronteira.append((x,y)); melhor = y
        ax.plot(*zip(*fronteira), color="#333333", linestyle="--", marker='D',
                markerfacecolor='none',markersize=7,alpha=.7, linewidth=1,
                label="Fronteira das médias (descritiva)")
        # Os rótulos já identificam cada largura; a legenda não deve cobrir pontos.
        legenda = [Line2D([], [], color=CORES[f], marker='o', linestyle='none',
                           label=rotulo)
                   for f, rotulo in [('denso', 'Denso 2h'), ('estrela', 'Estrela h')]]
        legenda.append(Line2D([], [], color='#333333', linestyle='--', marker='D',
                              markerfacecolor='none', label='Fronteira das médias (descritiva)'))
        ax.legend(handles=legenda, loc="lower left", fontsize=9)
        salvar_figura(ax.figure, "pareto_" + ("inferencia" if custo=="infer_flops" else "treino"))

    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.25), sharey=True)
    for ax, bloco in zip(eixos, BLOCOS):
        for familia in ["denso", "estrela", "denso_teto"]:
            rs = [r for r in tabela if r["bloco"]==bloco and r["familia"]==familia]
            ax.errorbar([r["h"] for r in rs], [100*r["acuracia_media"] for r in rs],
                        yerr=[100*r["acuracia_dp"] for r in rs], marker="o", capsize=3,
                        label=NOMES[familia], color=CORES[familia])
        ax.set_title(BLOCOS[bloco]); ax.set_xlabel("Largura de referência h")
        ax.set_xscale("log",base=2); ax.set_xticks([16,32,64,128], [16,32,64,128])
    eixos[0].set_ylabel("Acurácia no teste (%)")
    eixos[-1].legend(fontsize=10)
    fig.suptitle("Ablações: média ± DP de três sementes")
    fig.tight_layout(); salvar_figura(fig, "ablacoes")

    fig, ax = plt.subplots(figsize=(8,4.5))
    for familia in ["denso", "estrela", "soma", "glu", "geglu", "swiglu"]:
        rs = [r for r in tabela if r["bloco"]=="completo" and r["familia"]==familia]
        ax.errorbar([r["h"] for r in rs], [100*r["acuracia_media"] for r in rs],
                    yerr=[100*r["acuracia_dp"] for r in rs], marker="o", capsize=3,
                    label=NOMES[familia], color=CORES[familia])
    ax.set_xscale("log",base=2); ax.set_xticks([16,32,64,128],[16,32,64,128])
    ax.set_xlabel("Largura h dos ramos (denso usa 2h)"); ax.set_ylabel("Acurácia no teste (%)")
    ax.set_title("Combinação e ativação dos ramos — média ± DP")
    ax.legend(ncol=3,fontsize=9); fig.tight_layout(); salvar_figura(fig,"variantes")

    fig, eixos = plt.subplots(1,2,figsize=(9,3.7))
    for ax,campo,titulo in zip(eixos,["train_hist","val_hist"],["Treino", "Validação"]):
        for familia in ["denso", "estrela", "soma"]:
            curvas = np.array([r["extra"][campo] for r in grupos[("completo",familia,64)]])
            x = np.arange(1,101); mu=curvas.mean(axis=0); dp=curvas.std(axis=0,ddof=1)
            ax.plot(x,mu,label=NOMES[familia],color=CORES[familia])
            ax.fill_between(x,mu-dp,mu+dp,color=CORES[familia],alpha=.15)
        ax.set_title(titulo); ax.set_xlabel("Época"); ax.set_ylabel("Entropia cruzada")
        ax.set_ylim(bottom=0)
    eixos[-1].legend(fontsize=10); fig.suptitle("Curvas em h=64: média ± DP")
    fig.tight_layout(); salvar_figura(fig,"curvas")

    fig, ax = plt.subplots(figsize=(8,4))
    for j, controle in enumerate(["denso","denso_teto","soma"]):
        rs=[r for r in comparacoes if r["bloco"]=="completo" and r["controle"]==controle]
        x=np.arange(4)+(j-1)*.18
        ax.errorbar(x,[r["delta_media_pp"] for r in rs],
                    yerr=[r["ic95_superior_pp"]-r["delta_media_pp"] for r in rs],
                    fmt="o", capsize=4, label=f"Estrela − {NOMES[controle]}",color=CORES[controle])
    ax.axhline(0,color="black",linewidth=.8); ax.set_xticks(range(4),[16,32,64,128])
    ax.set_xlabel("Largura de referência h"); ax.set_ylabel("Diferença de acurácia (p.p.)")
    ax.set_title("Diferenças pareadas: média e IC t de 95% (n=3)")
    ax.legend(fontsize=8); fig.tight_layout(); salvar_figura(fig,"diferencas")


def main():
    (PASTA/"figuras").mkdir(exist_ok=True)
    resultados=carregar()
    tabela,grupos=agregar(resultados)
    comparacoes=comparar(tabela,grupos)
    hipoteses=analisar_hipoteses(comparacoes)
    gravar_csv(PASTA/"resultados/resumo.csv",tabela)
    gravar_csv(PASTA/"resultados/comparacoes.csv",comparacoes)
    gravar_csv(PASTA/"resultados/hipotese_mecanistica.csv",hipoteses["h2_contrastes"])
    individuais=[]
    for r in resultados:
        individuais.append({**r["configuracao"], "acuracia_teste":r["score"],
            "acuracia_treino":r["extra"]["acc_train"], "acuracia_validacao":r["extra"]["acc_val"],
            "flops_treino":r["train_flops"], "flops_inferencia":r["infer_flops"], "segundos":r["seconds"]})
    gravar_csv(PASTA/"resultados/execucoes.csv",individuais)
    res=[Result(**{k:r[k] for k in ["name","seed","score","train_flops","infer_flops","seconds","extra"]}) for r in resultados]
    resumo=summarize(res)  # Sem o veredicto automático baseado somente no DP da variante.
    cabecalho=(f"{'configuração':<28}{'n':>3}  {'acurácia ± DP':>16}  {'treino GFLOP':>12}"
               f"  {'infer. MFLOP':>12}  {'parâmetros':>9}  {'s/exec':>7}")
    resumo="\n".join([cabecalho, *resumo.splitlines()[1:]])
    (PASTA/"resultados/tabela_summarize.txt").write_text(resumo+"\n")
    gerar_figuras(tabela,grupos,comparacoes)
    resumo_json=dict(configuracoes=tabela,comparacoes=comparacoes,hipoteses=hipoteses,
                     total_execucoes=len(resultados), soma_tempo_treinos_s=sum(r["seconds"] for r in resultados),
                     total_flops_treino=sum(r["train_flops"] for r in resultados))
    (PASTA/"resultados/analise.json").write_text(json.dumps(resumo_json,ensure_ascii=False,indent=2)+"\n")
    # Leitura incremental do monitor: o volume do log não determina a memória usada.
    max_worker=0;max_conjunto=0;min_disponivel=float('inf');primeiro=None;ultimo=None
    with (PASTA/'resultados/monitoramento_memoria.jsonl').open() as f:
        for s in f:
            m=json.loads(s)
            if primeiro is None:primeiro=m
            ultimo=m
            max_worker=max(max_worker,m['worker_bytes'])
            max_conjunto=max(max_conjunto,m['worker_bytes']+m['vscode_bytes'])
            min_disponivel=min(min_disponivel,m['disponivel'])
    memoria=dict(pico_worker_monitorado_bytes=max_worker,pico_conjunto_monitorado_bytes=max_conjunto,
        disponivel_minimo_bytes=min_disponivel,pico_cgroup_treinos_bytes=max(r.get('pico_memoria_bytes',0) for r in resultados),
        paginas_swap_saida_adicionais=ultimo['swap_saida']-primeiro['swap_saida'],
        execucoes_protegidas=sum(r.get('ambiente_versao')=='seguro' for r in resultados))
    (PASTA/'resultados/resumo_recursos.json').write_text(json.dumps(memoria,ensure_ascii=False,indent=2)+'\n')
    maior=max(resultados,key=lambda r:r.get('pico_memoria_bytes',0))
    referencia=json.loads((PASTA/'resultados/equivalencia_memoria.json').read_text())
    fig,ax=plt.subplots(figsize=(7,3.8))
    for nome,r in [('Referência densa 64',referencia),(maior['name'],maior)]:
        pontos=r['memoria_epocas']
        ax.plot([v['epoca'] for v in pontos],[v['bytes']/1024**3 for v in pontos],marker='o',label=nome)
    ax.axhline(3,color='#ad3838',linestyle='--',label='Limite do serviço (3 GiB)')
    ax.set_xlabel('Época');ax.set_ylabel('Memória do cgroup após a coleta (GiB)')
    ax.set_title('Memória por época nas execuções protegidas')
    ax.legend(fontsize=8);fig.tight_layout();salvar_figura(fig,'memoria_epocas')
    print(f"Análise concluída: {len(tabela)} configurações, {len(resultados)} execuções.")
    for r in comparacoes:
        if r["bloco"]=="completo":
            print(f'h={r["h"]:3d}; estrela − {r["controle"]:10s}: {r["delta_media_pp"]:+.3f} p.p.; IC95 [{r["ic95_inferior_pp"]:+.3f}, {r["ic95_superior_pp"]:+.3f}]; {r["interpretacao"]}')


if __name__=="__main__":
    main()
