"""Produz relatório em Markdown/LaTeX/PDF e notebook com resultados executados."""
from datetime import datetime, timedelta
import hashlib
import io
import contextlib
import json
from pathlib import Path
import re
import subprocess
import sys
if __package__:
    from .recursos_documentos import executar_limitado
else:
    from recursos_documentos import executar_limitado

PASTA = Path(__file__).resolve().parent


def numero(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def escapar(s):
    mapa = {"&":r"\&", "%":r"\%", "$":r"\$", "#":r"\#", "_":r"\_",
            "{":r"\{", "}":r"\}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}"}
    return "".join(mapa.get(c,c) for c in str(s))


def inline(s):
    partes=re.split(r"(\\\(.*?\\\))",str(s))
    for i,p in enumerate(partes):
        if p.startswith(r"\("):
            continue
        p=escapar(p)
        p=re.sub(r"\*\*(.*?)\*\*",r"\\textbf{\1}",p)
        p=re.sub(r"`([^`]+)`",lambda m: r"\texttt{"+
                 m[1].replace("/",r"/\allowbreak{}").replace(r"\_",r"\_\allowbreak{}")+"}",p)
        partes[i]=p
    return "".join(partes)


class Documento:
    def __init__(self):
        self.md=["# Q5 — A operação estrela compensa seu custo em Adult?\n\nEstudo experimental • 13 de outubro de 2026\n"]
        self.tex=[]

    def secao(self,n,titulo):
        self.md.append(f"## {n}. {titulo}\n")
        self.tex.append(r"\section*{"+f"{n}. {escapar(titulo)}"+"}")

    def sub(self,titulo):
        self.md.append(f"### {titulo}\n")
        self.tex.append(r"\subsection*{"+escapar(titulo)+"}")

    def p(self,texto):
        self.md.append(texto+"\n")
        self.tex.append(inline(texto)+"\n\n")

    def equacao(self,eq):
        self.md.append("\\[\n"+eq+"\n\\]\n")
        self.tex.append("\\[\n"+eq+"\n\\]\n")

    def tabela(self,titulo,cabecalho,linhas):
        self.md.append("**"+titulo+"**\n\n| "+" | ".join(cabecalho)+" |\n| "+" | ".join(["---"]*len(cabecalho))+" |\n"+
            "\n".join("| "+" | ".join(str(c) for c in linha)+" |" for linha in linhas)+"\n")
        self.tex.append(r"\noindent\textbf{"+escapar(titulo)+"}\n\n"+r"{\footnotesize\setlength{\tabcolsep}{4pt}\begin{center}\begin{tabular}{"+
            "l"+"r"*(len(cabecalho)-1)+r"}\toprule"+"\n"+" & ".join(inline(c) for c in cabecalho)+r"\\\midrule"+"\n"+
            "\n".join(" & ".join(inline(c) for c in linha)+r"\\" for linha in linhas)+"\n"+r"\bottomrule\end{tabular}\end{center}}"+"\n")

    def figura(self,nome,legenda,largura=.95):
        self.md.append(f"![{legenda}](figuras/{nome}.png)\n\n{legenda}\n")
        self.tex.append(r"\begin{center}\includegraphics[width="+str(largura)+r"\linewidth]{figuras/"+nome+r".pdf}"+"\n"+r"\par\small "+inline(legenda)+r"\end{center}"+"\n")

    def codigo(self,texto):
        self.md.append("```bash\n"+texto+"\n```\n")
        self.tex.append("{\\small\n\\begin{verbatim}\n"+texto+"\n\\end{verbatim}\n}\n")

    def resumo(self,texto):
        self.md.append("```text\n"+texto.rstrip()+"\n```\n")
        self.tex.append("{\\fontsize{8.5}{10.2}\\selectfont\n\\begin{verbatim}\n"+
                        texto.rstrip()+"\n\\end{verbatim}\n}\n")

    def pagina(self):
        self.tex.append(r"\clearpage"+"\n")


def construir():
    a=json.loads((PASTA/"resultados/analise.json").read_text())
    audit=json.loads((PASTA/"resultados/auditoria_final.json").read_text())
    v=json.loads((PASTA/"resultados/verificacao.json").read_text())
    ambiente=json.loads((PASTA/"resultados/ambiente.json").read_text())
    ref=json.loads((PASTA/"resultados/referencia_original.json").read_text())
    equivalencia=json.loads((PASTA/'resultados/equivalencia_memoria.json').read_text())
    historicos=audit['execucoes_historicas']
    # Na reprodução em pasta vazia, a mesma execução estabelece a referência e
    # verifica sua equivalência. Seu tempo e seus FLOPs só entram uma vez.
    referencia_compartilhada=(historicos == 0 and ref['seconds'] == equivalencia['seconds'])
    extra_equivalencia=0 if referencia_compartilhada else equivalencia['train_flops']
    assert audit["aprovado"] and a["total_execucoes"]==156
    tabela=a["configuracoes"]; comparacoes=a["comparacoes"]
    def linha(f,h,bloco="completo"):
        return next(r for r in tabela if r["familia"]==f and r["h"]==h and r["bloco"]==bloco)
    def comp(h,controle="denso",bloco="completo"):
        return next(r for r in comparacoes if r["h"]==h and r["controle"]==controle and r["bloco"]==bloco)
    def media_dp(r):
        return numero(100*r["acuracia_media"])+" ± "+numero(100*r["acuracia_dp"])
    nomes={"denso":"Denso 2h", "estrela":"Estrela", "soma":"Soma", "glu":"GLU", "geglu":"GEGLU", "swiglu":"SwiGLU", "denso_teto":"Denso no teto"}
    doc=Documento()
    doc.secao(1,"A pergunta")
    doc.p("A Q5 pergunta se uma camada estrela, formada pelo produto elemento a elemento de dois ramos afins, oferece mais acurácia por FLOP que uma camada densa ReLU mais larga, e se a resposta muda com a largura. Investigamos a classificação de renda no Adult com o motor NumPy/CPU do BERT-CPU, pareando estrela de largura h e denso de largura 2h. Mantemos épocas, otimizador, taxa e sementes. ‘FLOPs iguais’ é uma aproximação, pois a rede inteira inclui a saída; publicamos os custos efetivos, um controle de teto de inferência e a comparação soma versus produto com os mesmos dois ramos.")
    doc.secao(2,"O que a literatura diz")
    doc.p("**Ma et al. [1] — Rewrite the Stars.** A multiplicação de projeções cria interações entre atributos sem construir explicitamente a expansão de monômios. Em redes de visão, o artigo observa ganhos ao substituir soma por produto e sugere maior utilidade em redes estreitas. A previsão transferida para Adult é uma vantagem da estrela nas larguras pequenas. Essa previsão é uma inferência: o trabalho usa blocos de visão, convoluções e outra escala; não mede esta MLP tabular. O controle Soma aproxima a ablação do artigo, enquanto Denso 2h responde à pergunta específica do notebook.")
    doc.p("**Shazeer [2] — GLU Variants Improve Transformer.** ReGLU tem a mesma forma algébrica da nossa estrela-ReLU; GEGLU e SwiGLU trocam a ativação de um ramo. O trabalho observa melhor perplexidade para algumas variantes no T5 e ajusta larguras para manter aproximadamente custo e parâmetros. Isso motiva nossa extensão com GLU, GEGLU e SwiGLU, mas não estabelece uma ordem universal de acurácia em Adult. Estrela e ReGLU não são contadas como arquiteturas diferentes.")
    doc.p("**Dauphin et al. [3] — Gated Convolutional Networks.** A GLU combina um ramo linear com uma porta sigmoide. O caminho linear dispensa uma derivada adicional de ativação naquele ramo, favorecendo a propagação de gradientes nos modelos estudados. A extrapolação verificável é que a porta afete convergência e estabilidade. A vantagem de paralelização em linguagem não é uma previsão de velocidade para nossas MLPs em CPU.")
    doc.p("**Grinsztajn et al. [4] — Modelos tabulares.** Em 45 bases, o estudo analisa orientação dos atributos, atributos pouco informativos e irregularidade das funções como fatores relevantes para a vantagem de árvores. Produtos entre features podem ajudar algumas interações, mas não eliminam esses fatores. Em indicadores one-hot, quadrados repetem o próprio indicador e produtos entre categorias exclusivas são zero. As ablações por tipo de atributo testam compatibilidade com a explicação por interações; não identificam sua causa nem comparam a estrela com árvores, que não foram treinadas aqui.")
    doc.p("As quatro fontes primárias, suas previsões e os limites de extrapolação estão detalhados em `literatura.md`; as referências completas aparecem ao final. A pergunta permanece empírica: capacidade representacional adicional não garante melhor generalização no mesmo protocolo de otimização.")
    doc.p(r"Há ainda uma particularidade matemática deste domínio: para indicadores binários, \(\operatorname{ReLU}(x_i+x_j-1)=x_i x_j\). Assim, uma unidade ReLU comum já representa a coocorrência de dois indicadores. A derivada mista nula dentro de regiões contínuas não impede representar essa interação no domínio discreto. O produto explícito não garante uma vantagem exclusiva nas categorias.")

    doc.pagina()
    doc.secao(3,"Hipótese")
    doc.p("**H1, registrada antes dos experimentos:** em h=32 e h=64, estrela-ReLU supera denso-2h por pelo menos 0,3 ponto percentual na acurácia média de teste, com separação dos intervalos média ± desvio-padrão amostral. A condição é conjuntiva: precisa ocorrer nas duas larguras. O limiar é uma escolha operacional deste estudo, não uma previsão numérica dos artigos.")
    doc.p("**H2, secundária:** a vantagem estrela−denso é maior usando somente indicadores categóricos que usando somente atributos contínuos. Para examinar essa direção, calculamos também a diferença das diferenças por semente. Um contraste positivo é compatível com a hipótese; três sementes e mudanças na informação disponível impedem uma conclusão causal.")
    doc.p("O texto em `protocolo.md` foi escrito antes do primeiro treino. Seu SHA-256 começa por `"+ambiente["protocolo_sha256"][:16]+"`; o valor integral e os hashes do código e dos dados estão em `resultados/ambiente.json`. Esse registro local torna alterações detectáveis dentro da entrega, mas não equivale a um pré-registro externo independente.")
    doc.secao(4,"O que foi implementado")
    origem=(f"Os primeiros {historicos} treinos pertencem à execução histórica; os demais usam os controles descritos em `adendo_recursos.md`." if historicos else "Todos os treinos usam os controles descritos em `adendo_recursos.md`.")
    doc.p("`modelos.py` implementa seis famílias com operações já diferenciadas pelo motor: denso, estrela/ReGLU, soma, GLU, GEGLU e SwiGLU. O denso no teto reutiliza a família densa com largura calculada. Os ramos usam Xavier, vieses zero e a mesma saída. `executar.py` direciona ao executor sequencial protegido, que chama `run_adult` sem alterar o instrumento. O hook coleta grafos a cada época, sem mudar operações numéricas; a referência e suas curvas foram novamente verificadas. "+origem)
    doc.p("As portas são ReLU(u), sigmoide(u), GELU(u) e u·sigmoide(u). GELU usa a aproximação por tanh do motor; SwiGLU fixa β=1. Na soma, combinamos ReLU(u)+v; nas portas, g(u)·v. Com entrada de dimensão d e lote B, os ramos têm pesos (d+1)×h e ativações h×B; a saída tem pesos (h+1)×2. A seleção preserva a ordem do carregador: seis contínuos, depois 102 indicadores.")
    doc.equacao(r"u=W_1^\top\widetilde X,\quad v=W_2^\top\widetilde X,\quad A_{\star}=\operatorname{ReLU}(u)\odot v.")
    doc.p("X̃ acrescenta uma linha de uns sobre X; cada exemplo é uma coluna e o viés ocupa a primeira linha dos pesos. A saída produz dois logits, usados com softmax e entropia cruzada categórica. As derivadas locais da estrela são:")
    doc.equacao(r"\frac{\partial J}{\partial u}=\frac{\partial J}{\partial A}\odot v\odot\mathbf{1}_{u>0},\qquad \frac{\partial J}{\partial v}=\frac{\partial J}{\partial A}\odot\operatorname{ReLU}(u).")
    doc.equacao(r"\frac{\partial J}{\partial W_1}=\widetilde X\left(\frac{\partial J}{\partial u}\right)^\top,\qquad \frac{\partial J}{\partial W_2}=\widetilde X\left(\frac{\partial J}{\partial v}\right)^\top.")
    doc.p("Na região u>0, um neurônio estrela produz (aᵀx̃)(bᵀx̃), incluindo termos cruzados. Com ReLU, o resultado é quadrático por partes. Os coeficientes não são todos independentes: a matriz quadrática simétrica (abᵀ+baᵀ)/2 tem posto no máximo dois por unidade. Uma MLP ReLU comum também produz dependências entre atributos, mas seus logits são afins dentro de cada região de ativação.")
    doc.p(f"**Verificação:** {v['testes_executados']} testes aprovados; {v['coordenadas_verificadas']} coordenadas de entradas, pesos e vieses comparadas com diferenças centrais de passo 10⁻⁶. Erro absoluto máximo: {v['erro_absoluto_maximo']:.4e}. O oráculo de forward é NumPy independente, com pontos afastados da fronteira da ReLU. Também verificamos gradientes nulos nos atributos excluídos, equivalência do denso com a referência e fórmulas de custo. Não foi criado um novo operador primitivo de autograd.")
    doc.p("Os arquivos de análise, auditoria e geração do relatório completam a implementação. A tabela de linhas de código na seção 9 explicita o tamanho da entrega. Resultados individuais, pesos finais e curvas de 100 épocas acompanham todas as configurações, permitindo inspecionar cada número das tabelas.")

    doc.pagina()
    doc.secao(5,"Protocolo")
    doc.tabela("Tabela 1 — Condições fixadas",["Item","Valor"],[
        ["Treino / validação / teste","26.049 / 6.512 / 16.281"],["Entradas completas","6 contínuas + 102 indicadores"],
        ["Larguras h","16, 32, 64, 128"],["Sementes","0, 1, 2"],["Épocas / lote","100 / treino inteiro"],
        ["Otimizador / taxa","Adam / 0,01"],["Betas / epsilon","0,9; 0,999 / 10⁻⁸"],["Precisão / decaimento","float64 / zero"],
        ["Grade / verificações adicionais","156 + referência/equivalência conjunta" if referencia_compartilhada else "156 + referência + equivalência"]])
    doc.p("O treino oficial tem 32.561 exemplos. A mesma permutação por semente separa 20% para validação antes da inicialização das arquiteturas. Não houve ajuste da taxa, escolha de checkpoint, early stopping ou seleção de largura pelo teste. Avaliamos todos os modelos após a época 100. A semente compartilhada pareia as divisões, mas não gera pesos idênticos em arquiteturas de formas diferentes. O DP reflete simultaneamente mudança de divisão e inicialização.")
    doc.p("No bloco completo, a grade contém sete configurações por largura: as seis famílias e o denso no teto. Nos blocos contínuo e categórico, contém estrela, denso-2h e denso no teto. São 84 treinos completos e 72 de ablação. A hipótese principal usa o pareamento h:2h; o controle soma mantém exatamente parâmetros e custo contado; as demais comparações são extensões declaradas, publicadas integralmente.")
    doc.equacao(r"C_{\rm denso}(m)=(2d+7)m+4,\qquad C_{\star}(h)=C_{\rm soma}(h)=(4d+10)h+4.")
    doc.p("Essas fórmulas dão o custo de inferência por exemplo conforme o motor, incluindo as duas classes, vieses, ReLU e combinação. Multiplicamos por 16.281 para obter o custo de uma passagem no teste. Em d=108, estrela custa 442h+4 e denso-2h custa 446h+4: cerca de 0,90% de diferença. Com seis contínuos, são 34h+4 e 38h+4, diferença próxima de 10,5%. Assim, o controle de teto é particularmente relevante nessa ablação.")
    doc.equacao(r"m_{\rm teto}=\left\lfloor\frac{(4d+10)h}{2d+7}\right\rfloor.")
    doc.p("O controle escolhe a maior largura densa inteira que cabe no custo da estrela. Trata-se do mesmo teto de inferência, com custo usado ligeiramente diferente por arredondamento; não de igualdade artificial obtida adicionando operações inúteis. Para treino T e validação V, os custos por época são:")
    doc.equacao(r"F_d=[(4d+17)m+26]T+[(2d+7)m+18]V+4,")
    doc.equacao(r"F_\star=[(8d+22)h+26]T+[(4d+10)h+18]V+4.")
    doc.p("O contador inclui forward, backward dos produtos matriciais e forward/perda de validação; omite a aritmética NumPy do Adam e o backward elementar. Reportamos essa convenção, não o total físico exato. O orçamento fixado é de épocas; treino e inferência não são intercambiáveis. As variantes GLU também pagam o custo contado de suas ativações.")
    doc.p("Reportamos média ± DP amostral (ddof=1). Para a diferença de acurácia, subtraímos resultados com a mesma semente e calculamos IC t de 95%, com 2 graus de liberdade. Esse IC é descritivo, depende de aproximação de normalidade e não corrige múltiplas comparações. A decisão de H1 segue sua regra registrada. A classificação adicional ‘vantagem clara’ exige separação das barras e IC que exclua zero; sobreposição não prova equivalência. Usamos `summarize()` sem seu diagnóstico automático e `pareto()` com fronteira das médias explicitamente descritiva.")

    doc.pagina()
    doc.secao(6,"Resultados")
    doc.p(f"**Referência reproduzida.** O denso 64 com semente 0 atingiu {numero(100*ref['score'],4)}% no teste, {numero(ref['train_flops']/1e9,7)} GFLOPs de treino e {numero(ref['infer_flops']/1e6,6)} MFLOPs de inferência. O denso 64 da grade (h=32) reproduziu exatamente essa acurácia e esses custos. Sua média nas três sementes é {media_dp(linha('denso',32))}%. A auditoria reconstruiu três inferências a partir dos pesos publicados e conferiu custos, parâmetros, curvas e arquivos das 156 execuções.")
    principais=[comp(h) for h in [16,32,64,128]]
    doc.p("**Resposta à pergunta principal.** "+("A hipótese H1 foi atendida pela regra registrada." if a['hipoteses']['h1_atendida_pela_regra_registrada'] else "A hipótese H1 não foi atendida pela regra registrada.")+f" A diferença média estrela−denso-2h varia de {numero(min(r['delta_media_pp'] for r in principais))} a {numero(max(r['delta_media_pp'] for r in principais))} p.p. nas quatro larguras (Tabela 3). A conclusão é condicionada ao protocolo de 100 épocas e Adam com taxa fixa; não estima o melhor desempenho possível de cada arquitetura.")
    doc.p("Em h=16, a pequena diferença positiva não se distingue do ruído. Em h=32, a média favorece o denso, mas o IC pareado ainda inclui zero. Em h=64 e h=128, o denso é favorecido pelos dois critérios declarados. Portanto, o resultado depende da largura: ampliar a estrela não recupera uma vantagem; sua diferença relativa piora nesta grade (Tabela 3).")
    linhas=[]
    for h in [16,32,64,128]:
        for f in ["denso","estrela","soma"]:
            r=linha(f,h)
            linhas.append([str(h),nomes[f]+f" ({r['largura']})",numero(100*r['acuracia_treino']),media_dp(r),numero(r['treino_gflops'],2),numero(r['inferencia_mflops'],2),str(r['parametros'])])
    doc.tabela("Tabela 2 — Dados completos; teste: média ± DP; treino: média",["h","Modelo (larg.)","Treino (%)","Teste (%)","Treino GF","Infer. MF","Parâm."],linhas)
    menor=linha('estrela',16);maior_estrela=linha('estrela',128)
    doc.p(f"Ao ampliar a estrela de 16 para 128, a acurácia média de treino sobe de {numero(100*menor['acuracia_treino'])}% para {numero(100*maior_estrela['acuracia_treino'])}%, enquanto a de teste cai (Tabela 2). Esse padrão é compatível com sobreajuste; não distingue sozinho capacidade excessiva, escala das ativações e configuração de otimização.")
    doc.figura("pareto_inferencia","Figura 1 — Pareto dos modelos principais. Barras: DP; diamantes: fronteira das médias, sem teste de dominância.",.80)
    doc.p("A Tabela 7 reproduz `summarize()` para as 52 configurações. Em `resultados/`, os arquivos `resumo.csv` e `execucoes.csv` detalham médias e execuções individuais. A figura correspondente de treino está em `figuras/pareto_treino.pdf`.")

    doc.pagina()
    doc.sub("Diferenças pareadas, largura e orçamento")
    linhas=[]
    for h in [16,32,64,128]:
        c=comp(h)
        linhas.append([h,numero(c['delta_media_pp']),numero(c['delta_dp_pp']),f"[{numero(c['ic95_inferior_pp'])}; {numero(c['ic95_superior_pp'])}]","sim" if c['barras_sobrepostas'] else "não"])
    doc.tabela("Tabela 3 — Estrela menos denso-2h; pontos percentuais",["h","Média Δ","DP Δ","IC t 95%","Barras sobrepostas"],linhas)
    mudanca=a['hipoteses']['mudanca_entre_larguras_extremas']
    doc.p(f"**Dependência da largura.** O contraste pareado (estrela−denso) em h=128 menos o mesmo contraste em h=16 vale {numero(mudanca['media_pp'])} p.p., com IC t de 95% [{numero(mudanca['ic95_inferior_pp'])}; {numero(mudanca['ic95_superior_pp'])}] (linhas extremas da Tabela 3; valores por semente em `analise.json`). Ele sugere piora relativa ao aumentar a largura. Essa análise é exploratória, acrescentada após observar a grade principal, sem alterar H1; mantém as limitações de três sementes e comparações múltiplas.")
    c=comp(64)
    doc.p(f"No bloco completo, a estrela usa aproximadamente {numero(100*c['razao_inferencia'],2)}% dos FLOPs de inferência e {numero(100*c['razao_treino'],2)}% dos FLOPs de treino do denso-2h nas quatro larguras; as pequenas variações decorrem dos termos constantes das fórmulas da seção 5. A Tabela 3 quantifica a incerteza da diferença de acurácia; a Tabela 4 verifica a sensibilidade ao teto de custo e à combinação dos ramos.")
    linhas=[]
    for h in [16,32,64,128]:
        r=linha('denso_teto',h);c=comp(h,'denso_teto');s=comp(h,'soma')
        linhas.append([h,r['largura'],media_dp(r),numero(r['treino_gflops'],2),numero(r['inferencia_mflops'],2),numero(c['delta_media_pp']),numero(s['delta_media_pp'])])
    doc.tabela("Tabela 4 — Controles no bloco completo",["h","Larg.","Teste teto (%)","Treino GF","Infer. MF","Δ teto","Δ soma"],linhas)
    doc.p("Δ teto e Δ soma significam estrela menos o respectivo controle. A comparação com soma isola a operação de combinação dentro do desenho adotado, mas também muda escala e condicionamento das ativações. Todas as diferenças individuais, seus DPs e ICs estão em `comparacoes.csv`. O arredondamento da largura no teto não permite alegar igualdade exata de custo usado.")
    doc.p("O denso no teto também é favorecido em h=64 e h=128 pelos dois critérios; h=16 e h=32 permanecem inconclusivos pela regra conjunta (Tabela 4 e `comparacoes.csv`). Assim, a leitura principal nas larguras maiores persiste ao restringir a inferência densa ao teto da estrela.")
    soma_favorecida=[str(h) for h in [16,32,64,128] if comp(h,'soma')['interpretacao']=='vantagem clara do controle no protocolo']
    soma_inconclusiva=[str(h) for h in [16,32,64,128] if comp(h,'soma')['interpretacao']=='inconclusivo com três sementes']
    doc.p("**Produto versus soma.** Com os mesmos parâmetros e FLOPs contados, a soma é favorecida pelos critérios registrados em h="+', '.join(soma_favorecida)+"; o contraste é inconclusivo em h="+', '.join(soma_inconclusiva)+" (Tabela 4 e ICs em `comparacoes.csv`). A previsão de vantagem multiplicativa transferida de Ma et al. não foi corroborada nestes pares. Isso delimita a transferência para esta MLP tabular; não refuta os resultados do artigo em visão.")
    doc.figura("curvas","Figura 2 — Curvas em h=64. Treino é medido antes da atualização; validação, depois. Faixas: média ± DP.",.90)
    doc.p("As curvas ajudam a distinguir ajuste ao treino de desempenho fora dele. Como a taxa não foi ajustada por família, uma diferença de convergência pode refletir otimização e escala das ativações. A curva de validação não foi usada para escolher a época final; todos os resultados publicados são da época 100.")

    doc.pagina()
    doc.sub("Portas e ablações dos atributos")
    linhas=[]
    for h in [16,32,64,128]:
        for f in ['glu','geglu','swiglu']:
            r=linha(f,h)
            linhas.append([h,nomes[f],media_dp(r),numero(r['treino_gflops'],2),numero(r['inferencia_mflops'],2)])
    doc.tabela("Tabela 5 — Portas adicionais; teste (%) ± DP",["h","Família","Teste (%)","Treino GF","Infer. MF"],linhas)
    maior_media=max(r['acuracia_media'] for r in tabela if r['bloco']=='completo')
    melhores=[r for r in tabela if r['bloco']=='completo' and abs(r['acuracia_media']-maior_media)<1e-12]
    rotulos=', '.join(f"{nomes[r['familia']]} em h={r['h']}" for r in melhores)
    doc.p(f"A maior média descritiva do bloco completo ocorre em {rotulos}, com dispersões e custos publicados na Tabela 5. Essa identificação é pós-experimento; não é seleção validada nem prova de superioridade sobre todas as configurações. A ordenação observada não sustenta transferir diretamente para Adult a expectativa de GEGLU/SwiGLU como melhores portas em linguagem. Os dados completos estão em `resumo.csv` e a visualização em `figuras/variantes.pdf`.")
    doc.figura("ablacoes","Figura 3 — Mesmas sementes e épocas; informação disponível muda entre blocos. Barras: DP.",1.0)
    linhas=[]
    for h in [16,32,64,128]:
        c=comp(h,bloco='categorico');t=comp(h,bloco='continuo')
        dd=next(r for r in a['hipoteses']['h2_contrastes'] if r['h']==h and r['controle']=='denso')
        linhas.append([h,numero(c['delta_media_pp']),numero(t['delta_media_pp']),numero(dd['diferenca_das_diferencas_pp']),f"[{numero(dd['ic95_inferior_pp'])}; {numero(dd['ic95_superior_pp'])}]"])
    doc.tabela("Tabela 6 — Hipótese mecanística; pontos percentuais",["h","Δ categ.","Δ contín.","Δ categ. − Δ contín.","IC t 95%"],linhas)
    doc.p("**H2 não foi corroborada.** As quatro estimativas da diferença das diferenças são negativas, direção oposta à prevista. Em h=64, o IC também fica inteiramente abaixo de zero; nas demais larguras, inclui zero (Tabela 6). Portanto, não há evidência, neste protocolo, de que a vantagem da estrela se concentre nos indicadores categóricos. Os sinais pontuais nas outras larguras não bastam para afirmar o mecanismo inverso.")
    doc.p("Cada Δ é estrela−denso-2h. A última diferença é calculada dentro de cada semente antes de obter média e IC. Os contrastes contra o denso no teto também são publicados em `hipotese_mecanistica.csv`. Um sinal favorável a H2 não demonstra que a rede aprendeu interações entre categorias: os blocos diferem em dimensão, informação preditiva e custo. Eliminar o bloco contínuo também elimina interações mistas contínuo×categoria.")

    doc.pagina()
    doc.sub("Tabela completa do instrumento de medição")
    doc.p("**Tabela 7 — Saída de summarize(), com cabeçalho traduzido.** Acurácia na escala 0–1; dispersão: DP amostral. GFLOP: treino de 100 épocas; MFLOP: uma inferência no teste inteiro. Cada linha agrega três sementes. A referência é `completo_denso_h32` (largura efetiva 64); o sufixo h nas demais linhas é a largura de referência do pareamento. A tabela inclui os custos de todas as ablações da Figura 3. Segundos por execução misturam sessões históricas concorrentes e sessões sequenciais, portanto não comparam latência entre modelos.")
    doc.resumo((PASTA/'resultados/tabela_summarize.txt').read_text())

    doc.pagina()
    doc.secao(7,"Ameaças à validade")
    doc.p("**Contabilidade incompleta do motor.** Os produtos matriciais dominantes são contados, mas o backward elementar e o Adam não. As ativações recebem custos aproximados; GELU, por exemplo, custa oito operações por elemento no contador. Multiplicação e soma entre ramos têm o mesmo custo registrado, embora seus backwards façam operações diferentes. Igualdade no contador não significa igualdade física exata. FLOPs tampouco medem memória, energia ou latência; os tempos concorrentes não sustentam uma afirmação de aceleração.")
    doc.p("**Três sementes e comparações múltiplas.** O tamanho amostral é mínimo e o IC t tem dois graus de liberdade. Os resultados de cada semente compartilham o mesmo teste oficial; o DP não mede toda a incerteza sobre a população. Avaliar muitas configurações aumenta a chance de uma média extrema. Não usamos o teste para adaptar a grade ou a taxa, mas qualquer escolha posterior da melhor variante é exploratória. Barras sobrepostas não estabelecem equivalência, e separar barras não substitui uma análise estatística completa.")
    doc.p("**Otimização fixa e escala.** A taxa 0,01 e 100 épocas reproduzem a referência, mas não são necessariamente adequadas a cada porta ou largura. Xavier com ramos multiplicativos muda a distribuição das ativações em relação ao denso. Um resultado desfavorável pode indicar dificuldade de treinamento no protocolo, e não incapacidade da classe de funções. O controle soma também muda condicionamento, portanto não separa expressividade de otimização.")
    doc.p("**Pré-processamento e divisão.** O carregador ajusta médias, desvios e vocabulários usando as 32.561 linhas do treino oficial antes da divisão interna. As features da validação influenciam essa transformação; seus rótulos e o teste não. Preservamos esse comportamento para reproduzir a referência e o declaramos como limitação. As sementes mudam simultaneamente inicialização e divisão, impedindo atribuir toda a dispersão a apenas um fator.")
    doc.p("**Métrica e domínio.** Acurácia pode ocultar desempenho desigual na classe minoritária. Não demonstramos melhoria em recall, calibração ou equidade. Adult é uma única base, com 108 features, CPU e float64. Não inferimos desempenho em imagens, grandes modelos de linguagem, GPU, outras precisões ou outras bases. As ablações removem informação e não equivalem a intervenções causais sobre interações específicas.")
    doc.p("**Pareamento de orçamento.** h:2h aproxima o custo apenas no problema completo; a diferença cresce nos contínuos. O denso no teto corrige a restrição de inferência até o arredondamento inteiro, mas não iguala treino, parâmetros ou duração. As fronteiras são calculadas sobre médias; deslocamentos menores que a variabilidade não estabelecem dominância.")
    doc.secao(8,"O que permanece em aberto")
    doc.p("O próximo experimento prioritário é ajustar a taxa de aprendizado por família usando somente validação, sob orçamento declarado para a busca, e então repetir em mais sementes. Isso ajudaria a distinguir vantagem arquitetural de uma taxa inadequada. Um desenho com split fixo e várias inicializações, seguido de vários splits, separaria as duas fontes de dispersão.")
    doc.p("Para investigar o mecanismo, testar interações controladas em dados sintéticos e perturbar grupos categóricos preservando marginais seria mais informativo que apenas remover blocos. Para generalização, repetir em outras bases tabulares e incluir uma referência de árvores com busca comparável. Para eficiência física, medir inferência isolada, memória e latência com aquecimento e controle de threads; para o uso preditivo, acrescentar métricas por classe e calibração. Nenhuma dessas análises é reivindicada como concluída nesta entrega.")

    doc.pagina()
    doc.secao(9,"Reprodução")
    doc.p("Todos os números foram produzidos localmente; não há resultados simulados ou copiados dos artigos. A entrega mantém o instrumento original e depende da cópia completa de BERT-CPU, incluindo `datasets/adult`. Execute os comandos abaixo a partir de `pesquisa/`. O executor pula configurações completas e recusa misturar código/dados com hashes diferentes.")
    doc.codigo("python q05_estudo/test_modelos.py --json -v\npython q05_estudo/executar.py --processos 1\npython q05_estudo/verificar_resultados.py\npython q05_estudo/analisar.py\npython q05_estudo/gerar_relatorio.py")
    doc.p("Para reproduzir apenas a auditoria, tabelas e documentos, omita o comando `executar.py`. Para um novo treino independente, use uma cópia integral do repositório e mova os resultados existentes para backup nessa cópia, conforme o README. O notebook abre como relatório e análise dos arquivos salvos; não inicia automaticamente os 156 treinos.")
    registros=[json.loads(p.read_text()) for p in (PASTA/'resultados').glob('*_s[012].json')]
    inicio=min(datetime.fromisoformat(r['concluido_utc'])-timedelta(seconds=r['duracao_total_s']) for r in registros)
    fim=max(datetime.fromisoformat(r['concluido_utc']) for r in registros)
    janela=(fim-inicio).total_seconds()
    soma=a['soma_tempo_treinos_s']+ref['seconds']
    if not referencia_compartilhada:
        soma+=equivalencia['seconds']
    picos=[r.get('pico_memoria_bytes',0) for r in registros]
    recursos=json.loads((PASTA/'resultados/resumo_recursos.json').read_text())
    historico_memoria=(f"Após {historicos} treinos em três processos, episódios de OOM exigiram retomada sequencial. " if historicos else "A reprodução usa execução sequencial desde o início. ")
    doc.p(f"**Ambiente de origem:** Python {ambiente['python'].split()[0]}, NumPy {ambiente['numpy']}, CachyOS Linux, Ryzen 5 5600, 16 GB de RAM. "+historico_memoria+f"A coleta de ciclos por época evita acumular grafos; cada filho usa um cgroup próprio, com teto de 3 GiB, swap zero e uma thread BLAS. Pico entre os treinos protegidos: {numero(max(picos)/1024**3,2)} GiB; menor MemAvailable monitorado: {numero(recursos['disponivel_minimo_bytes']/1024**3,2)} GiB. O backend permaneceu NumPy/CPU float64.")
    with (PASTA/'resultados/sessoes_seguras.jsonl').open() as f:
        tempo_seguro=sum(json.loads(s)['duracao_parede_s'] for s in f)
    doc.p(f"**Duração:** janela observada da grade: {numero(janela/60,2)} min, incluindo interrupções e períodos sem execução; serviços protegidos concluídos: {numero(tempo_seguro/60,2)} min de parede acumulados; soma dos treinos concluídos com referência e validação de equivalência: {numero(soma/60,2)} min. A referência original levou {numero(ref['seconds'],2)} s e sua validação protegida, {numero(equivalencia['seconds'],2)} s. Tempo de pesquisa/redação não foi cronometrado; execuções interrompidas não integram essas somas e o total anterior aos OOMs não é recuperável. Não apresentamos essas durações como comparação de latência entre arquiteturas.")
    if referencia_compartilhada:
        doc.p("Nesta reprodução, a referência e a equivalência foram medidas no mesmo serviço; seu tempo e seus FLOPs entram apenas uma vez nos totais.")
    arquivo_log=PASTA/'resultados/execucao.log'
    log=arquivo_log.read_text() if arquivo_log.exists() else ''
    retomadas=[156-int(n) for n in re.findall(r'Grade: 156 execuções; pendentes: (\d+)',log) if int(n)<156]
    preservados=', '.join(str(n) for n in retomadas)
    continuidade=(f"As retomadas históricas preservaram, respectivamente, {preservados} execuções; a retomada protegida preservou {historicos}. " if retomadas else "")
    doc.p(f"O custo acumulado registrado dos treinos concluídos é {numero((audit['total_flops_treino_com_referencia']+extra_equivalencia)/1e12,3)} TFLOPs, incluindo referência e validação de equivalência. Trabalho parcial interrompido não foi salvo nem incluído nesse total. "+continuidade+"Horários por execução, logs e `sessoes_seguras.jsonl` documentam a continuidade.")
    linhas=[]
    for nome in ['modelos.py','executar_seguro.py','analisar.py','test_modelos.py',
                 'verificar_resultados.py','gerar_relatorio.py','proveniencia.py',
                 'recursos_documentos.py','verificar_estrutura.py']:
        linhas.append([nome,len((PASTA/nome).read_text().splitlines())])
    doc.tabela("Tabela 8 — Principais arquivos (linhas físicas, incluindo comentários)",["Arquivo","Linhas"],linhas)
    doc.p("`resultados/ambiente.json` registra versões, hashes e atributos; `verificacao.json` registra testes e erros de gradiente; `auditoria_final.json` registra verificação de 156 pesos, custos e três inferências recalculadas. Cada JSON individual contém métricas, configuração e curvas; cada NPZ correspondente contém os pesos finais. As tabelas CSV permitem refazer as médias e diferenças sem executar treino. O PDF usa XeLaTeX; Markdown e notebook permanecem acessíveis sem LaTeX.")
    doc.sub("Referências")
    referencias=[
        "[1] MA, X. et al. Rewrite the Stars. CVPR, 2024. https://arxiv.org/abs/2403.19967",
        "[2] SHAZEER, N. GLU Variants Improve Transformer. 2020. https://arxiv.org/abs/2002.05202",
        "[3] DAUPHIN, Y. N. et al. Language Modeling with Gated Convolutional Networks. ICML, 2017. https://arxiv.org/abs/1612.08083",
        "[4] GRINSZTAJN, L.; OYALLON, E.; VAROQUAUX, G. Why do tree-based models still outperform deep learning on tabular data? NeurIPS, 2022. https://arxiv.org/abs/2207.08815"]
    for r in referencias:doc.p(r)
    return doc


def notebook(markdown):
    celulas=[]
    def md(s):
        celulas.append(dict(cell_type='markdown',id=f'celula{len(celulas):03d}',metadata={},source=s.splitlines(True)))
    ambiente={}
    contador=0
    def codigo(s):
        nonlocal contador
        saida=io.StringIO()
        with contextlib.redirect_stdout(saida):
            exec(compile(s,'<notebook_q05>','exec'),ambiente)
        contador+=1
        celulas.append(dict(cell_type='code',id=f'celula{len(celulas):03d}',metadata={},source=s.splitlines(True),
            execution_count=contador,outputs=[dict(output_type='stream',name='stdout',text=saida.getvalue().splitlines(True))]))
    md("# Estudo Q5 — relatório e análise reproduzível\n\nEste notebook contém o relatório em português e células de leitura dos resultados reais. Executar todas as células **não treina novamente** os modelos. Para treinar ou retomar, siga o README.\n")
    for parte in re.split(r'(?=^## \d+\.)',markdown,flags=re.M):
        if parte.strip():md(parte)
    md("## Conferir os resultados salvos\n\nAs células seguintes foram executadas pelo gerador e mostram os valores publicados.\n")
    codigo('from pathlib import Path\nimport json\n\nPASTA = Path.cwd()\nif (PASTA / "q05_estudo").is_dir():\n    PASTA = PASTA / "q05_estudo"\nelif (PASTA / "pesquisa/q05_estudo").is_dir():\n    PASTA = PASTA / "pesquisa/q05_estudo"\nanalise = json.loads((PASTA / "resultados/analise.json").read_text())\nauditoria = json.loads((PASTA / "resultados/auditoria_final.json").read_text())\nprint("Execuções:", analise["total_execucoes"])\nprint("Auditoria aprovada:", auditoria["aprovado"])\nprint("Hipótese principal atendida:", analise["hipoteses"]["h1_atendida_pela_regra_registrada"])\n')
    codigo('for r in analise["comparacoes"]:\n    if r["bloco"] == "completo" and r["controle"] == "denso":\n        print(f"h={r[\'h\']}: estrela − denso = {r[\'delta_media_pp\']:+.3f} p.p.; "\n              f"IC95=[{r[\'ic95_inferior_pp\']:+.3f}, {r[\'ic95_superior_pp\']:+.3f}]; "\n              f"{r[\'interpretacao\']}")\n')
    codigo('print((PASTA / "resultados/tabela_summarize.txt").read_text())\n')
    return dict(cells=celulas,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version=sys.version.split()[0])),nbformat=4,nbformat_minor=5)


def main():
    doc=construir()
    markdown="\n".join(doc.md)
    (PASTA/'relatorio.md').write_text(markdown)
    preambulo=r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.65cm,headheight=15pt]{geometry}
\usepackage{fontspec,amsmath,amssymb,booktabs,graphicx,fancyhdr,hyperref}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}[Scale=0.85]
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={Q5 — Operação estrela em Adult}}
\pagestyle{fancy}\fancyhf{}\lhead{BERT-CPU • Questão 5}\rhead{Estudo experimental}\cfoot{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{4em}
\begin{document}
{\Large\bfseries A operação estrela compensa seu custo em Adult?}\par
{\small Relatório da Q5 • 13 de outubro de 2026 • Português brasileiro}\par\medskip
"""
    (PASTA/'relatorio.tex').write_text(preambulo+"\n".join(doc.tex)+"\n\\end{document}\n")
    nb=notebook(markdown)
    (PASTA/'estudo_q05.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
    for _ in range(2):
        log=executar_limitado(['xelatex','-interaction=nonstopmode','-halt-on-error','relatorio.tex'],PASTA)
        (PASTA/'resultados/compilacao_pdf.log').write_text(log)
    # O checkpoint muda durante retomadas; não integra o manifesto científico final.
    fontes=sorted(p for p in PASTA.iterdir() if p.suffix in ('.py','.md','.bib','.ipynb','.tex','.pdf')
                  and p.name != 'CHECKPOINT.md')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fontes}
    (PASTA/'resultados/hashes_entrega.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+"\n")
    print('Relatório Markdown/LaTeX/PDF e notebook executado gerados.')


if __name__=='__main__':
    main()
