"""Confere a entrega final sem treinar: documentos, páginas, saídas e hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

PASTA = Path(__file__).resolve().parent


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def ler(nome):
    return json.loads((PASTA / nome).read_text())


def main():
    analise = ler("resultados/analise.json")
    auditoria = ler("resultados/auditoria_final.json")
    exigir(auditoria["aprovado"], "A auditoria científica não foi aprovada")
    exigir(analise["total_execucoes"] == 156 and len(analise["configuracoes"]) == 52,
           "A análise não cobre toda a grade")

    markdown = (PASTA / "relatorio.md").read_text()
    secoes = [int(n) for n in re.findall(r"^## (\d+)\.", markdown, re.M)]
    exigir(secoes == list(range(1, 10)), "As nove seções devem aparecer na ordem exigida")
    tabela = (PASTA / "resultados/tabela_summarize.txt").read_text().strip()
    exigir(tabela in markdown, "A tabela completa de summarize() não está no relatório")
    exigir("completo_denso_h32" in tabela, "Referência ausente da tabela")
    exigir(len(tabela.splitlines()) == 54, "A tabela deve ter 52 linhas de resultados")

    info = subprocess.check_output(["pdfinfo", str(PASTA / "relatorio.pdf")], text=True)
    paginas = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    exigir(6 <= paginas <= 10, f"Relatório tem {paginas} páginas; o enunciado exige 6–10")
    texto_pdf = subprocess.check_output(["pdftotext", str(PASTA / "relatorio.pdf"), "-"], text=True)
    exigir("completo_denso_h32" in texto_pdf, "Tabela não encontrada no PDF")
    log = (PASTA / "resultados/compilacao_pdf.log").read_text()
    exigir("Missing character:" not in log, "Há caracteres ausentes na fonte do PDF")
    excessos = [float(x) for x in re.findall(r"Overfull \\[hv]box \((\d+(?:\.\d+)?)pt too", log)]
    exigir(not excessos or max(excessos) <= 2, f"Conteúdo ultrapassa a área do PDF: {excessos}")

    nb = ler("estudo_q05.ipynb")
    exigir(nb["nbformat"] == 4, "Formato de notebook inesperado")
    codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
    exigir([c["execution_count"] for c in codigo] == list(range(1, len(codigo) + 1)),
           "Contagem de execução do notebook inconsistente")
    exigir(len(codigo) >= 3, "Faltam células de análise executadas")
    for c in codigo:
        exigir(c["outputs"] and all(o["output_type"] != "error" for o in c["outputs"]),
               "Célula sem saída ou com erro")
    exigir("156" in "".join(codigo[0]["outputs"][0]["text"]), "Saída do notebook desatualizada")
    figuras = re.findall(r"!\[[^\]]*\]\((figuras/[^)]+)\)", markdown)
    exigir(figuras and all((PASTA / f).is_file() for f in figuras), "Figura ausente")
    hashes = ler("resultados/hashes_entrega.json")
    for nome, esperado in hashes.items():
        exigir(hashlib.sha256((PASTA / nome).read_bytes()).hexdigest() == esperado,
               f"Artefato alterado após gerar a entrega: {nome}")
    resultado = dict(aprovado=True, verificado_utc=datetime.now(timezone.utc).isoformat(),
                     paginas_pdf=paginas, secoes=secoes, configuracoes=52, execucoes=156,
                     celulas_executadas=len(codigo), figuras_no_relatorio=len(figuras),
                     hashes_conferidos=len(hashes),
                     sha256_manifesto_entrega=hashlib.sha256(
                         (PASTA / 'resultados/hashes_entrega.json').read_bytes()).hexdigest(),
                     nota="A conferência visual das páginas e a revisão da interpretação complementam estas verificações.")
    (PASTA / "resultados/verificacao_entrega.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    print(f"Entrega aprovada: {paginas} páginas, nove seções, 52 configurações e {len(codigo)} células executadas.")


if __name__ == "__main__":
    main()
