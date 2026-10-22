"""Confere a migração de caminhos e a preservação dos artefatos científicos."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__:
    from .proveniencia import caminho_historico, conferir_hash
else:
    from proveniencia import caminho_historico, conferir_hash

PASTA = Path(__file__).resolve().parent
PACOTE = PASTA.parent


def sha(caminho):
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def sem_documentacao(texto):
    arvore = ast.parse(texto)
    for no in ast.walk(arvore):
        if isinstance(no, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if no.body and isinstance(no.body[0], ast.Expr):
                valor = no.body[0].value
                if isinstance(valor, ast.Constant) and isinstance(valor.value, str):
                    no.body = no.body[1:]
    return ast.dump(arvore, include_attributes=False)


def main():
    original = json.loads((PASTA / "historico/checkpoint_estrutura_original.json").read_text())
    migracao = json.loads((PASTA / "historico/migracao_estrutura.json").read_text())
    assert sha(PASTA / "historico/checkpoint_estrutura_original.json") == migracao["checkpoint_original_sha256"]
    # Estes três arquivos são derivados da revisão documental; os dados e a
    # auditoria científica original continuam íntegros, inclusive seus horários.
    derivados = {"resultados/hashes_entrega.json", "resultados/verificacao_entrega.json",
                 "resultados/compilacao_pdf.log"}
    preservados = 0
    for nome, esperado in original["resultados_sha256"].items():
        if nome in derivados:
            continue
        assert sha(PASTA / nome) == esperado, f"Resultado alterado: {nome}"
        preservados += 1
    for nome, esperado in original["figuras_sha256"].items():
        assert sha(PASTA / nome) == esperado, f"Figura alterada: {nome}"
    antes = original["fontes"]["research/q05_estudo/modelos.py"]["sha256"]
    assert sha(PASTA / "modelos.py") == antes, "As fórmulas dos modelos foram alteradas."
    instrumento_original = original["fontes"]["research/benchmark.py"]["conteudo"]
    assert sem_documentacao(instrumento_original) == sem_documentacao((PACOTE / "benchmark.py").read_text()), (
        "A lógica do instrumento de medição foi alterada."
    )
    historico = json.loads((PASTA / "resultados/ambiente.json").read_text())
    for nome, esperado in historico["hashes_codigo"].items():
        assert conferir_hash(caminho_historico(nome), esperado), f"Fonte sem proveniência: {nome}"
    seguro = json.loads((PASTA / "resultados/ambiente_seguro.json").read_text())
    for nome, esperado in seguro["hashes"].items():
        assert conferir_hash(PASTA / nome, esperado), f"Arquivo operacional sem proveniência: {nome}"
    for nome, entrada in migracao["fontes_alteradas"].items():
        fonte = original["fontes"][nome]
        assert fonte["sha256"] == entrada["sha256_anterior"]
        assert hashlib.sha256(fonte["conteudo"].encode()).hexdigest() == fonte["sha256"]
        assert sha(caminho_historico(entrada["caminho_atual"])) == entrada["sha256_atual"]
    assert (PACOTE / "RECURSOS.md").is_file()
    analise = json.loads((PASTA / "resultados/analise.json").read_text())
    assert analise["total_execucoes"] == 156 and len(analise["configuracoes"]) == 52
    resultado = dict(aprovado=True, verificado_em=datetime.now(ZoneInfo("America/Recife")).isoformat(),
                     pacote_atual=PACOTE.name, resultados_preservados=preservados,
                     figuras_preservadas=len(original["figuras_sha256"]),
                     execucoes=156, configuracoes=52, modelos_identicos=True,
                     logica_instrumento_identica=True,
                     manifesto_migracao_sha256=sha(PASTA / "historico/migracao_estrutura.json"),
                     alteracoes_documentadas=len(migracao["fontes_alteradas"]), novos_treinamentos=0)
    (PASTA / "resultados/verificacao_estrutura.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Estrutura aprovada: {preservados} arquivos de resultados e {len(original['figuras_sha256'])} figuras preservados; 156 treinamentos intactos.")


if __name__ == "__main__":
    main()
