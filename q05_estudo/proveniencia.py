"""Confere a proveniência original e a migração editorial dos caminhos da Q5.

Os manifestos históricos permanecem intactos. Uma diferença de hash somente é
aceita quando o registro de migração liga explicitamente aquela fonte original
ao arquivo atual e o checkpoint preservado confirma o conteúdo anterior.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

PASTA = Path(__file__).resolve().parent
PACOTE = PASTA.parent
RAIZ = PACOTE.parent
CHECKPOINT = PASTA / "historico/checkpoint_estrutura_original.json"
MIGRACAO = PASTA / "historico/migracao_estrutura.json"


def _sha256(caminho):
    """Calcula o hash em blocos, sem carregar o arquivo inteiro em memória."""
    resumo = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def caminho_historico(relativo):
    """Localiza um caminho de manifesto sem reescrever sua chave histórica."""
    relativo = PurePosixPath(relativo)
    if relativo.is_absolute() or ".." in relativo.parts or not relativo.parts:
        raise ValueError(f"Caminho de manifesto inválido: {relativo}")
    if relativo.as_posix() == "research/q05_estudo/executar.py":
        return PASTA / "historico/executar_original.py"
    if relativo.parts[0] in ("research", "pesquisa"):
        return PACOTE.joinpath(*relativo.parts[1:])
    return RAIZ.joinpath(*relativo.parts)


def conferir_hash(caminho, esperado):
    """Retorna se o hash é exato ou corresponde à ponte editorial registrada.

    A ponte exige simultaneamente o caminho atual explícito, os hashes anterior
    e atual, o hash do checkpoint e a reprodução do hash anterior pelo texto
    original nele preservado. Arquivos sem essa evidência são rejeitados.
    """
    try:
        caminho = Path(caminho).resolve()
        atual = _sha256(caminho)
        if atual == esperado:
            return True

        # A exceção editorial só alcança fontes deste pacote. Dados e arquivos
        # externos continuam sujeitos exclusivamente ao hash original exato.
        relativo = caminho.relative_to(PACOTE).as_posix()
        original = "research/" + relativo
        migracao = json.loads(MIGRACAO.read_text(encoding="utf-8"))
        entrada = migracao["fontes_alteradas"][original]
        if (entrada["caminho_atual"] != "pesquisa/" + relativo
                or entrada["sha256_anterior"] != esperado
                or entrada["sha256_atual"] != atual
                or _sha256(CHECKPOINT) != migracao["checkpoint_original_sha256"]):
            return False

        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        fonte = checkpoint["fontes"][original]
        return (fonte["sha256"] == esperado
                and hashlib.sha256(fonte["conteudo"].encode("utf-8")).hexdigest() == esperado)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
