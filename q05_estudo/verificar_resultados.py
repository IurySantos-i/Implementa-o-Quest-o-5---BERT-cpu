"""Audita a grade completa e reconstrói inferências usando os pesos publicados.

Na pasta pesquisa: python q05_estudo/verificar_resultados.py
Nenhum treinamento é executado. Uma grade incompleta não gera auditoria final.
"""
from __future__ import annotations

import os
for _nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_nome] = "1"

from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__:
    from .proveniencia import caminho_historico, conferir_hash
else:
    from proveniencia import caminho_historico, conferir_hash

import numpy as np

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]
RESULTADOS = PASTA / "resultados"
sys.path.insert(0, str(RAIZ))

import datasets
from bert_cpu import engine as cpu
from pesquisa.q05_estudo.modelos import criar_modelo


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def ler_json(caminho):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def hash_arquivo(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def identificador(configuracao):
    c = configuracao
    return f'{c["bloco"]}_{c["familia"]}_h{c["h"]}_s{c["semente"]}'


def grade_esperada():
    """Grade declarada no protocolo, construída independentemente do executor."""
    grade = {}
    for bloco, dimensao in (("completo", 108), ("continuo", 6), ("categorico", 102)):
        familias = ("denso", "estrela", "soma", "glu", "geglu", "swiglu", "denso_teto") if bloco == "completo" else ("denso", "estrela", "denso_teto")
        for familia in familias:
            for h in (16, 32, 64, 128):
                largura = 2 * h if familia == "denso" else (
                    (4 * dimensao + 10) * h // (2 * dimensao + 7)
                    if familia == "denso_teto" else h
                )
                for semente in (0, 1, 2):
                    c = dict(bloco=bloco, familia=familia, h=h, largura=largura,
                             semente=semente, epocas=100, taxa=0.01)
                    grade[identificador(c)] = c
    return grade


def verificar_custos(registro):
    c = registro["configuracao"]
    d = {"completo": 108, "continuo": 6, "categorico": 102}[c["bloco"]]
    m = c["largura"]
    if c["familia"] in ("denso", "denso_teto"):
        inferencia = (2 * d + 7) * m + 4
        treino = (4 * d + 17) * m + 26
        parametros = (d + 1) * m + 2 * (m + 1)
    else:
        elementares = {"estrela": 2, "soma": 2, "glu": 6, "geglu": 9, "swiglu": 7}[c["familia"]]
        inferencia = (4 * d + 8 + elementares) * m + 4
        treino = (8 * d + 20 + elementares) * m + 26
        parametros = 2 * (d + 1) * m + 2 * (m + 1)
    treino_total = 100 * (26049 * treino + 6512 * (inferencia + 14) + 4)
    exigir(registro["infer_flops"] == 16281 * inferencia, f"Inferência incompatível: {identificador(c)}")
    exigir(registro["train_flops"] == treino_total, f"Treino incompatível: {identificador(c)}")
    exigir(registro["extra"]["n_params"] == parametros, f"Parâmetros incompatíveis: {identificador(c)}")


def restaurar_pesos(c):
    familia = "denso" if c["familia"] == "denso_teto" else c["familia"]
    modelo = criar_modelo(108, familia, c["largura"], bloco=c["bloco"])
    parametros = dict(modelo.named_parameters())
    with np.load(RESULTADOS / f"{identificador(c)}_pesos.npz", allow_pickle=False) as arquivo:
        exigir(set(arquivo.files) == set(parametros), f"Nomes de parâmetros incorretos: {identificador(c)}")
        for nome, parametro in parametros.items():
            pesos = arquivo[nome]
            exigir(pesos.shape == parametro.shape, f"Forma incorreta: {identificador(c)} / {nome}")
            exigir(pesos.dtype == np.float64 and np.isfinite(pesos).all(), f"Pesos inválidos: {identificador(c)} / {nome}")
            parametro.data[:] = pesos
    return modelo


def conferir_inferencia(registro, modelo, teste):
    """Confere engine, cálculo NumPy independente, acurácia e FLOPs salvos."""
    c = registro["configuracao"]
    exigir(c["bloco"] == "completo" and c["familia"] in ("denso", "estrela"), "Oráculo externo limitado aos dois modelos principais")
    pesos = {nome: parametro.data for nome, parametro in modelo.named_parameters()}
    entradas = np.vstack((np.ones((1, teste.n_samples)), teste.X))
    ocultas = np.maximum(pesos["ramo1.weight"].T @ entradas, 0.0)
    if c["familia"] == "estrela":
        ocultas = ocultas * (pesos["ramo2.weight"].T @ entradas)
    esperado = pesos["saida.weight"].T @ np.vstack((np.ones((1, teste.n_samples)), ocultas))
    cpu.reset_flops()
    obtido = modelo(cpu.Tensor(teste.X, requires_grad=False)).data
    medido = cpu.flop_count()
    np.testing.assert_allclose(obtido, esperado, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(obtido.argmax(axis=0), esperado.argmax(axis=0))
    acuracia = float((esperado.argmax(axis=0) == teste.y).mean())
    exigir(acuracia == registro["score"], f"Acurácia não reproduzida pelos pesos: {identificador(c)}")
    exigir(medido == registro["infer_flops"], f"FLOPs não reproduzidos pelos pesos: {identificador(c)}")
    return dict(execucao=identificador(c), acuracia_recalculada=acuracia,
                acuracia_publicada=registro["score"], flops_recalculados=medido,
                flops_publicados=registro["infer_flops"], exemplos=teste.n_samples,
                erro_maximo_logits=float(np.max(np.abs(obtido - esperado))))


def main():
    inicio = time.perf_counter()
    esperada = grade_esperada()
    grade = ler_json(RESULTADOS / "grade_planejada.json")
    exigir(len(grade) == len(esperada) == 156, "A grade deve conter 156 execuções")
    exigir(Counter(identificador(c) for c in grade) == Counter(esperada.keys()), "Identificadores duplicados ou grade diferente do protocolo")
    for c in grade:
        exigir(c == esperada[identificador(c)], f"Configuração divergente: {identificador(c)}")
    faltantes = [f"{ident}{sufixo}" for ident in esperada for sufixo in (".json", "_pesos.npz")
                if not (RESULTADOS / f"{ident}{sufixo}").is_file()]
    exigir(not faltantes, f"Estudo incompleto: faltam {len(faltantes)} arquivos; primeiros: {', '.join(faltantes[:5])}")

    manifesto = ler_json(RESULTADOS / "ambiente.json")
    hashes = dict(manifesto["hashes_codigo"])
    hashes.update({f"datasets/adult/{nome}": valor for nome, valor in manifesto["hashes_dados"].items()})
    for relativo, esperado in hashes.items():
        exigir(conferir_hash(caminho_historico(relativo), esperado), f"Hash histórico divergente: {relativo}")
    exigir(conferir_hash(PASTA / "protocolo.md", manifesto["protocolo_sha256"]), "Protocolo alterado")
    verificacao = ler_json(RESULTADOS / "verificacao.json")
    exigir(verificacao["aprovado"] and verificacao["falhas"] == verificacao["erros"] == 0, "Verificações numéricas não aprovadas")
    seguro = ler_json(RESULTADOS / "ambiente_seguro.json")
    for relativo, esperado in seguro["hashes"].items():
        exigir(conferir_hash(PASTA / relativo, esperado), f"Hash do executor seguro divergente: {relativo}")
    equivalencia = ler_json(RESULTADOS / "equivalencia_memoria.json")
    exigir(equivalencia["aprovado"] and equivalencia["curvas_equivalentes"], "Coleta de grafos sem equivalência demonstrada")

    teste = datasets.load_adult("test")
    exigir(teste.X.shape == (108, 16281), "Dimensão inesperada do teste")
    escolhidas = {"completo_denso_h32_s0", "completo_estrela_h64_s0", "completo_estrela_h128_s0"}
    inferencias, registros, hashes_artefatos = [], [], {}
    for ident, c in esperada.items():
        arquivo = RESULTADOS / f"{ident}.json"
        r = ler_json(arquivo)
        exigir(r["configuracao"] == c and r["seed"] == c["semente"], f"Configuração/seed inconsistentes: {ident}")
        exigir(r["score"] == r["extra"]["acc_test"], f"Score divergente: {ident}")
        for campo in ("acc_train", "acc_val", "acc_test"):
            exigir(0.0 <= r["extra"][campo] <= 1.0, f"Acurácia inválida: {ident}")
        exigir(r["extra"]["epochs"] == 100 and r["extra"]["lr"] == 0.01, f"Protocolo de otimização incorreto: {ident}")
        for campo in ("train_hist", "val_hist"):
            exigir(len(r["extra"][campo]) == 100 and np.isfinite(r["extra"][campo]).all(), f"Curva incompleta/não finita: {ident}")
        verificar_custos(r)
        modelo = restaurar_pesos(c)
        if ident in escolhidas:
            inferencias.append(conferir_inferencia(r, modelo, teste))
        del modelo
        gc.collect()
        registros.append(r)
        for sufixo in (".json", "_pesos.npz"):
            nome = f"{ident}{sufixo}"
            hashes_artefatos[nome] = hash_arquivo(RESULTADOS / nome)
    exigir(len(inferencias) == 3, "Inferências de referência ausentes")
    referencia = ler_json(RESULTADOS / "referencia_original.json")
    reproduzido = ler_json(RESULTADOS / "completo_denso_h32_s0.json")
    for campo in ("score", "train_flops", "infer_flops"):
        exigir(referencia[campo] == reproduzido[campo], f"Referência original diferente do denso 64: {campo}")
    exigir(referencia["train_flops"] == 84227782200 and round(referencia["score"], 4) == 0.8543, "Baseline histórico não reproduzido")

    auditoria = dict(aprovado=True, execucoes=156, configuracoes=52, sementes=[0, 1, 2],
        arquivos_pesos_verificados=156, hashes_codigo_e_dados_verificados=len(hashes),
        custos_analiticos_conferidos=156, inferencias_recalculadas=inferencias,
        referencia_original_reproduzida=True,
        equivalencia_coleta_grafos=True,
        execucoes_sequenciais=sum(r.get("ambiente_versao") == "seguro" for r in registros),
        execucoes_historicas=sum(r.get("ambiente_versao") != "seguro" for r in registros),
        total_flops_treino_grade=sum(r["train_flops"] for r in registros),
        total_flops_treino_com_referencia=sum(r["train_flops"] for r in registros) + referencia["train_flops"],
        duracao_auditoria_s=time.perf_counter() - inicio,
        hashes_artefatos=hashes_artefatos,
        comando="python q05_estudo/verificar_resultados.py")
    destino = RESULTADOS / "auditoria_final.json"
    temporario = destino.with_suffix(".json.tmp")
    temporario.write_text(json.dumps(auditoria, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporario.replace(destino)
    print(f"Auditoria aprovada: 156 execuções, 156 arquivos de pesos e três inferências independentes. Resultado: {destino}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError) as erro:
        print(f"Auditoria não concluída: {erro}", file=sys.stderr)
        raise SystemExit(1)
