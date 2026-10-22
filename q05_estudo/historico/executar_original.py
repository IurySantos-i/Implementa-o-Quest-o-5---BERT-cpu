"""Execute a referência, o estudo completo ou retome resultados já salvos.

Exemplo: python q05_estudo/executar.py --processos 3
"""
import os
for _variavel in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variavel] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/q05_matplotlib")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

import numpy as np

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]
sys.path.insert(0, str(RAIZ))
from research.benchmark import run_adult, baseline_mlp, load_adult_cached
from research.q05_estudo.modelos import (criar_modelo, flops_inferencia,
    contar_parametros, largura_densa_no_teto)


def gravar_json(caminho, objeto):
    caminho = Path(caminho)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(json.dumps(objeto, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporario.replace(caminho)


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def configuracoes():
    configs = []
    for bloco, d in [("completo", 108), ("continuo", 6), ("categorico", 102)]:
        familias = ["denso", "estrela", "soma", "glu", "geglu", "swiglu"] if bloco == "completo" else ["denso", "estrela"]
        for h in [16, 32, 64, 128]:
            for familia in familias + ["denso_teto"]:
                largura = 2 * h if familia == "denso" else (
                    largura_densa_no_teto(d, h) if familia == "denso_teto" else h)
                for semente in [0, 1, 2]:
                    configs.append(dict(bloco=bloco, familia=familia, h=h,
                                        largura=largura, semente=semente,
                                        epocas=100, taxa=0.01))
    return configs


def identificador(c):
    return f'{c["bloco"]}_{c["familia"]}_h{c["h"]}_s{c["semente"]}'


def executar_uma(c):
    inicio = time.perf_counter()
    pasta = PASTA / "resultados"
    ident = identificador(c)
    familia = "denso" if c["familia"] == "denso_teto" else c["familia"]
    r = run_adult(lambda n: criar_modelo(n, familia, c["largura"], c["bloco"]),
                  seed=c["semente"], epochs=c["epocas"], lr=c["taxa"], name=ident.rsplit("_s", 1)[0])
    modelo = r.extra.pop("model")
    d = modelo.n_entradas
    assert r.infer_flops == flops_inferencia(d, familia, c["largura"], 16281)
    assert r.extra["n_params"] == contar_parametros(d, familia, c["largura"])
    assert np.isfinite(r.score) and np.isfinite(r.extra["train_hist"]).all()
    assert np.isfinite(r.extra["val_hist"]).all()
    pesos = {nome: p.data for nome, p in modelo.named_parameters()}
    np.savez_compressed(pasta / f"{ident}_pesos.npz", **pesos)
    saida = asdict(r)
    saida["configuracao"] = c
    saida["duracao_total_s"] = time.perf_counter() - inicio
    saida["concluido_utc"] = datetime.now(timezone.utc).isoformat()
    gravar_json(pasta / f"{ident}.json", saida)
    del modelo, r
    gc.collect()
    return ident, saida["score"], saida["seconds"]


def verificar_referencia():
    arquivo = PASTA / "resultados" / "referencia_original.json"
    if arquivo.exists():
        r = json.loads(arquivo.read_text())
    else:
        ref = run_adult(baseline_mlp, seed=0)
        ref.extra.pop("model")
        r = asdict(ref)
        gravar_json(arquivo, r)
    if round(r["score"], 4) != 0.8543 or r["train_flops"] != 84227782200:
        raise RuntimeError(f"Referência não reproduzida: {r['score']}, {r['train_flops']}")
    print(f"Referência reproduzida: acurácia={r['score']:.8f}; treino={r['train_flops']:,} FLOPs", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processos", type=int, default=3)
    p.add_argument("--somente-referencia", action="store_true")
    args = p.parse_args()
    if args.processos < 1:
        p.error("Use pelo menos um processo")
    (PASTA / "resultados").mkdir(exist_ok=True)
    inicio = time.perf_counter()
    inicio_utc = datetime.now(timezone.utc).isoformat()
    dados, teste = load_adult_cached()
    assert dados.X.shape == (108, 32561) and teste.X.shape == (108, 16281)
    assert sum("=" not in n for n in dados.feature_names) == 6
    codigo = [RAIZ / f for f in ["research/benchmark.py", "bert_cpu/engine.py", "bert_cpu/nn.py", "bert_cpu/optim.py", "bert_cpu/loss.py", "datasets/loaders.py"]]
    codigo += [PASTA / "modelos.py", PASTA / "executar.py", PASTA / "protocolo.md"]
    ambiente = dict(python=sys.version, numpy=np.__version__, plataforma=platform.platform(),
                    processador=platform.processor(), processos=args.processos, threads_blas=1,
                    treino_original=list(dados.X.shape), teste=list(teste.X.shape),
                    atributos=dados.feature_names, hashes_codigo={str(f.relative_to(RAIZ)):sha256(f) for f in codigo},
                    hashes_dados={f:sha256(RAIZ / "datasets/adult" / f) for f in ["adult.data", "adult.test"]},
                    protocolo_sha256=sha256(PASTA / "protocolo.md"))
    manifesto = PASTA / "resultados" / "ambiente.json"
    if manifesto.exists():
        anterior = json.loads(manifesto.read_text())
        if anterior["hashes_codigo"] != ambiente["hashes_codigo"] or anterior["hashes_dados"] != ambiente["hashes_dados"]:
            raise RuntimeError("Código/dados mudaram: preserve os resultados anteriores e use uma cópia limpa do estudo")
    else:
        gravar_json(manifesto, ambiente)
    verificar_referencia()
    if args.somente_referencia:
        return
    configs = configuracoes()
    # A grade completa é materializada antes de iniciar qualquer variante.
    gravar_json(PASTA / "resultados" / "grade_planejada.json", configs)
    pendentes = [c for c in configs if not (PASTA / "resultados" / f"{identificador(c)}.json").exists()]
    print(f"Grade: {len(configs)} execuções; pendentes: {len(pendentes)}", flush=True)
    # Python 3.14 usa forkserver no Linux; esse modo precisa de um socket local.
    # fork evita esse requisito no ambiente restrito e cada contador continua isolado.
    contexto = multiprocessing.get_context("fork" if sys.platform == "linux" else "spawn")
    with ProcessPoolExecutor(max_workers=args.processos, mp_context=contexto) as executor:
        futuros = [executor.submit(executar_uma, c) for c in pendentes]
        for i, futuro in enumerate(as_completed(futuros), 1):
            ident, acuracia, segundos = futuro.result()
            print(f"[{i}/{len(pendentes)}] {ident}: {acuracia:.6f}; {segundos:.1f} s", flush=True)
    final = dict(inicio_utc=inicio_utc, fim_utc=datetime.now(timezone.utc).isoformat(),
                 duracao_parede_s=time.perf_counter()-inicio, processos=args.processos,
                 execucoes_nesta_sessao=len(pendentes), total_planejado=len(configs))
    gravar_json(PASTA / "resultados" / f"sessao_{int(time.time())}.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
