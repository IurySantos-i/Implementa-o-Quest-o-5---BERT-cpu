"""Verificações numéricas independentes das arquiteturas da questão 5.

Execute na raiz do repositório com:
    python -m unittest pesquisa.q05_estudo.test_modelos -v
Ou na pasta pesquisa, salvando a verificação após os testes aprovados:
    python q05_estudo/test_modelos.py --json -v

Os oráculos de forward e diferenças finitas usam NumPy, sem chamar o forward
da implementação sob teste. Os pontos da ReLU permanecem longe de zero.
"""

from __future__ import annotations

import os
for _nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_nome] = "1"
import argparse
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]
sys.path.insert(0, str(RAIZ))

from bert_cpu import engine as cpu
from bert_cpu.loss import cross_entropy
from pesquisa.benchmark import baseline_mlp, default_loss
from pesquisa.q05_estudo.modelos import criar_modelo, flops_inferencia


FAMILIAS = ("denso", "estrela", "soma", "glu", "geglu", "swiglu")


def _preencher_parametros(modelo):
    """Pesos não nulos e sinais distintos revelam erros de rota e de ativação."""
    gerador = np.random.default_rng(731)
    parametros = modelo.parameters()
    for indice, parametro in enumerate(parametros):
        parametro.data[:] = gerador.uniform(-0.17, 0.17, parametro.shape)
        if indice < len(parametros) - 1:
            sinal = np.where(np.arange(parametro.shape[1]) % 2 == 0, 1.0, -1.0)
            parametro.data[0] = (0.9 if indice == 0 else -0.8) * sinal


def _selecionar(entradas, bloco):
    if bloco == "continuo":
        return entradas[:6]
    if bloco == "categorico":
        return entradas[6:]
    return entradas


def _forward_numpy(parametros, entradas, familia, bloco="completo"):
    """Definição matricial externa ao autograd, incluindo os vieses."""
    selecionadas = _selecionar(entradas, bloco)
    aumentadas = np.vstack((np.ones((1, selecionadas.shape[1])), selecionadas))
    u = parametros[0].data.T @ aumentadas
    if familia == "denso":
        ocultas = np.maximum(u, 0.0)
    else:
        v = parametros[1].data.T @ aumentadas
        if familia in ("estrela", "soma"):
            ativadas = np.maximum(u, 0.0)
        elif familia == "geglu":
            ativadas = 0.5 * u * (
                1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * u**3))
            )
        else:
            ativadas = 1.0 / (1.0 + np.exp(-u))
            if familia == "swiglu":
                ativadas = u * ativadas
        ocultas = ativadas + v if familia == "soma" else ativadas * v
    aumentadas_ocultas = np.vstack((np.ones((1, ocultas.shape[1])), ocultas))
    return parametros[-1].data.T @ aumentadas_ocultas


def _perda_numpy(parametros, entradas, familia, rotulos):
    logits = _forward_numpy(parametros, entradas, familia)
    deslocados = logits - logits.max(axis=0, keepdims=True)
    log_normalizador = np.log(np.exp(deslocados).sum(axis=0))
    return float(np.mean(log_normalizador - deslocados[rotulos, np.arange(len(rotulos))]))


def verificar_gradientes(familia, passo=1e-6):
    """Confere cada entrada, peso e viés; devolve erros para registrar no relatório."""
    cpu.set_seed(23)
    modelo = criar_modelo(4, familia, 3)
    _preencher_parametros(modelo)
    entradas = cpu.Tensor(np.linspace(-0.3, 0.35, 20).reshape(4, 5))
    rotulos = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    parametros = modelo.parameters()
    modelo.zero_grad()
    perda = cross_entropy(modelo(entradas).T, rotulos)
    perda.backward()
    tensores = [entradas, *parametros]
    analiticos = [tensor.grad.copy() for tensor in tensores]
    erros = []
    total = 0
    for tensor, analitico in zip(tensores, analiticos):
        numerico = np.empty_like(tensor.data)
        for indice in np.ndindex(tensor.shape):
            original = float(tensor.data[indice])
            tensor.data[indice] = original + passo
            mais = _perda_numpy(parametros, entradas.data, familia, rotulos)
            tensor.data[indice] = original - passo
            menos = _perda_numpy(parametros, entradas.data, familia, rotulos)
            tensor.data[indice] = original
            numerico[indice] = (mais - menos) / (2.0 * passo)
        np.testing.assert_allclose(analitico, numerico, rtol=2e-5, atol=2e-8)
        erros.append(float(np.max(np.abs(analitico - numerico))))
        total += tensor.data.size
    return {"familia": familia, "coordenadas": total, "erro_absoluto_maximo": max(erros)}


class TestarModelos(unittest.TestCase):
    def test_forward_independente_e_blocos(self):
        entradas = np.linspace(-0.04, 0.04, 108 * 5).reshape(108, 5)
        for familia in FAMILIAS:
            for bloco, dimensao in (("completo", 108), ("continuo", 6), ("categorico", 102)):
                with self.subTest(familia=familia, bloco=bloco):
                    cpu.set_seed(31)
                    modelo = criar_modelo(108, familia, 3, bloco=bloco)
                    _preencher_parametros(modelo)
                    parametros = modelo.parameters()
                    self.assertEqual(len(parametros), 2 if familia == "denso" else 3)
                    self.assertEqual(parametros[0].shape, (dimensao + 1, 3))
                    esperado = _forward_numpy(parametros, entradas, familia, bloco)
                    obtido = modelo(cpu.Tensor(entradas, requires_grad=False)).data
                    self.assertEqual(obtido.shape, (2, 5))
                    np.testing.assert_allclose(obtido, esperado, rtol=1e-13, atol=1e-13)

    def test_gradientes_todas_as_entradas_pesos_e_vieses(self):
        for familia in FAMILIAS:
            with self.subTest(familia=familia):
                resultado = verificar_gradientes(familia)
                self.assertLess(resultado["erro_absoluto_maximo"], 2e-8)

    def test_blocos_nao_recebem_gradientes_das_features_excluidas(self):
        for bloco, excluidas, mantidas in (
            ("continuo", slice(6, None), slice(None, 6)),
            ("categorico", slice(None, 6), slice(6, None)),
        ):
            with self.subTest(bloco=bloco):
                modelo = criar_modelo(108, "estrela", 3, bloco=bloco)
                _preencher_parametros(modelo)
                entradas = cpu.Tensor(np.linspace(-0.04, 0.04, 108 * 5).reshape(108, 5))
                modelo(entradas).sum().backward()
                np.testing.assert_array_equal(entradas.grad[excluidas], 0.0)
                self.assertGreater(np.max(np.abs(entradas.grad[mantidas])), 0.0)

    def test_inicializacao_densa_reproduz_baseline(self):
        cpu.set_seed(0)
        referencia = baseline_mlp(108)
        cpu.set_seed(0)
        implementado = criar_modelo(108, "denso", 64)
        for esperado, obtido in zip(referencia.parameters(), implementado.parameters()):
            np.testing.assert_array_equal(obtido.data, esperado.data)
        entradas = cpu.Tensor(np.linspace(-0.5, 0.5, 108 * 3).reshape(108, 3))
        np.testing.assert_array_equal(implementado(entradas).data, referencia(entradas).data)

    def test_flops_inferencia_pelo_contador_e_formula_independente(self):
        # Negação, exp, soma, recíproca e multiplicação por 1: sigmoide custa cinco.
        custos_elementares = {"estrela": 2, "soma": 2, "glu": 6, "geglu": 9, "swiglu": 7}
        for entradas in (6, 102, 108):
            for largura in (2, 16):
                for familia in FAMILIAS:
                    with self.subTest(entradas=entradas, largura=largura, familia=familia):
                        modelo = criar_modelo(entradas, familia, largura)
                        lote = 7
                        if familia == "denso":
                            por_exemplo = 2 * largura * (entradas + 1) + largura + 4 * (largura + 1)
                            parametros = (entradas + 1) * largura + 2 * (largura + 1)
                        else:
                            por_exemplo = 4 * largura * (entradas + 1) + custos_elementares[familia] * largura + 4 * (largura + 1)
                            parametros = 2 * (entradas + 1) * largura + 2 * (largura + 1)
                        cpu.reset_flops()
                        modelo(cpu.Tensor(np.ones((entradas, lote)), requires_grad=False))
                        self.assertEqual(cpu.flop_count(), lote * por_exemplo)
                        self.assertEqual(flops_inferencia(entradas, familia, largura, lote), lote * por_exemplo)
                        self.assertEqual(sum(p.data.size for p in modelo.parameters()), parametros)

    def test_custo_treino_inclui_backward_e_validacao(self):
        for familia in ("denso", "estrela", "soma"):
            for largura in (16, 32):
                with self.subTest(familia=familia, largura=largura):
                    modelo = criar_modelo(108, familia, largura)
                    treino = cpu.Tensor(np.zeros((108, 7)), requires_grad=False)
                    validacao = cpu.Tensor(np.zeros((108, 3)), requires_grad=False)
                    cpu.reset_flops()
                    default_loss(modelo, treino, np.zeros(7, dtype=np.int64)).backward()
                    default_loss(modelo, validacao, np.zeros(3, dtype=np.int64))
                    custo_treino = (449 if familia == "denso" else 886) * largura + 26
                    custo_validacao = (223 if familia == "denso" else 442) * largura + 18
                    self.assertEqual(cpu.flop_count(), 7 * custo_treino + 3 * custo_validacao + 4)

    def test_interacao_mista_estrela_e_ausencia_local_na_soma(self):
        # No mesmo regime ativo da ReLU, apenas o produto tem termo x1*x2.
        passo = 1e-3
        ponto = np.array([[0.1], [0.2]])
        for familia in ("denso", "estrela", "soma"):
            with self.subTest(familia=familia):
                modelo = criar_modelo(2, familia, 1)
                parametros = modelo.parameters()
                parametros[0].data[:] = [[2.0], [0.4], [0.6]]
                if familia != "denso":
                    parametros[1].data[:] = [[-0.1], [0.3], [0.5]]
                parametros[-1].data[:] = [[0.0, 0.0], [1.0, 0.0]]
                valores = []
                for sinal1, sinal2 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    deslocado = ponto + passo * np.array([[sinal1], [sinal2]])
                    valores.append(float(modelo(cpu.Tensor(deslocado)).data[0, 0]))
                derivada = (valores[0] - valores[1] - valores[2] + valores[3]) / (4 * passo**2)
                self.assertAlmostEqual(derivada, 0.38 if familia == "estrela" else 0.0, places=8)


def main():
    argumentos = argparse.ArgumentParser(add_help=False)
    argumentos.add_argument("--json", action="store_true")
    opcoes, restantes = argumentos.parse_known_args()
    inicio = time.perf_counter()
    programa = unittest.main(argv=[sys.argv[0], *restantes], exit=False)
    if not programa.result.wasSuccessful():
        raise SystemExit(1)
    if opcoes.json:
        gradientes = [verificar_gradientes(familia) for familia in FAMILIAS]
        verificacao = {
            "aprovado": True,
            "testes_executados": programa.result.testsRun,
            "falhas": len(programa.result.failures),
            "erros": len(programa.result.errors),
            "numpy": np.__version__,
            "tipo_numerico": str(np.dtype(cpu.default_dtype)),
            "passo_diferencas_centrais": 1e-6,
            "tolerancia_absoluta": 2e-8,
            "tolerancia_relativa": 2e-5,
            "gradientes": gradientes,
            "coordenadas_verificadas": sum(r["coordenadas"] for r in gradientes),
            "erro_absoluto_maximo": max(r["erro_absoluto_maximo"] for r in gradientes),
            "verificacoes": [
                "Forward NumPy independente: seis famílias e três blocos",
                "Todas as entradas, pesos e vieses por diferenças finitas",
                "Gradientes nulos nas features excluídas",
                "Inicialização e forward densos idênticos à referência",
                "FLOPs de inferência: seis famílias e dimensões 6, 102 e 108",
                "FLOPs de treino com backward e validação: denso, estrela e soma",
                "Derivada mista: 0,38 na estrela e zero no denso e na soma",
            ],
            "duracao_s": time.perf_counter() - inicio,
            "comando": "python q05_estudo/test_modelos.py --json -v",
        }
        destino = PASTA / "resultados" / "verificacao.json"
        destino.parent.mkdir(exist_ok=True)
        destino.write_text(json.dumps(verificacao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Verificações aprovadas e registradas em {destino}")


if __name__ == "__main__":
    main()
