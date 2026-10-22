"""Arquiteturas e contabilidade analítica da Q5; entradas em colunas."""
from pathlib import Path
import sys

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from bert_cpu import engine as cpu, nn

FAMILIAS = ("denso", "estrela", "soma", "glu", "geglu", "swiglu")
BLOCOS = ("completo", "continuo", "categorico")


def sigmoide(x):
    """Composição diferenciável, sem criar primitiva ou backward próprios."""
    return 1.0 / (1.0 + (-x).exp())


class RedeComparavel(nn.Module):
    """Dois ramos opcionais, uma combinação e uma saída de dois logits."""

    def __init__(self, n_features, familia, largura, bloco="completo"):
        if familia not in FAMILIAS or bloco not in BLOCOS or largura < 1:
            raise ValueError("Família, bloco ou largura inválidos")
        self.familia, self.largura, self.bloco = familia, largura, bloco
        self.indices = slice(None) if bloco == "completo" else (
            slice(0, 6) if bloco == "continuo" else slice(6, None))
        self.n_entradas = len(np.arange(n_features)[self.indices])
        if self.n_entradas == 0:
            raise ValueError("O bloco selecionado não contém atributos")
        self.ramo1 = nn.Linear(self.n_entradas, largura)
        if familia != "denso":
            self.ramo2 = nn.Linear(self.n_entradas, largura)
        self.saida = nn.Linear(largura, 2)

    def forward(self, x):
        if self.bloco != "completo":
            x = x[self.indices, :]
        u = self.ramo1(x)
        if self.familia == "denso":
            a = u.relu()
        else:
            v = self.ramo2(x)
            if self.familia in ("estrela", "soma"):
                g = u.relu()
            elif self.familia == "glu":
                g = sigmoide(u)
            elif self.familia == "geglu":
                g = u.gelu()
            else:
                g = u * sigmoide(u)
            a = g + v if self.familia == "soma" else g * v
        return self.saida(a)


def criar_modelo(n_features, familia, largura, bloco="completo"):
    return RedeComparavel(n_features, familia, largura, bloco)


def flops_inferencia(n_entradas, familia, largura, n_exemplos=1):
    """FLOPs do contador: matmuls exatos; ativações seguem a convenção do motor."""
    if familia == "denso":
        por_exemplo = (2 * n_entradas + 7) * largura + 4
    else:
        # Sigmoide composto: negação, exp, soma, potência recíproca, multiplicação.
        custo_ativacao = {"estrela": 1, "soma": 1, "glu": 5,
                          "geglu": 8, "swiglu": 6}[familia]
        por_exemplo = (4 * n_entradas + 9 + custo_ativacao) * largura + 4
    return por_exemplo * n_exemplos


def contar_parametros(n_entradas, familia, largura):
    ramos = 1 if familia == "denso" else 2
    return ramos * (n_entradas + 1) * largura + (largura + 1) * 2


def largura_densa_no_teto(n_entradas, h):
    """Maior largura inteira densa com inferência <= estrela(h)."""
    return ((4 * n_entradas + 10) * h) // (2 * n_entradas + 7)
