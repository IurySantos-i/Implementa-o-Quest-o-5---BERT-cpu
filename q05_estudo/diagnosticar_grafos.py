"""Diagnóstico pequeno de retenção de ciclos, sem dados ou treino do estudo."""
import os
for nome in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[nome]='1'
import gc
import json
from pathlib import Path
import resource
import sys
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
import numpy as np

PASTA=Path(__file__).resolve().parent
sys.path.insert(0,str(PASTA.parents[1]))
from bert_cpu import engine as cpu, optim
from bert_cpu.loss import cross_entropy
from pesquisa.q05_estudo.modelos import criar_modelo


def numero_tensores():
    return sum(isinstance(o,cpu.Tensor) for o in gc.get_objects())


def medir(coleta_por_epoca):
    gc.collect()
    cpu.set_seed(123)
    modelo=criar_modelo(8,'glu',16)
    x=cpu.Tensor(np.random.normal(size=(8,128)),requires_grad=False)
    y=np.random.randint(0,2,128)
    opt=optim.Adam(modelo.parameters(),lr=.01)
    contagens=[]; perdas=[]
    for epoca in range(60):
        opt.zero_grad()
        perda=cross_entropy(modelo(x).T,y)
        perdas.append(float(perda.data))
        perda.backward();opt.step()
        if coleta_por_epoca:gc.collect()
        contagens.append(numero_tensores())
    pesos=[p.data.copy() for p in modelo.parameters()]
    saida=dict(coleta_por_epoca=coleta_por_epoca,tensores_por_epoca=contagens,
               max_tensores=max(contagens),tensores_finais=contagens[-1])
    del perda,modelo,opt,x
    gc.collect()
    return saida,perdas,pesos


def main():
    sem,perdas0,pesos0=medir(False)
    com,perdas1,pesos1=medir(True)
    np.testing.assert_allclose(perdas0,perdas1,rtol=1e-12,atol=1e-12)
    for a,b in zip(pesos0,pesos1):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    resultado=dict(limiares_gc=gc.get_threshold(),epocas=60,lote=128,dimensao=8,largura=16,
                    coleta_automatica=sem,coleta_explicita=com,perdas_e_pesos_equivalentes=True,
                    pico_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    nota='Diagnóstico pequeno; número de objetos não é estimativa exata de RAM do Adult.')
    (PASTA/'resultados/diagnostico_grafos.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2)+'\n')
    print('Máximo de tensores, coleta automática:',sem['max_tensores'])
    print('Máximo de tensores, coleta por época:',com['max_tensores'])
    print('Perdas/pesos equivalentes; pico RSS MiB:',resultado['pico_rss_kib']/1024)


if __name__=='__main__':main()
