"""Execução sequencial com coleta de grafos, cgroup isolado e vigilância de RAM."""
import os
for _nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_nome] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/q05_matplotlib")

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from .proveniencia import caminho_historico, conferir_hash
else:
    from proveniencia import caminho_historico, conferir_hash

PASTA=Path(__file__).resolve().parent
RAIZ=PASTA.parents[1]
RESULTADOS=PASTA/"resultados"
GIB=1024**3
LIMITE_DISPONIVEL_INICIO=7*GIB
LIMITE_DISPONIVEL_EXECUCAO=int(4.5*GIB)
LIMITE_CONJUNTO=9_000_000_000


def gravar(caminho, valor):
    temporario=Path(str(caminho)+".tmp")
    temporario.write_text(json.dumps(valor,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporario.replace(caminho)


def sha(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def memoria():
    info={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if len(s.split())>=2 and s.split()[1].isdigit()}
    vm={s.split()[0]:int(s.split()[1]) for s in Path('/proc/vmstat').read_text().splitlines()}
    psi={}
    for s in Path('/proc/pressure/memory').read_text().splitlines():
        partes=s.split();psi[partes[0]]=float(partes[1].split('=')[1])
    # Inclui todos os escopos ativos de VS Code do usuário, sem somar RSS compartilhado.
    raiz=Path(f'/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice')
    codigo=sum(int((p/'memory.current').read_text()) for p in raiz.glob('*VSCode*.scope') if (p/'memory.current').exists())
    return dict(disponivel=info['MemAvailable'],total=info['MemTotal'],
        swap_usada=info['SwapTotal']-info['SwapFree'],swap_entrada=vm.get('pswpin',0),
        swap_saida=vm.get('pswpout',0),psi_some10=psi.get('some',0),psi_full10=psi.get('full',0),vscode_bytes=codigo)


def identificador(c):
    return f'{c["bloco"]}_{c["familia"]}_h{c["h"]}_s{c["semente"]}'


def grade():
    saida=[]
    for bloco,d in [('completo',108),('continuo',6),('categorico',102)]:
        familias=['denso','estrela','soma','glu','geglu','swiglu'] if bloco=='completo' else ['denso','estrela']
        for h in [16,32,64,128]:
            for f in familias+['denso_teto']:
                largura=2*h if f=='denso' else ((4*d+10)*h//(2*d+7) if f=='denso_teto' else h)
                for s in [0,1,2]:
                    saida.append(dict(bloco=bloco,familia=f,h=h,largura=largura,semente=s,epocas=100,taxa=.01))
    return saida


def validar_cgroup():
    caminho=Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
    if 'bert-q05-calculo-' not in str(caminho):
        raise RuntimeError('Trabalhador precisa estar no serviço isolado bert-q05-calculo-*')
    if int((caminho/'memory.max').read_text()) > 3*GIB or (caminho/'memory.swap.max').read_text().strip()!='0':
        raise RuntimeError('Limites de RAM/swap não foram aplicados ao trabalhador')
    return caminho


def trabalhador(nome):
    grupo=validar_cgroup()
    # Limite de espaço virtual é uma segunda barreira, apenas neste filho.
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(4*GIB,4*GIB))
    import numpy as np
    from dataclasses import asdict
    sys.path.insert(0,str(RAIZ))
    from pesquisa.benchmark import run_adult, baseline_mlp
    from pesquisa.q05_estudo.modelos import criar_modelo, flops_inferencia, contar_parametros
    inicio=time.perf_counter()
    memoria_epocas=[]
    def coletar(epoca,modelo,opt):
        coletados=gc.collect()
        consumo=int((grupo/'memory.current').read_text())
        supervisor=int(os.environ['Q05_SUPERVISOR_PID'])
        try:os.kill(supervisor,0)
        except ProcessLookupError:raise RuntimeError('Supervisor encerrado; evitando trabalhador órfão')
        m=memoria()
        if m['disponivel']<LIMITE_DISPONIVEL_EXECUCAO or m['vscode_bytes']+consumo>LIMITE_CONJUNTO:
            raise RuntimeError('Margem de memória insuficiente no trabalhador')
        if epoca==1 or epoca%10==0:
            memoria_epocas.append(dict(epoca=epoca,bytes=consumo,objetos_coletados=coletados))
    if nome=='equivalencia':
        r=run_adult(baseline_mlp,seed=0,on_epoch=coletar)
        referencia=RESULTADOS/'referencia_original.json'
        antigo=json.loads((referencia if referencia.exists() else PASTA/'historico/referencia_esperada.json').read_text())
        for campo in ['score','train_flops','infer_flops']:
            assert getattr(r,campo)==antigo[campo],campo
        for campo in ['train_hist','val_hist']:
            np.testing.assert_allclose(r.extra[campo],antigo['extra'][campo],rtol=1e-12,atol=1e-12)
        if not referencia.exists():
            r.extra.pop('model')
            gravar(referencia,asdict(r))
        gravar(RESULTADOS/'equivalencia_memoria.json',dict(aprovado=True,score=r.score,
            train_flops=r.train_flops,infer_flops=r.infer_flops,seconds=r.seconds,
            curvas_equivalentes=True,memoria_epocas=memoria_epocas,
            pico_bytes=int((grupo/'memory.peak').read_text()),concluido_utc=datetime.now(timezone.utc).isoformat()))
    else:
        c=next(c for c in grade() if identificador(c)==nome)
        familia='denso' if c['familia']=='denso_teto' else c['familia']
        r=run_adult(lambda n:criar_modelo(n,familia,c['largura'],c['bloco']),seed=c['semente'],
            epochs=c['epocas'],lr=c['taxa'],on_epoch=coletar,name=nome.rsplit('_s',1)[0])
        modelo=r.extra.pop('model'); d=modelo.n_entradas
        assert r.infer_flops==flops_inferencia(d,familia,c['largura'],16281)
        assert r.extra['n_params']==contar_parametros(d,familia,c['largura'])
        assert np.isfinite(r.extra['train_hist']).all() and np.isfinite(r.extra['val_hist']).all()
        np.savez_compressed(RESULTADOS/f'{nome}_pesos.npz',**{n:p.data for n,p in modelo.named_parameters()})
        saida=asdict(r)
        saida.update(configuracao=c,duracao_total_s=time.perf_counter()-inicio,
            concluido_utc=datetime.now(timezone.utc).isoformat(),ambiente_versao='seguro',
            memoria_epocas=memoria_epocas,pico_memoria_bytes=int((grupo/'memory.peak').read_text()))
        gravar(RESULTADOS/f'{nome}.json',saida)
        print(f'{nome}: acurácia={r.score:.6f}; pico={saida["pico_memoria_bytes"]/GIB:.3f} GiB',flush=True)
    gc.collect()


def executar_servico(nome):
    antes=memoria()
    if antes['disponivel']<LIMITE_DISPONIVEL_INICIO or antes['vscode_bytes']+3*GIB>LIMITE_CONJUNTO or antes['psi_full10']>.5:
        raise RuntimeError(f'Margem insuficiente para iniciar com segurança: {antes}')
    unidade=f'bert-q05-calculo-{os.getpid()}-{nome.replace("_","-")}'
    comando=['systemd-run','--user','--quiet','--wait','--pipe','--collect',f'--unit={unidade}',
        '--property=MemoryHigh=2G','--property=MemoryMax=3G','--property=MemorySwapMax=0',
        '--property=TasksMax=16','--property=CPUQuota=100%','--property=OOMPolicy=stop',
        '--property=Nice=10','--property=RuntimeMaxSec=600','--property=KillMode=control-group',
        '--property=TimeoutStopSec=5',f'--working-directory={PASTA}',
        f'--setenv=Q05_SUPERVISOR_PID={os.getpid()}',
        sys.executable,'-u',str(Path(__file__).resolve()),'--worker',nome]
    inicio=time.perf_counter()
    pico=0;minimo=antes['disponivel'];limite_swap=256*1024*1024//os.sysconf('SC_PAGE_SIZE')
    with (RESULTADOS/'execucao_segura.log').open('a') as log:
        p=subprocess.Popen(comando,stdout=log,stderr=subprocess.STDOUT)
        try:
            time.sleep(.4)
            consulta=subprocess.run(['systemctl','--user','show',unidade,'--property=ControlGroup','--value'],capture_output=True,text=True,check=True)
            cg=Path('/sys/fs/cgroup')/consulta.stdout.strip().lstrip('/')
            if not consulta.stdout.strip():
                raise RuntimeError('Não foi possível localizar o cgroup isolado')
            while p.poll() is None:
                m=memoria()
                consumo=int((cg/'memory.current').read_text()) if (cg/'memory.current').exists() else 0
                pico=max(pico,consumo);minimo=min(minimo,m['disponivel'])
                registro=dict(utc=datetime.now(timezone.utc).isoformat(),execucao=nome,worker_bytes=consumo,**m)
                with (RESULTADOS/'monitoramento_memoria.jsonl').open('a') as f:f.write(json.dumps(registro)+'\n')
                motivo=None
                if m['disponivel']<LIMITE_DISPONIVEL_EXECUCAO:motivo='MemAvailable abaixo de 4,5 GiB'
                elif m['vscode_bytes']+consumo>LIMITE_CONJUNTO:motivo='Conjunto editor/cálculo acima de 9 GB'
                elif m['psi_full10']>1 or m['psi_some10']>3:motivo='Pressão de memória sustentada'
                elif m['swap_saida']-antes['swap_saida']>limite_swap:motivo='Mais de 256 MiB enviados a swap no sistema'
                if motivo:raise RuntimeError(motivo)
                time.sleep(2)
            if p.returncode:
                raise RuntimeError(f'Filho terminou com código {p.returncode}; veja execucao_segura.log')
        finally:
            if p.poll() is None:
                subprocess.run(['systemctl','--user','stop',unidade],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                try:p.wait(timeout=8)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=3)
    saida=dict(execucao=nome,duracao_parede_s=time.perf_counter()-inicio,pico_monitorado_bytes=pico,
        disponivel_minimo_bytes=minimo,concluido_utc=datetime.now(timezone.utc).isoformat())
    with (RESULTADOS/'sessoes_seguras.jsonl').open('a') as f:f.write(json.dumps(saida)+'\n')
    print(f'Concluído {nome}: pico {pico/GIB:.2f} GiB; disponível mínimo {minimo/GIB:.2f} GiB',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processos',type=int,choices=[1],default=1)
    p.add_argument('--worker',help=argparse.SUPPRESS)
    p.add_argument('--somente-referencia',action='store_true')
    p.add_argument('--inspecionar',action='store_true')
    args=p.parse_args()
    if args.worker:
        trabalhador(args.worker);return
    if args.inspecionar:
        print(json.dumps(memoria(),indent=2));return
    RESULTADOS.mkdir(exist_ok=True)
    # Evita duas retomadas concorrentes, inclusive em terminais diferentes.
    import fcntl
    with (PASTA/'execucao.lock').open('w') as trava:
        fcntl.flock(trava,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (RESULTADOS/'ambiente.json').exists():
            # Proveniência da implementação original; o ambiente operacional atual
            # é registrado separadamente em ambiente_seguro.json.
            gravar(RESULTADOS/'ambiente.json',json.loads((PASTA/'historico/ambiente_original.json').read_text()))
        if not (RESULTADOS/'grade_planejada.json').exists():gravar(RESULTADOS/'grade_planejada.json',grade())
        manifesto=json.loads((RESULTADOS/'ambiente.json').read_text())
        for relativo,esperado in manifesto['hashes_codigo'].items():
            if not conferir_hash(caminho_historico(relativo),esperado):
                raise RuntimeError(f'Código científico/histórico alterado: {relativo}')
        for nome,esperado in manifesto['hashes_dados'].items():
            if sha(RAIZ/'datasets/adult'/nome)!=esperado:raise RuntimeError(f'Dados alterados: {nome}')
        arquivos=['executar_seguro.py','executar.py','adendo_recursos.md','modelos.py']
        seguro=dict(versao='seguro',processos=1,threads=1,memory_max_bytes=3*GIB,swap_max_bytes=0,
            inicio_utc=datetime.now(timezone.utc).isoformat(),hashes={n:sha(PASTA/n) for n in arquivos})
        arquivo=RESULTADOS/'ambiente_seguro.json'
        if arquivo.exists():
            anteriores=json.loads(arquivo.read_text())['hashes']
            if (set(anteriores)!=set(seguro['hashes'])
                    or not all(conferir_hash(PASTA/n,esperado)
                               for n,esperado in anteriores.items())):
                raise RuntimeError('Executor seguro alterado após início; documente uma nova versão')
        else:gravar(arquivo,seguro)
        if not (RESULTADOS/'equivalencia_memoria.json').exists():executar_servico('equivalencia')
        assert json.loads((RESULTADOS/'equivalencia_memoria.json').read_text())['aprovado']
        if args.somente_referencia:return
        pendentes=[c for c in grade() if not (RESULTADOS/f'{identificador(c)}.json').exists()]
        print(f'Execução sequencial protegida: {156-len(pendentes)} preservados, {len(pendentes)} pendentes.',flush=True)
        for i,c in enumerate(pendentes,1):
            print(f'[{i}/{len(pendentes)}] Iniciando {identificador(c)}',flush=True)
            executar_servico(identificador(c))


if __name__=='__main__':
    try:main()
    except (RuntimeError,BlockingIOError,KeyboardInterrupt) as erro:
        print(f'Execução interrompida com segurança: {erro}',file=sys.stderr)
        raise SystemExit(1)
