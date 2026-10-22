"""Execução limitada de verificações e compilação de documentos, sem treinamento."""
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile
import time

for nome in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[nome] = "1"


def memoria():
    campos = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
    vm = dict(l.split() for l in Path("/proc/vmstat").read_text().splitlines())
    psi = Path("/proc/pressure/memory").read_text()
    return dict(disponivel=int(campos["MemAvailable"].split()[0]) * 1024,
                psi_full10=float(re.search(r"full avg10=([\d.]+)", psi)[1]),
                swap_saida=int(vm["pswpout"]))


def exigir_margem():
    atual = memoria()
    if atual["disponivel"] < 4.5 * 1024**3 or atual["psi_full10"] > 1:
        raise RuntimeError(f"Margem de memória insuficiente: {atual}")
    return atual


def executar_limitado(comando, pasta, limite_s=120):
    """Limita somente o filho e interrompe-o antes de pressão crítica de memória."""
    antes = exigir_margem()

    def limites():
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (limite_s, limite_s))

    with tempfile.TemporaryFile(mode="w+") as log:
        filho = subprocess.Popen(comando, cwd=pasta, stdout=log, stderr=subprocess.STDOUT,
                                 preexec_fn=limites, start_new_session=True)
        inicio = time.monotonic()
        try:
            while filho.poll() is None:
                atual = exigir_margem()
                if atual["swap_saida"] - antes["swap_saida"] > 4096:
                    raise RuntimeError("Crescimento de swap; operação interrompida.")
                if time.monotonic() - inicio > limite_s:
                    raise RuntimeError("Operação excedeu o limite de tempo.")
                time.sleep(.5)
        except BaseException:
            try:
                os.killpg(filho.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                filho.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(filho.pid, signal.SIGKILL)
                filho.wait()
            raise
        log.seek(0)
        texto = log.read()
    if filho.returncode:
        raise RuntimeError(f"Operação falhou ({filho.returncode}):\n{texto[-6000:]}")
    return texto
