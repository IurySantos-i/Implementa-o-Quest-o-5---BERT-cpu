"""Entrada padrão: um trabalhador protegido; preserva a interface de retomada."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from executar_seguro import main

if __name__ == "__main__":
    main()
