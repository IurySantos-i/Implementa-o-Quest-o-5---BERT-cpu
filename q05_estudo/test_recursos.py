"""Confere que margens inseguras impedem iniciar qualquer processo de cálculo."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import executar_seguro as seguro


class TestarRecursos(unittest.TestCase):
    def test_memoria_baixa_bloqueia_antes_de_criar_filho(self):
        amostra=dict(disponivel=6*seguro.GIB,vscode_bytes=seguro.GIB,psi_full10=0)
        with patch.object(seguro,'memoria',return_value=amostra), patch.object(seguro.subprocess,'Popen') as iniciar:
            with self.assertRaisesRegex(RuntimeError,'Margem insuficiente'):
                seguro.executar_servico('teste')
            iniciar.assert_not_called()

    def test_limite_conjunto_bloqueia_antes_de_criar_filho(self):
        amostra=dict(disponivel=9*seguro.GIB,vscode_bytes=7*seguro.GIB,psi_full10=0)
        with patch.object(seguro,'memoria',return_value=amostra), patch.object(seguro.subprocess,'Popen') as iniciar:
            with self.assertRaises(RuntimeError):seguro.executar_servico('teste')
            iniciar.assert_not_called()

    def test_pressao_de_memoria_bloqueia_antes_de_criar_filho(self):
        amostra=dict(disponivel=9*seguro.GIB,vscode_bytes=seguro.GIB,psi_full10=2)
        with patch.object(seguro,'memoria',return_value=amostra), patch.object(seguro.subprocess,'Popen') as iniciar:
            with self.assertRaises(RuntimeError):seguro.executar_servico('teste')
            iniciar.assert_not_called()

    def test_grade_cientifica_preservada(self):
        import json
        grade=seguro.grade()
        esperada=json.loads((seguro.RESULTADOS/'grade_planejada.json').read_text())
        self.assertEqual(grade,esperada)
        self.assertEqual(len(grade),156)
        self.assertEqual(len({seguro.identificador(c) for c in grade}),156)


if __name__=='__main__':unittest.main(verbosity=2)
