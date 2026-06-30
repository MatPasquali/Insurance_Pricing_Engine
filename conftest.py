import pathlib
import sys

# Garante que `import src.pricing...` funcione ao rodar o pytest da raiz.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
