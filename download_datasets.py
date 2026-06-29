"""Materializa todas as bases do projeto em data/raw/.

  - freMTPL2 (v1): via OpenML (sem autenticacao) -> CSV em data/raw/.
  - Portfolio ES (v2) e Mercado BR (v4): via Kaggle (precisa de credencial da API).

Uso:  python download_datasets.py

Os arquivos ficam em data/raw/ (gitignored): copias locais e fixas do dado,
uteis para trabalhar offline e congelar a versao usada nos resultados.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASETS = {
    "es_portfolio_v2": "mexwell/motor-vehicle-insurance-portfolio",
    "br_market_v4": "rodrigodomingos/brazilian-insurance-motor-market",
}


def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(ROOT))


def save_fremtpl2() -> None:
    """freMTPL2 freq+sev (raw, sem limpeza) -> CSV."""
    from sklearn.datasets import fetch_openml

    print("== freMTPL2 (v1) via OpenML ==")
    for name, data_id in [("freMTPL2freq", 41214), ("freMTPL2sev", 41215)]:
        df = fetch_openml(data_id=data_id, as_frame=True).frame
        out = RAW / f"{name}.csv"
        df.to_csv(out, index=False)
        size_mb = out.stat().st_size / 1e6
        print(f"  OK {name}: {df.shape[0]:,} linhas x {df.shape[1]} cols "
              f"-> {rel(out)} ({size_mb:.1f} MB)")


def kaggle_creds_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (pathlib.Path.home() / ".kaggle" / "kaggle.json").exists()


def print_kaggle_setup() -> None:
    token_path = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    print("  ! Credencial do Kaggle nao encontrada. Para habilitar o download:")
    print("    1. kaggle.com -> sua foto -> Settings -> API -> 'Create New API Token'")
    print(f"    2. salve o kaggle.json baixado em: {token_path}")
    print("    3. se faltar a lib:  pip install kaggle")
    print("    4. rode de novo:     python download_datasets.py")


def download_kaggle() -> None:
    print("\n== Kaggle (v2 ES, v4 BR) ==")
    if not kaggle_creds_present():
        print_kaggle_setup()
        return
    if shutil.which("kaggle") is None:
        print("  ! CLI 'kaggle' nao instalada. Rode: pip install kaggle")
        return
    for sub, slug in KAGGLE_DATASETS.items():
        dest = RAW / sub
        dest.mkdir(exist_ok=True)
        print(f"  baixando {slug} -> {rel(dest)} ...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", slug,
                 "-p", str(dest), "--unzip"],
                check=True,
            )
            print(f"  OK {slug}")
        except subprocess.CalledProcessError as exc:
            print(f"  ! falhou ({slug}): {exc}")


def main() -> None:
    save_fremtpl2()
    download_kaggle()
    print("\nConteudo atual de data/raw/:")
    for p in sorted(RAW.rglob("*")):
        if p.is_file():
            print(f"  {rel(p)}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
