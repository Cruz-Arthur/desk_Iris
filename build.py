"""
build.py — Empacota desk_Iris para Windows EXE
===============================================

Modos:
    python build.py             # PADRÃO: PyInstaller onefile — EXE único, build em minutos
    python build.py --nuitka    # Nuitka onefile sem LTO — EXE único, build lento (compila C)
    python build.py --release   # Nuitka onefile + LTO — EXE único otimizado (~60 min)
    python build.py --folder    # Nuitka standalone — ~5 min, pasta, sem LTO
    python build.py --cxfreeze  # cx_Freeze — ~30s, PASTA (não single file), só dev

PyInstaller é o padrão: gera um único .exe SEM compilar C (empacota Python +
binários), por isso a build é de minutos. Nuitka compila tudo para C — binário
mais rápido em runtime, mas a build leva muito mais tempo.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
ICON_ICO = ROOT / "app" / "src" / "assets" / "img" / "logo.ico"
ICON_PNG = ROOT / "app" / "src" / "assets" / "img" / "logo.png"

_VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _reexec_in_venv_if_needed() -> None:
    """
    As dependências de build (nuitka, cx_Freeze) vivem na .venv do projeto.
    Se este script foi chamado por outro Python (ex: o global), re-executa
    a si mesmo com o interpretador da venv — assim `python build.py` sempre
    funciona, independente de qual Python estiver no PATH.
    """
    current = Path(sys.executable).resolve()
    if not _VENV_PY.exists():
        return                      # sem venv — segue com o interpretador atual
    if current == _VENV_PY.resolve():
        return                      # já estamos na venv — evita loop infinito
    print(f"[build] Re-executando com a venv: {_VENV_PY}")
    result = subprocess.run([str(_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit(result.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# cx_Freeze — rápido (não compila C, só empacota)
# ─────────────────────────────────────────────────────────────────────────────

def _copy_data(src: Path, dst: Path) -> None:
    """Copia árvore de diretório, ignorando __pycache__."""
    import shutil
    if not src.exists():
        print(f"  [AVISO] Fonte não encontrada, pulando: {src}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"  Copiado: {src.name} → {dst}")


def build_fast() -> None:
    """
    cx_Freeze lean: comprime todo .py num único zip, exclui numba/tests/Qt
    plugins desnecessários. Objetivo: < 60s na maioria das máquinas.
    """
    import shutil

    out_dir = DIST_DIR / "Iris_fast"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        if out_dir.exists():
            print("[AVISO] Não foi possível limpar dist anterior (EXE em uso?) — continuando...")
    out_dir.mkdir(parents=True, exist_ok=True)

    icon = str(ICON_ICO) if ICON_ICO.exists() else (str(ICON_PNG) if ICON_PNG.exists() else None)
    icon_line = f"icon={icon!r}," if icon else ""

    # Escrito como script separado para evitar problemas de serialização
    script = f"""
import sys
sys.argv = ['setup.py', 'build_exe']
from cx_Freeze import setup, Executable

build_options = dict(
    packages=[
        "onnxruntime", "cv2", "numpy", "PyQt6",
        "websockets", "pyzbar", "app", "app.src.utils",
    ],
    excludes=[
        "numba", "llvmlite",
        "pytest", "unittest", "setuptools", "pip", "wheel",
        "distutils", "pydoc", "doctest", "pdb", "difflib",
        "tkinter", "turtle", "turtledemo",
        "scipy", "pandas", "matplotlib", "sklearn",
    ],
    zip_include_packages="*",
    zip_exclude_packages=["websockets", "onnxruntime", "cv2", "PyQt6", "pyzbar"],
    build_exe={str(out_dir)!r},
    silent=False,
)

setup(
    name="Iris",
    version="1.0.0",
    options={{"build_exe": build_options}},
    executables=[
        Executable(
            {str(ROOT / 'run.py')!r},
            base="Win32GUI",
            target_name="Iris.exe",
            {icon_line}
        )
    ],
)
"""

    tmp = ROOT / "_cx_build_tmp.py"
    tmp.write_text(script, encoding="utf-8")

    print("=" * 60)
    print("  cx_Freeze build — lean/fast mode")
    print(f"  Saída: {out_dir}")
    print("=" * 60)
    print()

    try:
        result = subprocess.run([sys.executable, str(tmp)], cwd=ROOT)
    finally:
        tmp.unlink(missing_ok=True)

    if result.returncode != 0:
        print("\n[ERRO] Build cx_Freeze falhou.")
        sys.exit(result.returncode)

    # cx_Freeze 8 mudou a API de include_files — copia dados manualmente
    print("\nCopiando dados de runtime...")
    _copy_data(ROOT / "app" / "src" / "models", out_dir / "app" / "src" / "models")
    _copy_data(ROOT / "app" / "src" / "assets", out_dir / "app" / "src" / "assets")
    _copy_data(ROOT / "docs",                   out_dir / "docs")

    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1_048_576
    print(f"\n[OK] {out_dir}  (~{size_mb:.0f} MB total)")
    print(f"     Execute: {out_dir / 'Iris.exe'}")


# ─────────────────────────────────────────────────────────────────────────────
# PyInstaller — single file de verdade, build rápido (sem compilar C)
# ─────────────────────────────────────────────────────────────────────────────

def build_pyinstaller() -> None:
    """
    Gera dist/Iris.exe (single file) sem compilar C — empacota o Python e os
    binários numa SFX. Build de minutos, não de horas.
    """
    import shutil

    work = ROOT / "build" / "_pyi"
    spec = ROOT / "Iris.spec"

    # Separador de --add-data no Windows é ';' (origem;destino_no_bundle)
    sep = ";"
    add_data = [
        f"{ROOT / 'app' / 'src' / 'models'}{sep}app/src/models",
        f"{ROOT / 'app' / 'src' / 'assets'}{sep}app/src/assets",
        f"{ROOT / 'docs'}{sep}docs",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--noconsole",                 # sem janela de console (modo headless)
        "--name", "Iris",
        "--distpath", str(DIST_DIR),
        "--workpath", str(work),
        "--specpath", str(ROOT),
        "--noconfirm",
        "--clean",
        "--paths", str(ROOT),          # acha o pacote `app`
        # Pacotes com DLLs/dados que o modulegraph não resolve sozinho
        "--collect-all", "onnxruntime",
        "--collect-all", "zxingcpp",   # decoder primário — .pyd nativo
        "--collect-all", "pyzbar",     # fallback (libzbar precisa de MSVCR120)
        "--collect-submodules", "app",
        # cv2/PyQt6/websockets têm hooks oficiais — PyInstaller resolve sozinho
    ]

    icon = ICON_ICO if ICON_ICO.exists() else (ICON_PNG if ICON_PNG.exists() else None)
    if icon is not None:
        cmd += ["--icon", str(icon)]

    for d in add_data:
        cmd += ["--add-data", d]

    cmd.append(str(ROOT / "run.py"))

    print("=" * 60)
    print("  PyInstaller build — onefile (single .exe, rápido)")
    print(f"  Saída: {DIST_DIR / 'Iris.exe'}")
    print("=" * 60)
    print()

    result = subprocess.run(cmd, cwd=ROOT)
    # Limpeza dos artefatos intermediários
    shutil.rmtree(work, ignore_errors=True)
    spec.unlink(missing_ok=True)

    if result.returncode != 0:
        print("\n[ERRO] Build PyInstaller falhou.")
        sys.exit(result.returncode)

    exe = DIST_DIR / "Iris.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / 1_048_576
        print(f"\n[OK] {exe}  ({size_mb:.1f} MB)")
        print(f"     Execute: {exe}")
    else:
        print(f"\n[OK] Build concluído em {DIST_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Nuitka — compilação C real (mais lento, binário menor e mais rápido)
# ─────────────────────────────────────────────────────────────────────────────

def build_nuitka(onefile: bool = True, lto: bool = False) -> None:
    cpu_count = os.cpu_count() or 4

    cmd = [
        sys.executable, "-m", "nuitka",

        # ── Modo de saída ────────────────────────────────────────────────────
        "--onefile" if onefile else "--standalone",

        # ── Plugins obrigatórios ─────────────────────────────────────────────
        "--enable-plugin=pyqt6",

        # ── Pacotes que Nuitka não detecta automaticamente ───────────────────
        "--include-package=onnxruntime",
        "--include-package=cv2",
        "--include-package=websockets",
        "--include-package=pyzbar",
        "--include-package=app",

        # ── Dados não-Python ─────────────────────────────────────────────────
        f"--include-data-dir={ROOT / 'app' / 'src' / 'models'}=app/src/models",
        f"--include-data-dir={ROOT / 'app' / 'src' / 'assets'}=app/src/assets",
        f"--include-data-dir={ROOT / 'docs'}=docs",

        # ── Windows ──────────────────────────────────────────────────────────
        "--windows-console-mode=disable",
        "--product-name=Iris",
        "--product-version=1.0.0",
        "--file-description=Iris Live QR Reader",
        "--copyright=Iris",

        # ── Saída ────────────────────────────────────────────────────────────
        f"--output-dir={DIST_DIR}",
        f"--output-filename={'Iris' if onefile else 'Iris_folder'}",

        # ── Velocidade de build ──────────────────────────────────────────────
        f"--jobs={cpu_count}",            # todos os cores na compilação C
        "--lto=" + ("yes" if lto else "no"),   # LTO off = linking 5-10x mais rápido
        "--assume-yes-for-downloads",     # nunca trava esperando input (gcc/deps)

        str(ROOT / "run.py"),
    ]

    # ccache acelera drasticamente a 2ª build em diante (reusa objetos .o)
    os.environ.setdefault("NUITKA_CACHE_DIR", str(ROOT / ".nuitka-cache"))

    if ICON_ICO.exists():
        cmd.insert(-1, f"--windows-icon-from-ico={ICON_ICO}")
    elif ICON_PNG.exists():
        cmd.insert(-1, f"--windows-icon-from-png={ICON_PNG}")

    label = (
        f"onefile {'+LTO (release)' if lto else 'fast (no LTO)'}"
        if onefile else "folder (dev)"
    )
    print("=" * 60)
    print(f"  Nuitka build — {label}  [{cpu_count} jobs]")
    print(f"  Saída: {DIST_DIR}")
    print("  2ª execução usa cache (.nuitka-cache) — muito mais rápida.")
    print("=" * 60)
    print()

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("\n[ERRO] Build Nuitka falhou.")
        sys.exit(result.returncode)

    exe = DIST_DIR / ("Iris.exe" if onefile else "Iris_folder.dist" / "Iris_folder.exe")
    if exe.exists():
        size_mb = exe.stat().st_size / 1_048_576
        print(f"\n[OK] {exe}  ({size_mb:.1f} MB)")
    else:
        print(f"\n[OK] Build concluído em {DIST_DIR}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _reexec_in_venv_if_needed()

    p = argparse.ArgumentParser(description="Iris EXE builder")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--release", action="store_true",
                   help="Nuitka onefile + LTO: ~60 min, EXE único otimizado")
    g.add_argument("--nuitka",  action="store_true",
                   help="Nuitka onefile sem LTO: ~lento, EXE único")
    g.add_argument("--cxfreeze", action="store_true",
                   help="cx_Freeze: ~30s, PASTA (não single file), só dev")
    g.add_argument("--folder",  action="store_true",
                   help="Nuitka standalone: ~5 min, pasta, sem LTO")
    args = p.parse_args()

    if args.release:
        build_nuitka(onefile=True, lto=True)
    elif args.nuitka:
        build_nuitka(onefile=True, lto=False)
    elif args.folder:
        build_nuitka(onefile=False)
    elif args.cxfreeze:
        build_fast()
    else:
        # Padrão: single file de verdade, build rápido (PyInstaller, sem compilar C)
        build_pyinstaller()


if __name__ == "__main__":
    main()
