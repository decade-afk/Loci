import ctypes
import os
import platform
from pathlib import Path


def locate_library(repo_root: Path) -> Path:
    system = platform.system().lower()
    if "windows" in system:
        return repo_root / "target" / "release" / "loci.dll"
    if "darwin" in system:
        return repo_root / "target" / "release" / "libloci.dylib"
    return repo_root / "target" / "release" / "libloci.so"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    lib_path = locate_library(repo_root)
    if not lib_path.exists():
        raise FileNotFoundError(f"library not found: {lib_path}")

    # On Windows, ensure dependent dll lookup works from release dir.
    if os.name == "nt":
        os.add_dll_directory(str(lib_path.parent))

    lib = ctypes.CDLL(str(lib_path))

    lib.loci_engine_new.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int32]
    lib.loci_engine_new.restype = ctypes.c_void_p

    lib.loci_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_float,
    ]
    lib.loci_generate.restype = ctypes.c_void_p

    lib.loci_free_string.argtypes = [ctypes.c_void_p]
    lib.loci_free_string.restype = None

    lib.loci_engine_free.argtypes = [ctypes.c_void_p]
    lib.loci_engine_free.restype = None

    lib.loci_get_last_error.argtypes = []
    lib.loci_get_last_error.restype = ctypes.c_char_p

    model = b"D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf"
    engine = lib.loci_engine_new(model, 512, 0)
    if not engine:
        err = lib.loci_get_last_error()
        raise RuntimeError(f"loci_engine_new failed: {err.decode() if err else '(no error)'}")

    out_ptr = lib.loci_generate(engine, b"Hello from Python", 32, ctypes.c_float(0.7))
    if not out_ptr:
        err = lib.loci_get_last_error()
        lib.loci_engine_free(engine)
        raise RuntimeError(f"loci_generate failed: {err.decode() if err else '(no error)'}")

    out = ctypes.cast(out_ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")
    print(out)

    lib.loci_free_string(out_ptr)
    lib.loci_engine_free(engine)


if __name__ == "__main__":
    main()
