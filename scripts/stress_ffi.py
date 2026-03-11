#!/usr/bin/env python3
"""
Loci C-ABI stress runner for integration stability checks.

Focus:
- repeated generate/free_string lifecycle
- concurrent calls on the same engine handle
- verifies busy-lock behavior instead of process crashes
"""

import argparse
import ctypes
import os
import sys
import threading
from ctypes import c_char_p, c_float, c_int32, c_uint32, c_void_p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress Loci C ABI integration")
    parser.add_argument("--dll", required=True, help="Path to loci.dll / libloci.so / libloci.dylib")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--threads", type=int, default=4, help="Concurrent worker threads")
    parser.add_argument("--iters", type=int, default=20, help="Iterations per thread")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU layer count (0=CPU)")
    parser.add_argument("--max-tokens", type=int, default=8, help="Tokens per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--prompt-prefix", default="stress", help="Prompt prefix")
    parser.add_argument(
        "--use-wait",
        action="store_true",
        help="Use loci_generate_wait instead of loci_generate",
    )
    parser.add_argument(
        "--wait-timeout-ms",
        type=int,
        default=1000,
        help="Timeout for *_wait APIs (milliseconds)",
    )
    parser.add_argument(
        "--strict-timeout",
        action="store_true",
        help="Treat lock timeout in --use-wait mode as failure",
    )
    parser.add_argument(
        "--use-len-api",
        action="store_true",
        help="Use *_with_len APIs (explicit prompt length) instead of C-string APIs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dll_path = os.path.abspath(args.dll)
    model_path = os.path.abspath(args.model)

    if not os.path.isfile(dll_path):
        print(f"ERROR: DLL not found: {dll_path}")
        return 2
    if not os.path.isfile(model_path):
        print(f"ERROR: model not found: {model_path}")
        return 2

    dll_dir = os.path.dirname(dll_path)
    if os.name == "nt":
        dll_dirs = []
        if dll_dir:
            dll_dirs.append(dll_dir)
        # Common MinGW runtime path used by local builds.
        if os.path.isdir(r"D:\mingw64\bin"):
            dll_dirs.append(r"D:\mingw64\bin")
        # Optional extra dirs: semicolon-separated.
        extra_dirs = os.environ.get("LOCI_EXTRA_DLL_DIRS", "")
        for item in extra_dirs.split(";"):
            item = item.strip()
            if item and os.path.isdir(item):
                dll_dirs.append(item)
        for d in dll_dirs:
            os.add_dll_directory(d)

    lib = ctypes.CDLL(dll_path)
    lib.loci_engine_new.argtypes = [c_char_p, c_uint32, c_int32]
    lib.loci_engine_new.restype = c_void_p
    lib.loci_generate.argtypes = [c_void_p, c_char_p, c_uint32, c_float]
    lib.loci_generate.restype = c_void_p
    lib.loci_generate_wait.argtypes = [c_void_p, c_char_p, c_uint32, c_float, c_uint32]
    lib.loci_generate_wait.restype = c_void_p
    if hasattr(lib, "loci_generate_with_len"):
        lib.loci_generate_with_len.argtypes = [c_void_p, c_char_p, c_uint32, c_uint32, c_float]
        lib.loci_generate_with_len.restype = c_void_p
    if hasattr(lib, "loci_generate_wait_with_len"):
        lib.loci_generate_wait_with_len.argtypes = [c_void_p, c_char_p, c_uint32, c_uint32, c_float, c_uint32]
        lib.loci_generate_wait_with_len.restype = c_void_p
    lib.loci_free_string.argtypes = [c_void_p]
    lib.loci_free_string.restype = None
    lib.loci_get_last_error.argtypes = []
    lib.loci_get_last_error.restype = c_char_p
    lib.loci_engine_free_safe.argtypes = [ctypes.POINTER(c_void_p)]
    lib.loci_engine_free_safe.restype = None

    engine = lib.loci_engine_new(model_path.encode("utf-8"), args.ctx, args.gpu_layers)
    if not engine:
        err = lib.loci_get_last_error()
        print(f"ERROR: loci_engine_new failed: {(err or b'').decode('utf-8', errors='replace')}")
        return 3

    lock = threading.Lock()
    stats = {
        "ok": 0,
        "busy": 0,
        "timeout": 0,
        "other_err": 0,
    }

    def worker(tid: int) -> None:
        for i in range(args.iters):
            prompt = f"{args.prompt_prefix}-t{tid}-i{i}".encode("utf-8")
            prompt_len = c_uint32(len(prompt))
            if args.use_len_api and not hasattr(lib, "loci_generate_with_len"):
                with lock:
                    stats["other_err"] += 1
                    print("[worker] loci_generate_with_len not exported by current DLL")
                return
            if args.use_len_api and args.use_wait and not hasattr(lib, "loci_generate_wait_with_len"):
                with lock:
                    stats["other_err"] += 1
                    print("[worker] loci_generate_wait_with_len not exported by current DLL")
                return

            if args.use_len_api:
                if args.use_wait:
                    out_ptr = lib.loci_generate_wait_with_len(
                        engine,
                        prompt,
                        prompt_len,
                        c_uint32(args.max_tokens),
                        c_float(args.temperature),
                        c_uint32(args.wait_timeout_ms),
                    )
                else:
                    out_ptr = lib.loci_generate_with_len(
                        engine,
                        prompt,
                        prompt_len,
                        c_uint32(args.max_tokens),
                        c_float(args.temperature),
                    )
            else:
                if args.use_wait:
                    out_ptr = lib.loci_generate_wait(
                        engine,
                        prompt,
                        c_uint32(args.max_tokens),
                        c_float(args.temperature),
                        c_uint32(args.wait_timeout_ms),
                    )
                else:
                    out_ptr = lib.loci_generate(
                        engine,
                        prompt,
                        c_uint32(args.max_tokens),
                        c_float(args.temperature),
                    )
            if out_ptr:
                lib.loci_free_string(out_ptr)
                with lock:
                    stats["ok"] += 1
                continue

            err = lib.loci_get_last_error()
            msg = (err or b"").decode("utf-8", errors="replace")
            with lock:
                if "engine is busy" in msg:
                    stats["busy"] += 1
                elif "engine lock timeout" in msg or "lock timeout" in msg:
                    stats["timeout"] += 1
                else:
                    stats["other_err"] += 1
                    print(f"[T{tid}] unexpected error at iter {i}: {msg}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(args.threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    engine_ref = c_void_p(engine)
    lib.loci_engine_free_safe(ctypes.byref(engine_ref))

    total = args.threads * args.iters
    print("=== loci ffi stress summary ===")
    print(
        f"total={total} ok={stats['ok']} busy={stats['busy']} "
        f"timeout={stats['timeout']} other_err={stats['other_err']}"
    )
    print(f"engine_ptr_after_free={engine_ref.value}")

    # Busy errors are expected under lock contention.
    # Timeout errors are also expected in wait mode with short timeout windows.
    if stats["other_err"] > 0:
        return 1
    if args.strict_timeout and stats["timeout"] > 0:
        return 1
    if engine_ref.value is not None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
