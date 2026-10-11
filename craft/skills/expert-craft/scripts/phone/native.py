"""Builds and loads the Flash Experts native kernels (fx_kernels.c) on this device.

The C source ships with Expert Craft and is compiled the first time it is needed with the device's own compiler (Termux: pkg install clang;
Linux: gcc or clang; macOS: Xcode command line tools). The library is cached in ~/.cache/expertcraft (EXPERTCRAFT_CACHE to change).
If there is no compiler, or EXPERTCRAFT_NATIVE=0, everything still works with the NumPy code, only slower.
"""
import ctypes, hashlib, os, platform, shutil, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'fx_kernels.c')
_lib = None
_why = None


def _compile(out):
    compilers = [c for c in (os.environ.get('CC'), 'cc', 'clang', 'gcc') if c and shutil.which(c)]
    if not compilers: return 'no C compiler found (Termux: pkg install clang)'
    errs = []
    for cc in compilers:
        for arch in (['-march=native'], ['-mcpu=native'], []):
            tmp = out + '.%d.tmp' % os.getpid()
            cmd = [cc, '-O3', '-fPIC', '-shared', '-fno-math-errno', '-ffp-contract=fast'] + arch + ['-o', tmp, SRC, '-lm']
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            except Exception as e:
                errs.append('%s: %s' % (cc, e)); continue
            if r.returncode == 0:
                os.replace(tmp, out); return None
            errs.append('%s %s: %s' % (cc, ' '.join(arch), r.stderr.strip()[:200]))
    return 'compile failed: ' + ' | '.join(errs[-2:])


def lib():
    """the loaded kernel library, or None (see why())"""
    global _lib, _why
    if _lib is not None or _why is not None: return _lib
    if os.environ.get('EXPERTCRAFT_NATIVE', '1') == '0':
        _why = 'turned off with EXPERTCRAFT_NATIVE=0'; return None
    try:
        src = open(SRC, 'rb').read()
        tag = hashlib.sha256(src + platform.machine().encode() + platform.system().encode()).hexdigest()[:16]
        cache = os.path.expanduser(os.environ.get('EXPERTCRAFT_CACHE', '~/.cache/expertcraft'))
        try: os.makedirs(cache, exist_ok=True)
        except OSError: cache = tempfile.gettempdir()
        so = os.path.join(cache, 'fx_%s.so' % tag)
        if not os.path.exists(so):
            err = _compile(so)
            if err: _why = err; return None
        L = ctypes.CDLL(so)
    except Exception as e:
        _why = 'could not load the native kernels: %s' % e; return None
    P, I64, I, F, U64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_uint64
    L.fx_forward.restype = I64
    L.fx_forward.argtypes = [P, I64, I, I, P, I64, U64, I64, I64, P, P, P, P, P]
    L.fx_backward.restype = I64
    L.fx_backward.argtypes = [P, I64, I, I, P, I64, U64, I64, I64, P, P, P, P, P, P, P, F, F, F, U64, I, P, P]
    L.fx_adam.restype = None
    L.fx_adam.argtypes = [P, P, P, P, I64, F, F, F, F, F, F, F, F]
    _lib = L
    return _lib


def why():
    lib(); return _why or 'native kernels in use'
