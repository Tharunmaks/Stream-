import glob, os, sysconfig, torch

def load_corpus(path=None, max_bytes=2_000_000):
    """Byte-level corpus. Default: Python stdlib sources (always present offline)."""
    if path and os.path.exists(path):
        data = open(path, "rb").read()[:max_bytes]
    else:
        data = b""
        for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))):
            data += open(f, "rb").read()
            if len(data) >= max_bytes:
                break
        data = data[:max_bytes]
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long()

def batches(data, batch, seq, gen):
    while True:
        ix = torch.randint(0, data.numel() - seq - 1, (batch,), generator=gen)
        x = torch.stack([data[i:i+seq] for i in ix]); y = torch.stack([data[i+1:i+seq+1] for i in ix])
        yield x, y
