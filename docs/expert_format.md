# Expert file format (v1)

One file per (layer, expert): `<root>/L<layer:03d>/E<expert:04d>.exp`, plus
`<root>/manifest.txt`. The full definition is in `src/es_format.h`.

```
offset 0       es_header, exactly 4096 bytes
offset 4096    tensor 0 (gate)   padded to a 4096 multiple
               tensor 1 (up)     padded to a 4096 multiple
               tensor 2 (down)   padded to a 4096 multiple
file size      multiple of 4096
```

## Why it looks like this

- **One read per expert.** The header is part of the file, so the loader reads
  the whole file in one pass into a 4096-aligned buffer and validates it in
  place. A separate header read would add a full flash round trip per expert.
- **4096 alignment everywhere.** O_DIRECT requires the file offset, the length
  and the buffer address to be multiples of the logical block size (4096 on
  UFS). Padding costs at most 3 × 4 KiB per expert (under 0.2% at ~6 MiB).
- **Tensors start on a 4096 boundary**, which also gives NEON kernels
  cache-line-aligned data.
- **Self-describing tensors** (role, qtype, rows, cols, offset, nbytes) so the
  same loader works for different model shapes and quant types.
- **Per-tensor hash + header hash.** `es_validate(buf, len, 0)` checks only
  the header (cheap, done on every load). `es_validate(buf, len, 1)` also
  hashes the data. Use that when pulling files from the cloud tier or after
  converting.
- **`model_id`** (a hash of the model name) stops files from two different
  models being mixed in one cache directory.

## Header fields

| field         | meaning                                         |
|---------------|-------------------------------------------------|
| magic         | `EXP1`                                          |
| version       | 1                                               |
| header_bytes  | 4096                                            |
| n_tensors     | usually 3 (gate, up, down)                      |
| layer, expert | must match the file name                        |
| flags         | bit 0: synthetic (benchmark data)               |
| file_bytes    | total file size                                 |
| model_id      | hash of the model name                          |
| t[8]          | tensor descriptors, 64 bytes each               |
| header_hash   | hash of the first 4088 bytes                    |

## Quant types

`ES_QT_Q2_K` is a placeholder that reserves the size: 256-weight blocks of
84 bytes (2.625 bits per weight). The exact layout inside a block gets
fixed in the kernel step, after we check which dot-product instructions
your CPU has.

## Sizes this implies (Q2_K, 3 matrices)

| model shape                         | hidden × inter | per expert |
|-------------------------------------|----------------|------------|
| Qwen3-235B-A22B (94 layers, top-8)  | 4096 × 1536    | 5.9 MiB    |
| DeepSeek-V3 / Kimi K2 (top-8)       | 7168 × 2048    | 13.8 MiB   |

## Known limitation

Some newer Android devices use 16 KiB memory pages. The format only promises
4096 alignment, which is fine for `pread`/O_DIRECT (block-size bound) but
means individual tensors can't be `mmap`ed separately on those kernels. We
don't mmap expert files, so this doesn't matter for now.
