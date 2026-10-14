# SparseForge in Stream-

This folder is the SparseForge phone/CPU trainer and MCP server, ready to be committed
to github.com/Tharunmaks/Stream- (agent must not push).

Suggested layout: keep as `sparseforge/` at the repo root, next to `craft/` and `src/`.
Implements the Expert Craft idea (dense core in RAM + hashed experts on disk) in pure
Python+NumPy for Termux, plus an optional PyTorch engine for PCs, and `sparseforge-mcp`.

See README.md and BENCHMARKS.md.
