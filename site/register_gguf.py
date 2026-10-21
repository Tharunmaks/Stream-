#!/usr/bin/env python3
"""Register a GGUF chat model in the shared model library: register_gguf.py NAME PATH TOTAL_PARAMS"""
import sys, es_common as C
C.db(); name, path, n = sys.argv[1], sys.argv[2], float(sys.argv[3])
if C.get_model(name): print("exists", C.get_model(name)["id"]); sys.exit()
print(C.register_model(None, name, path, dict(total_params=n, active_params=n, materialized=True), "gguf"))
