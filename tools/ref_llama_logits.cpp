// Reference logits from llama.cpp, for scripts/test_200.py: ref_llama_logits MODEL.gguf OUT.bin ID ID ...
// (every position, one token per decode as es_chat -D does). Built by the test against a llama.cpp build.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char **argv) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model *m = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 4; if (getenv("KVF32")) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; }   // exact: no 16-bit rounding
    llama_context *ctx = llama_init_from_model(m, cp);
    std::vector<llama_token> toks; for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    FILE *f = fopen(argv[2], "wb");
    for (size_t i = 0; i < toks.size(); i++) {   // one token per decode, like es_chat
        llama_batch b = llama_batch_get_one(&toks[i], 1);
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        fwrite(llama_get_logits_ith(ctx, -1), sizeof(float), nv, f);
    }
    fclose(f);
    return 0;
}
