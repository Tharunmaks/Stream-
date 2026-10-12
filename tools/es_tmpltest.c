/* Print how the engine formats a test conversation for a model's chat template (read from a GGUF; no weights are loaded).
 *   es_tmpltest MODEL.gguf [with-system]   -> kind on stderr, rendered text on stdout */
#include <stdio.h>
#include <string.h>

#include "../src/es_engine.h"
#include "../src/es_gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: es_tmpltest MODEL.gguf [with-system]\n"); return 2; }
    es_gguf g; const char *err;
    if (es_gguf_open(&g, argv[1], &err)) { fprintf(stderr, "%s\n", err); return 2; }
    const char *t = es_gguf_str(&g, "tokenizer.chat_template");
    const es_msg conv[] = {{"system", "You are a helpful assistant."}, {"user", "Hi, who are you?"}, {"assistant", "I am a language model."}, {"user", "What is 2+2?"}};
    const int sys = argc > 2;
    static char out[1 << 16];
    const int n = es_engine_render_with(t, conv + (sys ? 0 : 1), sys ? 4 : 3, out, sizeof out);
    fprintf(stderr, "%s\n", es_engine_template_kind(t));
    if (n < 0) return 1;
    fwrite(out, 1, (size_t)n, stdout);
    return 0;
}
