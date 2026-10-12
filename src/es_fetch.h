/*
 * Byte ranges of remote files over HTTP(S), for models that are bigger than the device's storage.
 * Uses the `curl` command (Termux: pkg install curl; preinstalled on Linux, macOS and Windows 10+), so the engine needs
 * no TLS library. HF_TOKEN, if set, is sent for gated or private Hugging Face repositories.
 */
#ifndef ES_FETCH_H
#define ES_FETCH_H

#include <stdint.h>

/* size of a remote file in bytes (following redirects), or -1 */
int64_t es_http_size(const char *url);
/* bytes [off, off + len) of url, written to fd at out_off; 0 on success */
int     es_http_range(const char *url, uint64_t off, uint64_t len, int fd, uint64_t out_off);
/* Hugging Face download URL of a file in a repository */
void    es_hf_url(char *out, int cap, const char *repo, const char *file);

#endif
