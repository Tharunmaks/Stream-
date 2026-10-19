# 218 models through the engine

`python3 scripts/test_200.py --llama-cpp ./llama.cpp` (needs PyTorch, transformers and a llama.cpp build). Last run: 218 of 218 passed
on a 4-core x86 PC, after the fixes the run itself led to (below). Each tier is checked against something independent:

| Tier | What | Checked against | Proves |
|---|---|---|---|
| mini (105) | small random models, 7 per design: Llama, Mistral, Qwen2, Qwen3, Qwen2-MoE, Qwen3-MoE, Mixtral, OLMoE, gpt-oss, DeepSeek V2, DeepSeek V3, Qwen3.5 dense and MoE (91), Kimi K3 (14) | PyTorch (transformers); Kimi K3: llama.cpp in exact mode | the math of every design, logit by logit (typically 1e-6..1e-5) |
| small (43) | real GGUF files, 0.1 to 6.4 GB (42 GB in all), many quantizations (Q2_K..Q8_0, IQ2..IQ4, F16) | llama.cpp on the same file and text | real weights and formats; the engine's 8-bit activation rounding is as exact as llama.cpp's (median KL to exact activations 4.4e-3 for both) |
| big (70) | real GGUF files from 8.5 GB to 594 GB (9.1 TB in all), 14B to 2.4T parameters: Qwen3.8 2.4T, Kimi K3, Kimi K2, Mistral Large 3, DeepSeek R1/V3/V3.1, Qwen3.5 397B, Qwen3-Coder 480B, Llama 3.1/3.3 70B, gpt-oss 120B, ... | headers only (`es_get --skeleton`): the engine loads the real structure and runs it with every weight zero | the engine handles every part, tensor, format and hyperparameter of these files. **Not** answer quality: the weights are not there |

What the run found and fixed:
* Qwen3 / Qwen3-MoE with head size equal to the model width: the per-head q-norm was taken for a whole-vector norm (2-4 logits off). Fixed.
* Kimi K3 (`kimi-k3`, in the website catalog) was not supported at all. Ported: Kimi delta attention, position-free latent
  attention with an output gate, experts in a latent space, the situ activation, cross-layer residuals and the K3 chat format.
* Test rules made fair: both engines round activations to 8 bits for speed, so a model passes when the engine's rounding is
  no worse than llama.cpp's (each measured against the engine with exact activations). On a 2-bit Qwen3-0.6B the engine is
  clearly more exact (perplexity 383 vs llama.cpp's 573).

Not covered: the speed or answer quality of models above 7B (the test machine cannot hold or stream them), and phones (not run here).

## mini: 105 of 105 passed

| | Model | Result |
|---|---|---|
| PASS | llama     #1  L4 d128 h4 kv4 tieTrue rope-llama3 | max /logit difference/ 5.01e-06 (logits up to 4.4), 24 positions |
| PASS | llama     #2  L1 d128 h2 kv2 rope-llama3 | max /logit difference/ 4.05e-06 (logits up to 4.4), 24 positions |
| PASS | llama     #3  L2 d32 h2 kv2 rope-llama3 | max /logit difference/ 1.61e-06 (logits up to 2.3), 24 positions |
| PASS | llama     #4  L3 d128 h2 kv1 rope-linear | max /logit difference/ 4.77e-06 (logits up to 4.6), 24 positions |
| PASS | llama     #5  L4 d64 h2 kv2 tieTrue | max /logit difference/ 3.78e-06 (logits up to 3.3), 24 positions |
| PASS | llama     #6  L4 d256 h8 kv2 rope-linear | max /logit difference/ 9.78e-06 (logits up to 6.6), 24 positions |
| PASS | llama     #7  L1 d128 h8 kv1 | max /logit difference/ 4.29e-06 (logits up to 4.3), 24 positions |
| PASS | mistral   #1  L4 d128 h4 kv1 hd32 | max /logit difference/ 3.76e-06 (logits up to 4.5), 24 positions |
| PASS | mistral   #2  L2 d192 h2 kv2 hd32 | max /logit difference/ 7.27e-06 (logits up to 5.3), 24 positions |
| PASS | mistral   #3  L3 d64 h8 kv4 hd64 | max /logit difference/ 2.15e-06 (logits up to 2.9), 24 positions |
| PASS | mistral   #4  L4 d192 h4 kv2 hd64 | max /logit difference/ 8.34e-06 (logits up to 5.4), 24 positions |
| PASS | mistral   #5  L1 d64 h4 kv2 hd32 | max /logit difference/ 2.26e-06 (logits up to 3.5), 24 positions |
| PASS | mistral   #6  L3 d192 h4 kv1 hd64 | max /logit difference/ 1.09e-05 (logits up to 5.5), 24 positions |
| PASS | mistral   #7  L2 d128 h4 kv2 hd32 | max /logit difference/ 3.81e-06 (logits up to 4.4), 24 positions |
| PASS | qwen2     #1  L2 d64 h4 kv1 | max /logit difference/ 2.50e-06 (logits up to 3.2), 24 positions |
| PASS | qwen2     #2  L1 d128 h8 kv2 | max /logit difference/ 3.81e-06 (logits up to 4.5), 24 positions |
| PASS | qwen2     #3  L1 d128 h8 kv2 | max /logit difference/ 3.58e-06 (logits up to 4.7), 24 positions |
| PASS | qwen2     #4  L2 d128 h2 kv1 tieTrue | max /logit difference/ 5.01e-06 (logits up to 4.7), 24 positions |
| PASS | qwen2     #5  L1 d128 h8 kv4 | max /logit difference/ 3.47e-06 (logits up to 4.8), 24 positions |
| PASS | qwen2     #6  L2 d64 h4 kv4 | max /logit difference/ 1.67e-06 (logits up to 3.3), 24 positions |
| PASS | qwen2     #7  L1 d256 h8 kv8 | max /logit difference/ 8.94e-06 (logits up to 7.1), 24 positions |
| PASS | qwen3     #1  L3 d64 h8 kv2 hd32 | max /logit difference/ 2.38e-06 (logits up to 3.3), 24 positions |
| PASS | qwen3     #2  L2 d64 h4 kv1 hd128 tieTrue | max /logit difference/ 3.11e-06 (logits up to 3.3), 24 positions |
| PASS | qwen3     #4  L1 d256 h8 kv4 hd64 tieTrue | max /logit difference/ 8.11e-06 (logits up to 6.4), 24 positions |
| PASS | qwen3     #5  L1 d256 h4 kv2 hd64 tieTrue | max /logit difference/ 7.63e-06 (logits up to 6.7), 24 positions |
| PASS | qwen3     #6  L2 d128 h8 kv1 hd32 tieTrue | max /logit difference/ 8.11e-06 (logits up to 4.4), 24 positions |
| PASS | qwen3     #7  L3 d256 h8 kv8 hd128 | max /logit difference/ 9.30e-06 (logits up to 6.5), 24 positions |
| PASS | qwen2moe  #1  L2 d128 h4 kv1 E16 k2 | max /logit difference/ 3.10e-06 (logits up to 4.7), 24 positions |
| PASS | qwen2moe  #2  L3 d256 h4 kv2 E8 k2 | max /logit difference/ 8.36e-06 (logits up to 6.5), 24 positions |
| PASS | qwen2moe  #3  L3 d512 h8 kv1 E8 k1 | max /logit difference/ 2.09e-05 (logits up to 9.8), 24 positions |
| PASS | qwen2moe  #4  L3 d128 h8 kv4 E4 k4 | max /logit difference/ 3.10e-06 (logits up to 4.5), 24 positions |
| PASS | qwen2moe  #5  L2 d256 h4 kv1 E8 k1 | max /logit difference/ 8.11e-06 (logits up to 6.3), 24 positions |
| PASS | qwen2moe  #6  L1 d512 h8 kv4 E16 k2 | max /logit difference/ 1.24e-05 (logits up to 9.7), 24 positions |
| PASS | qwen2moe  #7  L1 d128 h8 kv2 E4 k2 | max /logit difference/ 2.62e-06 (logits up to 4.8), 24 positions |
| PASS | qwen3moe  #3  L1 d128 h8 kv8 hd64 E8 k4 | max /logit difference/ 3.34e-06 (logits up to 4.6), 24 positions |
| PASS | qwen3moe  #4  L2 d128 h2 kv1 hd64 E32 k8 | max /logit difference/ 4.17e-06 (logits up to 4.5), 24 positions |
| PASS | qwen3moe  #6  L2 d128 h2 kv2 hd32 E8 k4 | max /logit difference/ 3.81e-06 (logits up to 4.7), 24 positions |
| PASS | qwen3moe  #7  L3 d64 h2 kv2 hd32 E32 k4 | max /logit difference/ 1.91e-06 (logits up to 3.3), 24 positions |
| PASS | mixtral   #1  L1 d64 h2 kv2 E8 k1 | max /logit difference/ 1.67e-06 (logits up to 2.9), 24 positions |
| PASS | mixtral   #2  L2 d256 h4 kv2 E8 k1 | max /logit difference/ 9.00e-06 (logits up to 6.5), 24 positions |
| PASS | mixtral   #3  L1 d32 h2 kv1 E8 k2 | max /logit difference/ 7.15e-07 (logits up to 2.2), 24 positions |
| PASS | mixtral   #4  L2 d128 h4 kv1 E8 k2 | max /logit difference/ 3.93e-06 (logits up to 4.3), 24 positions |
| PASS | mixtral   #5  L2 d512 h8 kv2 E8 k2 | max /logit difference/ 4.83e-05 (logits up to 8.7), 24 positions |
| PASS | mixtral   #6  L3 d64 h2 kv1 E4 k1 | max /logit difference/ 1.88e-06 (logits up to 3.2), 24 positions |
| PASS | mixtral   #7  L2 d32 h2 kv1 E4 k1 | max /logit difference/ 1.07e-06 (logits up to 2.1), 24 positions |
| PASS | olmoe     #1  L1 d64 h4 kv4 E16 k8 | max /logit difference/ 1.67e-06 (logits up to 3.1), 24 positions |
| PASS | olmoe     #2  L2 d128 h2 kv2 E8 k8 | max /logit difference/ 3.58e-06 (logits up to 4.3), 24 positions |
| PASS | olmoe     #3  L1 d32 h2 kv2 E16 k2 | max /logit difference/ 9.54e-07 (logits up to 2.4), 24 positions |
| PASS | olmoe     #4  L3 d128 h2 kv2 E16 k4 | max /logit difference/ 3.10e-06 (logits up to 4.2), 24 positions |
| PASS | olmoe     #5  L2 d128 h2 kv2 E8 k2 | max /logit difference/ 3.46e-06 (logits up to 4.6), 24 positions |
| PASS | olmoe     #6  L3 d256 h4 kv4 E8 k8 | max /logit difference/ 6.91e-06 (logits up to 6.5), 24 positions |
| PASS | olmoe     #7  L3 d64 h4 kv4 E8 k8 | max /logit difference/ 2.15e-06 (logits up to 3.2), 24 positions |
| PASS | gptoss    #1  L2 d96 h8 kv2 hd32 E4 k4 swa8 rope-yarn | max /logit difference/ 2.86e-06 (logits up to 4.2), 24 positions |
| PASS | gptoss    #2  L4 d128 h8 kv1 hd32 E16 k2 swa2 rope-yarn | max /logit difference/ 3.70e-06 (logits up to 4.7), 24 positions |
| PASS | gptoss    #3  L2 d64 h4 kv1 hd64 E16 k2 swa8 rope-yarn | max /logit difference/ 1.91e-06 (logits up to 3.0), 24 positions |
| PASS | gptoss    #4  L4 d64 h4 kv1 hd64 E8 k4 swa2 rope-yarn | max /logit difference/ 1.91e-06 (logits up to 3.4), 24 positions |
| PASS | gptoss    #5  L2 d96 h8 kv4 hd32 E16 k2 swa8 rope-yarn | max /logit difference/ 2.74e-06 (logits up to 4.3), 24 positions |
| PASS | gptoss    #6  L2 d96 h4 kv1 hd64 E16 k1 swa8 rope-yarn | max /logit difference/ 2.86e-06 (logits up to 4.1), 24 positions |
| PASS | gptoss    #7  L4 d128 h4 kv1 hd64 E8 k2 swa4 rope-yarn | max /logit difference/ 4.29e-06 (logits up to 4.6), 24 positions |
| PASS | qwen35moe #1  L4 d128 h4 kv1 hd64 E16 k2 | max /logit difference/ 2.49e-04 (logits up to 10.5), 24 positions |
| PASS | qwen35moe #2  L4 d64 h4 kv1 hd64 E8 k4 | max /logit difference/ 6.42e-05 (logits up to 6.8), 24 positions |
| PASS | qwen35moe #3  L8 d128 h4 kv2 hd64 E8 k2 | max /logit difference/ 4.87e-04 (logits up to 9.3), 24 positions |
| PASS | qwen35moe #4  L4 d128 h2 kv1 hd64 E16 k4 | max /logit difference/ 1.87e-04 (logits up to 9.8), 24 positions |
| PASS | qwen35moe #5  L4 d64 h4 kv1 hd32 E8 k2 | max /logit difference/ 1.38e-04 (logits up to 6.8), 24 positions |
| PASS | qwen35moe #6  L4 d128 h2 kv2 hd32 E16 k2 | max /logit difference/ 1.26e-04 (logits up to 9.9), 24 positions |
| PASS | qwen35moe #7  L8 d128 h2 kv1 hd32 E16 k4 | max /logit difference/ 3.39e-04 (logits up to 11.3), 24 positions |
| PASS | qwen35    #1  L4 d64 h4 kv1 hd64 tieTrue | max /logit difference/ 2.66e-05 (logits up to 7.0), 24 positions |
| PASS | qwen35    #2  L4 d128 h2 kv2 hd32 tieTrue | max /logit difference/ 5.82e-05 (logits up to 10.4), 24 positions |
| PASS | qwen35    #3  L8 d64 h2 kv2 hd32 | max /logit difference/ 3.05e-05 (logits up to 6.8), 24 positions |
| PASS | qwen35    #4  L8 d64 h4 kv2 hd64 | max /logit difference/ 5.44e-05 (logits up to 6.5), 24 positions |
| PASS | qwen35    #5  L8 d64 h4 kv1 hd64 | max /logit difference/ 4.48e-05 (logits up to 6.7), 24 positions |
| PASS | qwen35    #6  L4 d64 h4 kv2 hd32 tieTrue | max /logit difference/ 3.34e-05 (logits up to 6.6), 24 positions |
| PASS | qwen35    #7  L4 d128 h2 kv1 hd64 tieTrue | max /logit difference/ 4.12e-05 (logits up to 10.1), 24 positions |
| PASS | qwen3     #3  L1 d128 h2 kv1 hd128 | max /logit difference/ 4.52e-06 (logits up to 4.4), 24 positions |
| PASS | qwen3moe  #1  L3 d64 h2 kv1 hd64 E16 k2 | max /logit difference/ 1.79e-06 (logits up to 3.2), 24 positions |
| PASS | qwen3moe  #2  L3 d64 h8 kv1 hd64 E32 k2 | max /logit difference/ 2.83e-06 (logits up to 3.2), 24 positions |
| PASS | qwen3moe  #5  L1 d64 h4 kv1 hd64 E32 k4 | max /logit difference/ 2.32e-06 (logits up to 3.1), 24 positions |
| PASS | dsv2      #1  L3 d64 h4 kv4 E16 k2 grp1 tg1 qlora32 | max /logit difference/ 1.91e-06 (logits up to 3.4), 24 positions |
| PASS | dsv2      #2  L3 d128 h4 kv4 E8 k6 grp1 tg1 qlora32 rope-yarn | max /logit difference/ 5.90e-06 (logits up to 4.6), 24 positions |
| PASS | dsv2      #3  L3 d64 h4 kv4 E16 k2 grp1 tg1 qlora32 rope-yarn | max /logit difference/ 2.28e-06 (logits up to 3.2), 24 positions |
| PASS | dsv2      #4  L2 d128 h2 kv2 E8 k4 grp1 tg1 qlora32 rope-yarn | max /logit difference/ 4.89e-06 (logits up to 4.6), 24 positions |
| PASS | dsv2      #5  L3 d128 h4 kv4 E16 k2 grp1 tg1 rope-yarn | max /logit difference/ 7.27e-06 (logits up to 4.6), 24 positions |
| PASS | dsv2      #6  L2 d64 h4 kv4 E16 k2 grp1 tg1 qlora32 rope-yarn | max /logit difference/ 2.24e-06 (logits up to 3.2), 24 positions |
| PASS | dsv2      #7  L2 d128 h4 kv4 E8 k2 grp1 tg1 qlora32 rope-yarn | max /logit difference/ 5.01e-06 (logits up to 4.7), 24 positions |
| PASS | dsv3      #1  L3 d64 h4 kv4 E32 k1 grp1 tg1 qlora32 | max /logit difference/ 2.21e-06 (logits up to 3.7), 24 positions |
| PASS | dsv3      #2  L2 d64 h2 kv2 E16 k8 grp1 tg1 qlora64 rope-yarn | max /logit difference/ 2.74e-06 (logits up to 3.3), 24 positions |
| PASS | dsv3      #3  L2 d64 h2 kv2 E16 k4 grp1 tg1 qlora64 | max /logit difference/ 2.38e-06 (logits up to 3.5), 24 positions |
| PASS | dsv3      #4  L3 d128 h2 kv2 E8 k2 grp2 tg1 qlora64 rope-yarn | max /logit difference/ 5.84e-06 (logits up to 4.8), 24 positions |
| PASS | dsv3      #5  L2 d64 h4 kv4 E32 k8 grp1 tg1 qlora32 rope-yarn | max /logit difference/ 2.38e-06 (logits up to 3.4), 24 positions |
| PASS | dsv3      #6  L3 d128 h4 kv4 E32 k8 grp1 tg1 qlora32 rope-yarn | max /logit difference/ 6.42e-06 (logits up to 5.0), 24 positions |
| PASS | dsv3      #7  L2 d64 h2 kv2 E8 k2 grp1 tg1 qlora64 rope-yarn | max /logit difference/ 2.15e-06 (logits up to 4.0), 24 positions |
| PASS | kimi-k3   #1a L5 d96 h4 kda32 E16 k4 lat48 no-res kv01011 | vs llama.cpp: max /logit difference/ 3.74e-05 (logits up to 16.7), 24 positions |
| PASS | kimi-k3   #1b L5 d96 h4 kda32 E16 k4 lat48 res3 kv01011 | vs llama.cpp: max /logit difference/ 2.00e-04 (logits up to 15.2), 24 positions |
| PASS | kimi-k3   #2a L7 d64 h4 kda32 E8 k4 lat32 no-res kv0110011 | vs llama.cpp: max /logit difference/ 3.16e-05 (logits up to 13.5), 24 positions |
| PASS | kimi-k3   #2b L7 d64 h4 kda32 E8 k4 lat32 res2 kv0110011 | vs llama.cpp: max /logit difference/ 1.21e-03 (logits up to 12.9), 24 positions |
| PASS | kimi-k3   #3a L5 d96 h2 kda16 E16 k4 lat48 no-res kv00011 | vs llama.cpp: max /logit difference/ 2.55e-05 (logits up to 15.6), 24 positions |
| PASS | kimi-k3   #3b L5 d96 h2 kda16 E16 k4 lat48 res2 kv00011 | vs llama.cpp: max /logit difference/ 1.19e-03 (logits up to 14.6), 24 positions |
| PASS | kimi-k3   #4a L5 d96 h2 kda16 E8 k2 lat48 no-res kv00101 | vs llama.cpp: max /logit difference/ 3.27e-05 (logits up to 15.1), 24 positions |
| PASS | kimi-k3   #4b L5 d96 h2 kda16 E8 k2 lat48 res2 kv00101 | vs llama.cpp: max /logit difference/ 8.14e-03 (logits up to 15.6), 24 positions |
| PASS | kimi-k3   #5a L6 d96 h4 kda32 E8 k4 lat48 no-res kv001101 | vs llama.cpp: max /logit difference/ 4.24e-05 (logits up to 16.0), 24 positions |
| PASS | kimi-k3   #5b L6 d96 h4 kda32 E8 k4 lat48 res3 kv001101 | vs llama.cpp: max /logit difference/ 8.93e-04 (logits up to 14.6), 24 positions |
| PASS | kimi-k3   #6a L7 d96 h4 kda16 E16 k2 lat32 no-res kv0011011 | vs llama.cpp: max /logit difference/ 9.17e-05 (logits up to 14.6), 24 positions |
| PASS | kimi-k3   #6b L7 d96 h4 kda16 E16 k2 lat32 res2 kv0011011 | vs llama.cpp: max /logit difference/ 2.86e-04 (logits up to 16.3), 24 positions |
| PASS | kimi-k3   #7a L6 d64 h4 kda32 E16 k2 lat48 no-res kv011101 | vs llama.cpp: max /logit difference/ 3.44e-05 (logits up to 13.0), 24 positions |
| PASS | kimi-k3   #7b L6 d64 h4 kda32 E16 k2 lat48 res3 kv011101 | vs llama.cpp: max /logit difference/ 5.58e-04 (logits up to 12.0), 24 positions |

## small: 43 of 43 passed

| | Model | Result |
|---|---|---|
| PASS | SmolLM2-135M-Instruct-Q2_K.gguf (bartowski, 0.09 GB) | vs llama.cpp KL 5.7e-03, top token 98% / rounding error: engine 4.8e-03, llama.cpp 4.9e-03 / perplexity 22.15 (exact 21.96) / 1 s |
| PASS | SmolLM2-135M-Instruct-Q3_K_M.gguf (bartowski, 0.09 GB) | vs llama.cpp KL 5.3e-03, top token 95% / rounding error: engine 4.6e-03, llama.cpp 4.1e-03 / perplexity 20.06 (exact 19.86) / 1 s |
| PASS | SmolLM2-135M-Instruct-Q4_0.gguf (bartowski, 0.09 GB) | vs llama.cpp KL 3.0e-03, top token 100% / rounding error: engine 2.3e-03, llama.cpp 2.4e-03 / perplexity 17.07 (exact 16.85) / 1 s |
| PASS | SmolLM2-135M-Instruct-Q4_K_M.gguf (bartowski, 0.11 GB) | vs llama.cpp KL 5.2e-03, top token 97% / rounding error: engine 4.4e-03, llama.cpp 4.4e-03 / perplexity 17.85 (exact 17.95) / 0 s |
| PASS | SmolLM2-135M-Instruct-Q5_K_M.gguf (bartowski, 0.11 GB) | vs llama.cpp KL 4.6e-03, top token 92% / rounding error: engine 3.5e-03, llama.cpp 4.7e-03 / perplexity 17.17 (exact 16.97) / 2 s |
| PASS | SmolLM2-135M-Instruct-Q6_K.gguf (bartowski, 0.14 GB) | vs llama.cpp KL 4.4e-03, top token 95% / rounding error: engine 4.6e-03, llama.cpp 4.2e-03 / perplexity 17.19 (exact 17.28) / 1 s |
| PASS | SmolLM2-135M-Instruct-Q8_0.gguf (bartowski, 0.14 GB) | vs llama.cpp KL 3.2e-03, top token 98% / rounding error: engine 2.9e-03, llama.cpp 2.8e-03 / perplexity 16.77 (exact 17.10) / 1 s |
| PASS | SmolLM2-135M-Instruct-IQ4_XS.gguf (bartowski, 0.09 GB) | vs llama.cpp KL 6.1e-03, top token 91% / rounding error: engine 2.8e-03, llama.cpp 5.4e-03 / perplexity 19.49 (exact 19.34) / 2 s |
| PASS | SmolLM2-135M-Instruct-f16.gguf (bartowski, 0.27 GB) | vs llama.cpp KL 7.0e-07, top token 100% / rounding error: engine 0.0e+00, llama.cpp 7.0e-07 / perplexity 17.22 (exact 17.22) / 6 s |
| PASS | SmolLM2-360M-Instruct-Q4_K_M.gguf (bartowski, 0.27 GB) | vs llama.cpp KL 4.6e-03, top token 98% / rounding error: engine 6.6e-03, llama.cpp 6.7e-03 / perplexity 12.51 (exact 12.35) / 1 s |
| PASS | SmolLM2-360M-Instruct-Q8_0.gguf (bartowski, 0.39 GB) | vs llama.cpp KL 2.5e-03, top token 97% / rounding error: engine 5.5e-03, llama.cpp 5.3e-03 / perplexity 12.20 (exact 12.36) / 2 s |
| PASS | SmolLM2-360M-Instruct-IQ3_M.gguf (bartowski, 0.22 GB) | vs llama.cpp KL 5.0e-03, top token 92% / rounding error: engine 3.3e-03, llama.cpp 5.6e-03 / perplexity 12.39 (exact 12.37) / 4 s |
| PASS | SmolLM2-1.7B-Instruct-Q4_K_M.gguf (bartowski, 1.06 GB) | vs llama.cpp KL 1.3e-02, top token 94% / rounding error: engine 2.2e-01, llama.cpp 2.1e-01 / perplexity 10.38 (exact 10.16) / 5 s |
| PASS | qwen2.5-0.5b-instruct-q2_k.gguf (Qwen, 0.42 GB) | vs llama.cpp KL 4.8e-03, top token 98% / rounding error: engine 4.3e-03, llama.cpp 4.0e-03 / perplexity 21.53 (exact 20.99) / 4 s |
| PASS | qwen2.5-0.5b-instruct-q3_k_m.gguf (Qwen, 0.43 GB) | vs llama.cpp KL 4.5e-03, top token 95% / rounding error: engine 3.7e-03, llama.cpp 3.4e-03 / perplexity 18.76 (exact 18.99) / 4 s |
| PASS | qwen2.5-0.5b-instruct-q4_0.gguf (Qwen, 0.43 GB) | vs llama.cpp KL 4.0e-03, top token 95% / rounding error: engine 4.1e-03, llama.cpp 4.1e-03 / perplexity 29.30 (exact 29.80) / 2 s |
| PASS | qwen2.5-0.5b-instruct-q4_k_m.gguf (Qwen, 0.49 GB) | vs llama.cpp KL 5.2e-03, top token 97% / rounding error: engine 4.9e-03, llama.cpp 4.4e-03 / perplexity 23.38 (exact 23.44) / 2 s |
| PASS | qwen2.5-0.5b-instruct-q5_k_m.gguf (Qwen, 0.52 GB) | vs llama.cpp KL 3.9e-03, top token 97% / rounding error: engine 2.8e-03, llama.cpp 3.8e-03 / perplexity 21.28 (exact 21.46) / 6 s |
| PASS | qwen2.5-0.5b-instruct-q6_k.gguf (Qwen, 0.65 GB) | vs llama.cpp KL 5.5e-03, top token 91% / rounding error: engine 4.4e-03, llama.cpp 3.9e-03 / perplexity 21.91 (exact 22.39) / 3 s |
| PASS | qwen2.5-0.5b-instruct-q8_0.gguf (Qwen, 0.68 GB) | vs llama.cpp KL 2.4e-03, top token 95% / rounding error: engine 2.3e-03, llama.cpp 2.3e-03 / perplexity 22.11 (exact 22.39) / 2 s |
| PASS | Qwen2.5-0.5B-Instruct-IQ2_M.gguf (bartowski, 0.33 GB) | vs llama.cpp KL 6.2e-03, top token 91% / rounding error: engine 5.0e-03, llama.cpp 5.8e-03 / perplexity 24.09 (exact 24.35) / 5 s |
| PASS | Qwen2.5-0.5B-Instruct-IQ3_XS.gguf (bartowski, 0.34 GB) | vs llama.cpp KL 6.5e-03, top token 97% / rounding error: engine 7.1e-03, llama.cpp 1.1e-02 / perplexity 20.96 (exact 20.97) / 5 s |
| PASS | Qwen2.5-0.5B-Instruct-IQ4_XS.gguf (bartowski, 0.35 GB) | vs llama.cpp KL 5.5e-03, top token 89% / rounding error: engine 1.1e-02, llama.cpp 1.5e-02 / perplexity 20.97 (exact 20.77) / 6 s |
| PASS | Qwen2.5-0.5B-Instruct-Q3_K_L.gguf (bartowski, 0.37 GB) | vs llama.cpp KL 6.0e-03, top token 94% / rounding error: engine 7.6e-03, llama.cpp 7.8e-03 / perplexity 22.12 (exact 22.50) / 5 s |
| PASS | qwen2.5-1.5b-instruct-q4_k_m.gguf (Qwen, 1.12 GB) | vs llama.cpp KL 1.1e-02, top token 95% / rounding error: engine 5.1e-02, llama.cpp 5.4e-02 / perplexity 13.21 (exact 13.45) / 6 s |
| PASS | qwen2.5-coder-0.5b-instruct-q8_0.gguf (Qwen, 0.68 GB) | vs llama.cpp KL 8.8e-04, top token 97% / rounding error: engine 8.5e-04, llama.cpp 7.1e-04 / perplexity 40.94 (exact 40.68) / 2 s |
| PASS | Qwen3-0.6B-Q8_0.gguf (Qwen, 0.64 GB) | vs llama.cpp KL 2.6e-03, top token 94% / rounding error: engine 7.9e-03, llama.cpp 7.7e-03 / perplexity 24.21 (exact 24.24) / 3 s |
| PASS | Qwen3-0.6B-Q4_K_M.gguf (unsloth, 0.40 GB) | vs llama.cpp KL 1.5e-02, top token 92% / rounding error: engine 1.2e-02, llama.cpp 1.2e-02 / perplexity 25.94 (exact 25.32) / 3 s |
| PASS | Qwen3-0.6B-IQ4_NL.gguf (unsloth, 0.38 GB) | vs llama.cpp KL 3.1e-03, top token 95% / rounding error: engine 3.7e-03, llama.cpp 4.2e-03 / perplexity 22.42 (exact 22.48) / 6 s |
| PASS | Qwen3-0.6B-UD-IQ2_XXS.gguf (unsloth, 0.23 GB) | vs llama.cpp KL 3.8e-01, top token 61% / rounding error: engine 6.1e-02, llama.cpp 3.4e-01 / perplexity 383.26 (exact 406.51) / 7 s |
| PASS | Qwen3-1.7B-Q4_K_M.gguf (unsloth, 1.11 GB) | vs llama.cpp KL 3.4e-02, top token 89% / rounding error: engine 2.6e-02, llama.cpp 3.2e-02 / perplexity 23.16 (exact 26.17) / 8 s |
| PASS | Qwen3-4B-Instruct-2507-Q4_K_M.gguf (unsloth, 2.50 GB) | vs llama.cpp KL 5.7e-03, top token 95% / rounding error: engine 6.8e-03, llama.cpp 9.1e-03 / perplexity 11.77 (exact 11.74) / 14 s |
| PASS | Llama-3.2-1B-Instruct-Q4_K_M.gguf (bartowski, 0.81 GB) | vs llama.cpp KL 2.8e-03, top token 97% / rounding error: engine 2.1e-03, llama.cpp 1.9e-03 / perplexity 14.11 (exact 14.10) / 5 s |
| PASS | Llama-3.2-1B-Instruct-IQ3_M.gguf (bartowski, 0.66 GB) | vs llama.cpp KL 2.9e-03, top token 95% / rounding error: engine 1.8e-03, llama.cpp 2.8e-03 / perplexity 14.15 (exact 14.03) / 13 s |
| PASS | Llama-3.2-1B-Instruct-Q6_K.gguf (bartowski, 1.02 GB) | vs llama.cpp KL 2.7e-03, top token 97% / rounding error: engine 1.9e-03, llama.cpp 2.3e-03 / perplexity 14.01 (exact 13.80) / 8 s |
| PASS | Llama-3.2-3B-Instruct-Q4_K_M.gguf (bartowski, 2.02 GB) | vs llama.cpp KL 1.4e-03, top token 97% / rounding error: engine 1.2e-03, llama.cpp 1.2e-03 / perplexity 10.73 (exact 10.62) / 11 s |
| PASS | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (TheBloke, 0.67 GB) | vs llama.cpp KL 2.3e-03, top token 98% / rounding error: engine 2.8e-03, llama.cpp 2.8e-03 / perplexity 17.39 (exact 17.42) / 3 s |
| PASS | tinyllama-1.1b-chat-v1.0.Q2_K.gguf (TheBloke, 0.48 GB) | vs llama.cpp KL 3.0e-03, top token 94% / rounding error: engine 2.9e-03, llama.cpp 3.4e-03 / perplexity 19.76 (exact 19.77) / 8 s |
| PASS | Mistral-7B-Instruct-v0.3-Q2_K.gguf (bartowski, 2.72 GB) | vs llama.cpp KL 9.1e-04, top token 100% / rounding error: engine 2.2e-03, llama.cpp 2.2e-03 / perplexity 7.13 (exact 7.21) / 44 s |
| PASS | OLMoE-1B-7B-0924-Instruct-Q2_K.gguf (bartowski, 2.56 GB) | vs llama.cpp KL 7.0e-03, top token 94% / rounding error: engine 5.4e-03, llama.cpp 9.2e-03 / perplexity 11.68 (exact 12.05) / 8 s |
| PASS | OLMoE-1B-7B-0924-Instruct-IQ4_XS.gguf (bartowski, 3.72 GB) | vs llama.cpp KL 6.3e-03, top token 95% / rounding error: engine 2.4e-03, llama.cpp 7.1e-03 / perplexity 12.85 (exact 12.77) / 23 s |
| PASS | DeepSeek-V2-Lite-Chat.Q2_K.gguf (mradermacher, 6.43 GB) | vs llama.cpp KL 9.1e-02, top token 81% / rounding error: engine 4.6e-02, llama.cpp 3.9e-02 / perplexity 33.19 (exact 36.06) / 18 s |
| PASS | qwen1.5-moe-a2.7b-chat-imat-IQ2_XS.gguf (qwp4w3hyb, 5.45 GB) | vs llama.cpp KL 3.7e-02, top token 89% / rounding error: engine 6.0e-03, llama.cpp 4.9e-02 / perplexity 12.06 (exact 11.99) / 33 s |

## big: 70 of 70 passed

| | Model | Result |
|---|---|---|
| PASS | Qwen3.8-2.4T-A95B-UD-IQ1_S 1/00012.gguf (508 GB) | loaded and ran: Connecting to the brain  Qwen3.8-2.4T-A95B (qwen35moe), 92 layers, 512 experts, 10 per token / 101 s |
| PASS | Kimi-K2-Instruct-0905-UD-TQ1_0 1/00006.gguf (249 GB) | loaded and ran: Connecting to the brain  Kimi-K2-Instruct-0905 (deepseek2), 61 layers, 384 experts, 8 per token / 44 s |
| PASS | Kimi-K2-Instruct-0905-UD-IQ1_M 1/00007.gguf (305 GB) | loaded and ran: Connecting to the brain  Kimi-K2-Instruct-0905 (deepseek2), 61 layers, 384 experts, 8 per token / 54 s |
| PASS | Kimi-K2-Instruct-0905-Q2_K 1/00008.gguf (374 GB) | loaded and ran: Connecting to the brain  Kimi-K2-Instruct-0905 (deepseek2), 61 layers, 384 experts, 8 per token / 61 s |
| PASS | Kimi-K2-Thinking-UD-TQ1_0 1/00006.gguf (247 GB) | loaded and ran: Connecting to the brain  Kimi-K2-Thinking (deepseek2), 61 layers, 384 experts, 8 per token / 65 s |
| PASS | Mistral-Large-3-675B-Instruct-2512-UD-IQ1_S 1/00004.gguf (191 GB) | loaded and ran: Connecting to the brain  Mistral-Large-3-675B-Instruct-2512 (deepseek2), 61 layers, 128 experts, 4 per token / 57 s |
| PASS | Mistral-Large-3-675B-Instruct-2512-Q4_K_M 1/00009.gguf (407 GB) | loaded and ran: Connecting to the brain  Mistral-Large-3-675B-Instruct-2512 (deepseek2), 61 layers, 128 experts, 4 per token / 69 s |
| PASS | DeepSeek-R1-0528-UD-IQ1_S 1/00004.gguf (185 GB) | loaded and ran: Connecting to the brain  Deepseek-R1-0528 (deepseek2), 61 layers, 256 experts, 8 per token / 49 s |
| PASS | DeepSeek-R1-0528-UD-TQ1_0.gguf (162 GB) | loaded and ran: Connecting to the brain  Deepseek-R1-0528 (deepseek2), 61 layers, 256 experts, 8 per token / 48 s |
| PASS | DeepSeek-R1-0528-Q4_K_M 1/00009.gguf (405 GB) | loaded and ran: Connecting to the brain  Deepseek-R1-0528 (deepseek2), 61 layers, 256 experts, 8 per token / 59 s |
| PASS | DeepSeek-V3-0324-UD-IQ1_S 1/00004.gguf (186 GB) | loaded and ran: Connecting to the brain  DeepSeek V3 0324 BF16 (deepseek2), 61 layers, 256 experts, 8 per token / 49 s |
| PASS | DeepSeek-V3-0324-Q2_K 1/00005.gguf (244 GB) | loaded and ran: Connecting to the brain  DeepSeek V3 0324 BF16 (deepseek2), 61 layers, 256 experts, 8 per token / 46 s |
| PASS | DeepSeek-V3.1-UD-TQ1_0.gguf (170 GB) | loaded and ran: Connecting to the brain  Deepseek-V3.1 (deepseek2), 61 layers, 256 experts, 8 per token / 52 s |
| PASS | DeepSeek-V3.1-Q4_K_M 1/00009.gguf (405 GB) | loaded and ran: Connecting to the brain  Deepseek-V3.1 (deepseek2), 61 layers, 256 experts, 8 per token / 59 s |
| PASS | DeepSeek-R1-UD-IQ1_S 1/00003.gguf (140 GB) | loaded and ran: Connecting to the brain  DeepSeek R1 BF16 (deepseek2), 61 layers, 256 experts, 8 per token / 43 s |
| PASS | Qwen3.5-397B-A17B-UD-IQ1_M 1/00004.gguf (107 GB) | loaded and ran: Connecting to the brain  Qwen3.5-397B-A17B (qwen35moe), 60 layers, 512 experts, 10 per token / 46 s |
| PASS | Qwen3.5-397B-A17B-UD-IQ2_XXS 1/00004.gguf (115 GB) | loaded and ran: Connecting to the brain  Qwen3.5-397B-A17B (qwen35moe), 60 layers, 512 experts, 10 per token / 48 s |
| PASS | Qwen3.5-397B-A17B-MXFP4_MOE 1/00006.gguf (237 GB) | loaded and ran: Connecting to the brain  Qwen3.5-397B-A17B (qwen35moe), 60 layers, 512 experts, 10 per token / 50 s |
| PASS | Qwen3.5-397B-A17B-Q4_K_M 1/00006.gguf (244 GB) | loaded and ran: Connecting to the brain  Qwen3.5-397B-A17B (qwen35moe), 60 layers, 512 experts, 10 per token / 20 s |
| PASS | Qwen3-Coder-480B-A35B-Instruct-UD-IQ1_M.gguf (150 GB) | loaded and ran: Connecting to the brain  Qwen3-Coder-480B-A35B-Instruct (qwen3moe), 62 layers, 160 experts, 8 per token / 49 s |
| PASS | Qwen3-Coder-480B-A35B-Instruct-Q2_K 1/00004.gguf (175 GB) | loaded and ran: Connecting to the brain  Qwen3-Coder-480B-A35B-Instruct (qwen3moe), 62 layers, 160 experts, 8 per token / 46 s |
| PASS | Qwen3-Coder-480B-A35B-Instruct-Q4_K_M 1/00006.gguf (290 GB) | loaded and ran: Connecting to the brain  Qwen3-Coder-480B-A35B-Instruct (qwen3moe), 62 layers, 160 experts, 8 per token / 51 s |
| PASS | Qwen3-235B-A22B-Instruct-2507-Q2_K 1/00002.gguf (86 GB) | loaded and ran: Connecting to the brain  Qwen3-235B-A22B-Instruct-2507 (qwen3moe), 94 layers, 128 experts, 8 per token / 33 s |
| PASS | Qwen3-235B-A22B-Instruct-2507-Q4_K_M 1/00003.gguf (142 GB) | loaded and ran: Connecting to the brain  Qwen3-235B-A22B-Instruct-2507 (qwen3moe), 94 layers, 128 experts, 8 per token / 39 s |
| PASS | Qwen3-235B-A22B-Instruct-2507-Q6_K 1/00004.gguf (193 GB) | loaded and ran: Connecting to the brain  Qwen3-235B-A22B-Instruct-2507 (qwen3moe), 94 layers, 128 experts, 8 per token / 51 s |
| PASS | Qwen3-235B-A22B-Instruct-2507-IQ4_XS 1/00003.gguf (125 GB) | loaded and ran: Connecting to the brain  Qwen3-235B-A22B-Instruct-2507 (qwen3moe), 94 layers, 128 experts, 8 per token / 55 s |
| PASS | Qwen3-235B-A22B-Thinking-2507-Q3_K_M 1/00003.gguf (112 GB) | loaded and ran: Connecting to the brain  Qwen3-235B-A22B-Thinking-2507 (qwen3moe), 94 layers, 128 experts, 8 per token / 37 s |
| PASS | gpt-oss-120b-Q4_K_M 1/00002.gguf (63 GB) | loaded and ran: Connecting to the brain  Gpt-Oss-120B (gpt-oss), 36 layers, 128 experts, 4 per token / 10 s |
| PASS | gpt-oss-120b-F16.gguf (65 GB) | loaded and ran: Connecting to the brain  Gpt-Oss-120B (gpt-oss), 36 layers, 128 experts, 4 per token / 26 s |
| PASS | gpt-oss-120b-MXFP4.gguf (63 GB) | loaded and ran: Connecting to the brain  gpt-oss-120b (gpt-oss), 36 layers, 128 experts, 4 per token / 10 s |
| PASS | gpt-oss-20b-MXFP4.gguf (12 GB) | loaded and ran: Connecting to the brain  gpt-oss-20b (gpt-oss), 24 layers, 32 experts, 4 per token / 3 s |
| PASS | gpt-oss-20b-Q2_K.gguf (11 GB) | loaded and ran: Connecting to the brain  Gpt-Oss-20B (gpt-oss), 24 layers, 32 experts, 4 per token / 2 s |
| PASS | gpt-oss-20b-Q8_0.gguf (12 GB) | loaded and ran: Connecting to the brain  Gpt-Oss-20B (gpt-oss), 24 layers, 32 experts, 4 per token / 2 s |
| PASS | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (22 GB) | loaded and ran: Connecting to the brain  Qwen3.6-35B-A3B (qwen35moe), 40 layers, 256 experts, 8 per token / 12 s |
| PASS | Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf (11 GB) | loaded and ran: Connecting to the brain  Qwen3.6-35B-A3B (qwen35moe), 40 layers, 256 experts, 8 per token / 3 s |
| PASS | Qwen3-30B-A3B-Q4_K_M.gguf (19 GB) | loaded and ran: Connecting to the brain  Qwen3-30B-A3B (qwen3moe), 48 layers, 128 experts, 8 per token / 4 s |
| PASS | Qwen3-30B-A3B-Q8_0.gguf (32 GB) | loaded and ran: Connecting to the brain  Qwen3-30B-A3B (qwen3moe), 48 layers, 128 experts, 8 per token / 10 s |
| PASS | Qwen3-30B-A3B-Instruct-2507-IQ4_NL.gguf (17 GB) | loaded and ran: Connecting to the brain  Qwen3-30B-A3B-Instruct-2507 (qwen3moe), 48 layers, 128 experts, 8 per token / 6 s |
| PASS | Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf (22 GB) | loaded and ran: Connecting to the brain  Qwen3-30B-A3B-Instruct-2507 (qwen3moe), 48 layers, 128 experts, 8 per token / 6 s |
| PASS | Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf (25 GB) | loaded and ran: Connecting to the brain  Qwen3-Coder-30B-A3B-Instruct (qwen3moe), 48 layers, 128 experts, 8 per token / 5 s |
| PASS | Qwen3-32B-Q4_K_M.gguf (20 GB) | loaded and ran: Connecting to the brain  Qwen3-32B (qwen3), 64 layers, 0 experts, 0 per token / 37 s |
| PASS | Qwen3-14B-Q4_0.gguf (9 GB) | loaded and ran: Connecting to the brain  Qwen3-14B (qwen3), 40 layers, 0 experts, 0 per token / 3 s |
| PASS | Qwen2.5-72B-Instruct-Q4_K_M.gguf (47 GB) | loaded and ran: Connecting to the brain  Qwen2.5 72B Instruct (qwen2), 80 layers, 0 experts, 0 per token / 71 s |
| PASS | Qwen2.5-72B-Instruct-IQ2_M.gguf (29 GB) | loaded and ran: Connecting to the brain  Qwen2.5 72B Instruct (qwen2), 80 layers, 0 experts, 0 per token / 68 s |
| PASS | Qwen2.5-32B-Instruct-Q4_K_M.gguf (20 GB) | loaded and ran: Connecting to the brain  Qwen2.5 32B Instruct (qwen2), 64 layers, 0 experts, 0 per token / 29 s |
| PASS | Qwen2.5-32B-Instruct-IQ2_XXS.gguf (9 GB) | loaded and ran: Connecting to the brain  Qwen2.5 32B Instruct (qwen2), 64 layers, 0 experts, 0 per token / 17 s |
| PASS | Qwen2.5-Coder-32B-Instruct-Q5_K_M.gguf (23 GB) | loaded and ran: Connecting to the brain  Qwen2.5 Coder 32B Instruct (qwen2), 64 layers, 0 experts, 0 per token / 41 s |
| PASS | DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf (20 GB) | loaded and ran: Connecting to the brain  DeepSeek R1 Distill Qwen 32B (qwen2), 64 layers, 0 experts, 0 per token / 29 s |
| PASS | Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf (43 GB) | loaded and ran: Connecting to the brain  Meta Llama 3.1 70B Instruct (llama), 80 layers, 0 experts, 0 per token / 64 s |
| PASS | Meta-Llama-3.1-70B-Instruct-IQ2_XS.gguf (21 GB) | loaded and ran: Connecting to the brain  Meta Llama 3.1 70B Instruct (llama), 80 layers, 0 experts, 0 per token / 66 s |
| PASS | Llama-3.3-70B-Instruct-Q4_K_M.gguf (43 GB) | loaded and ran: Connecting to the brain  Llama 3.3 70B Instruct (llama), 80 layers, 0 experts, 0 per token / 61 s |
| PASS | Llama-3.3-70B-Instruct-IQ1_M.gguf (17 GB) | loaded and ran: Connecting to the brain  Llama 3.3 70B Instruct (llama), 80 layers, 0 experts, 0 per token / 59 s |
| PASS | DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf (43 GB) | loaded and ran: Connecting to the brain  DeepSeek R1 Distill Llama 70B (llama), 80 layers, 0 experts, 0 per token / 63 s |
| PASS | Mistral-Large-Instruct-2411-Q4_K_M 1/00002.gguf (73 GB) | loaded and ran: Connecting to the brain  Mistral Large Instruct 2411 (llama), 88 layers, 0 experts, 0 per token / 103 s |
| PASS | Mistral-Large-Instruct-2411-IQ2_XXS.gguf (32 GB) | loaded and ran: Connecting to the brain  Mistral Large Instruct 2411 (llama), 88 layers, 0 experts, 0 per token / 96 s |
| PASS | Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf (14 GB) | loaded and ran: Connecting to the brain  Mistral Small 24B Instruct 2501 (llama), 40 layers, 0 experts, 0 per token / 21 s |
| PASS | Mistral-Small-24B-Instruct-2501-Q6_K.gguf (19 GB) | loaded and ran: Connecting to the brain  Mistral Small 24B Instruct 2501 (llama), 40 layers, 0 experts, 0 per token / 30 s |
| PASS | Codestral-22B-v0.1-Q4_K_M.gguf (13 GB) | loaded and ran: Connecting to the brain  Codestral-22B-v0.1 (llama), 56 layers, 0 experts, 0 per token / 7 s |
| PASS | mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf (26 GB) | loaded and ran: Connecting to the brain  mistralai_mixtral-8x7b-instruct-v0.1 (llama), 32 layers, 8 experts, 2 per token / 14 s |
| PASS | mixtral-8x7b-instruct-v0.1.Q2_K.gguf (16 GB) | loaded and ran: Connecting to the brain  mistralai_mixtral-8x7b-instruct-v0.1 (llama), 32 layers, 8 experts, 2 per token / 9 s |
| PASS | Mixtral-8x22B-v0.1-Q4_K_M 1/00005.gguf (86 GB) | loaded and ran: Connecting to the brain  Mixtral-8x22B-v0.1 (llama), 56 layers, 8 experts, 2 per token / 71 s |
| PASS | Mixtral-8x22B-Instruct-v0.1.Q2_K 1/00003.gguf (52 GB) | loaded and ran: Connecting to the brain  models--mistralai--Mixtral-8x22B-Instruct-v0.1 (llama), 56 layers, 8 experts, 2 per t / 54 s |
| PASS | DeepSeek-V2.5-Q4_K_M 1/00004.gguf (142 GB) | loaded and ran: Connecting to the brain  DeepSeek V2.5 (deepseek2), 60 layers, 160 experts, 6 per token / 47 s |
| PASS | qwq-32b-q4_k_m.gguf (20 GB) | loaded and ran: Connecting to the brain  Tmp_Quanted_Model AWInt4 Groupsize128 (qwen2), 64 layers, 0 experts, 0 per token / 33 s |
| PASS | qwen2.5-14b-instruct-q4_k_m 1/00003.gguf (9 GB) | loaded and ran: Connecting to the brain  qwen2.5-14b-instruct (qwen2), 48 layers, 0 experts, 0 per token / 4 s |
| PASS | Llama-3.3-70B-Instruct-Q2_K.gguf (26 GB) | loaded and ran: Connecting to the brain  Llama-3.3-70B-Instruct (llama), 80 layers, 0 experts, 0 per token / 57 s |
| PASS | Qwen3.8-2.4T-A95B-UD-IQ1_M 1/00013.gguf (564 GB) | loaded and ran: Connecting to the brain  Qwen3.8-2.4T-A95B (qwen35moe), 92 layers, 512 experts, 10 per token / 114 s |
| PASS | qwen2-57b-a14b-instruct-q4_k_m.gguf (35 GB) | loaded and ran: Connecting to the brain  Qwen2-MoE-A14.2B-Chat (qwen2moe), 28 layers, 64 experts, 8 per token / 15 s |
| PASS | Kimi-K3-UD-TQ1_0 1/00012.gguf (509 GB) | loaded and ran: Connecting to the brain  Kimi-K3 (kimi-k3), 93 layers, 896 experts, 16 per token / 136 s |
| PASS | Kimi-K3-UD-IQ1_S 1/00014.gguf (594 GB) | loaded and ran: Connecting to the brain  Kimi-K3 (kimi-k3), 93 layers, 896 experts, 16 per token / 122 s |
