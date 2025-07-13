\# Embedding Research

\## Performance Benchmarks for Ollama Embeddings



Empirical data on Ollama’s embedding latency is scarce, but community reports indicate local embedding can be \*\*very slow on CPU\*\*. For example, one user found generating embeddings for a 120 KB text file took \*almost an hour\* on an i7-9850H CPU with a modest GPU (Quadro RTX 3000) – hardware utilization barely exceeded 5% during embedding. Even using a high-end GPU (NVIDIA RTX 3090), a 100-page document still took \\~15 minutes to embed; GPU utilization was low (<20%) indicating a bottleneck elsewhere. These anecdotes suggest Ollama’s embedding (via `mxbai-embed-large`) is often \*\*memory-bandwidth limited\*\*, not purely compute-bound. In one test on the 1.25 GB (fp32) mxbai model, an H100 GPU was only \\~20% faster than a 4090 at the same batch size. By contrast, using FP16 precision on the 4090 gave \\~2.8× speedup. In other words, advanced GPUs can help, but not magically – optimally-sized batches and reduced precision yield far more benefit than raw compute.



\*\*CPU vs GPU:\*\*  Ollama automatically uses any available GPU (NVIDIA, AMD, Apple) to accelerate inference. In practice, a GPU can drastically reduce latency. For example, one user reported 100% GPU and near-memory-saturation on a 4090, whereas CPU-only embedding was glacial. If no GPU is available, expect multi-second-per-chunk latencies or worse. In contrast, a modern GPU (with mixed-precision) might embed a short sentence in a few hundred milliseconds, but longer texts or larger batches will exceed 200 ms by a wide margin. \*\*Sub-200 ms per 1024-dim embedding\*\* is generally \*not feasible\* on commodity hardware except for very tiny inputs. Given reported times (minutes for large texts on even fast GPUs), only extremely short queries and top-tier accelerators could approach sub-0.2 s – and even then only with maximal optimizations (e.g. FP16, large batches).



\*\*Batching:\*\* Ollama supports batching multiple texts in one request. For example, the API can accept an array of inputs in one call. Using batches amortizes overhead and boosts throughput: a single large request (e.g. 10–100 documents) will be faster overall than 10 separate calls, because each request incurs model-loading and HTTP overhead. As one commenter notes, \*\*inference is memory-bandwidth-bound\*\*, so you should “use maximally sized batches… otherwise you are mostly measuring the overhead of moving data around in Python”. In practice, experiment with batch sizes: large batches (limited by GPU memory) maximize throughput, while single-item batches minimize individual latency but suffer inefficiency.



\*\*Example:\*\* Suppose you have 10 documents. Sending them in one curl call like below will process them together:



```bash

curl -X POST http://localhost:11434/api/embed -d '{

&nbsp; "model": "mxbai-embed-large",

&nbsp; "input": \["Text of doc1", "Text of doc2", ..., "Text of doc10"]

}'

```



This returns a JSON with an array of 10 embedding vectors, leveraging batch processing. In tests, using a larger batch on the same GPU yielded up to \\~2× speedups over smaller batches. In summary, \*\*CPU-only embedding is extremely slow\*\* (often seconds to minutes), while \*\*GPU embedding is markedly faster\*\*, especially with large batches and FP16 – but even on GPUs sub-0.2 s per text is unlikely except for very short inputs.



\## Optimization Techniques



\* \*\*Request Queuing \& Rate Limiting:\*\* Because each embedding call may load and run a large model, it’s wise to limit concurrency. Implement a simple token-bucket or queue so you don’t overwhelm the server with parallel requests. For example, allow at most one request per model per GPU at a time, and queue extras. This avoids thrashing the model memory (as some users saw Ollama repeatedly loading/unloading large models). Also consider retry/backoff logic: catch network or HTTP errors and retry after delay. If using Ollama’s OpenAI-compatible API (`/v1/embeddings`), any non-empty API key is ignored, but network timeouts and connection failures still occur; wrap calls in try/catch (e.g. Python `requests` with `timeout=...`) and handle errors gracefully.



\* \*\*Batch Scheduling Trade-offs:\*\* In a semantic pipeline (e.g. many users querying a vector DB), you may batch multiple queries at once or process individually. Batching more texts per call increases throughput (see above) but adds wait time for accumulating the batch. If low-latency per-query is critical, smaller batches (even batch size 1) might be used at cost of GPU underutilization. In contrast, high-throughput offline indexing favors huge batches. A mixed strategy is to group requests arriving close in time into batches (e.g. aggregate 5–10 queries every 100 ms). Always tune batch size for your hardware: one Reddit user found that after maximizing batch size on an H100, performance doubled.



\* \*\*Caching \& Reuse:\*\* To avoid redundant embedding work, cache results for repeated inputs. A simple approach is hashing the input text and storing the embedding vector in memory or a fast key-value store (e.g. Redis). On repeat queries, return the cached vector instead of calling Ollama. For \*slightly modified\* inputs (typos or small edits), simple cache lookup won’t hit; in some cases you might pre-normalize text (e.g. remove stopwords, or use fuzzy matching) to reuse close embeddings, but generally embeddings are not easily “incremental”. In practice, most systems simply avoid duplicate work: if your application e.g. re-asks the same question or repeatedly ingests identical documents, first check a cache of hashed embeddings. This can massively cut load. (No direct citations here, but caching is a standard optimization in embedding pipelines.)



\* \*\*Prefetch/Keep Models Warm:\*\* Ollama loads models from disk each time unless kept in memory. If your system sees intermittent requests, consider “warming” Ollama by making a dummy call (or using `ollama run`) so the model stays loaded in RAM/VRAM. This avoids the initial load penalty (which one GitHub user noted costs extra seconds per call). In Docker or server setups, you might keep a permanent Ollama process running (`ollama serve`) rather than spin up per request.



\## Ollama Embedding API \& Integration Architecture



\*\*HTTP API Format:\*\* Ollama exposes a REST endpoint for embeddings. In Ollama’s examples, the endpoint is `/api/embed` with JSON body containing `"model"` and `"input"`. For example, using `curl` or Python one can do:



```bash

curl http://localhost:11434/api/embed -d '{

&nbsp; "model": "mxbai-embed-large",

&nbsp; "input": "Ollama makes running LLMs locally easy."

}'

```



This returns JSON with an `"embeddings"` array (and typically `"usage"` stats). In the Python library, the call looks like `response = ollama.embed(model="mxbai-embed-large", input="text")`, and the vector is `response\["embeddings"]`.  (In newer Ollama versions, an alternative endpoint `/api/embeddings` or OpenAI-compatible `/v1/embeddings` may be supported; e.g. one guide shows `/api/embeddings` with a `{"prompt": "..."} ` JSON and notes that `/v1/embeddings` mirrors OpenAI’s API. In any case, the format is simple JSON with the model name and text, returning a JSON with the numeric vector.)



\*\*Handling Timeouts/Errors:\*\* Ollama’s clients typically allow specifying a timeout. For instance, Haystack’s `OllamaDocumentEmbedder` sets a default `timeout=120` seconds. In practice, you should catch exceptions and set reasonable timeouts on HTTP calls to avoid hanging (especially for long inputs or if Ollama is down). If the Ollama service is unavailable (server down, port blocked), the client will get connection errors. Best practice is to retry a few times with exponential backoff, and fall back gracefully if embedding fails. (Since Ollama is local-first, network issues are rare, but Docker or firewall misconfiguration can happen.)  No authentication is required (the `/v1/embeddings` endpoint ignores the API key), but ensure your network allows reaching `localhost:11434` or the host/port you configure (you can set `OLLAMA\_HOST` to change the listen address if needed).



\*\*Parallel Ollama Instances:\*\* To scale beyond one thread or GPU, you can run multiple Ollama servers. For multi-GPU machines, Ollama uses environment variables to bind processes to GPUs. For example, setting `CUDA\_VISIBLE\_DEVICES=0` before `ollama serve` forces usage of GPU 0, `=1` for GPU 1, etc.. On Linux/Windows with AMD GPUs, use `ROCR\_VISIBLE\_DEVICES` similarly. In practice, you could launch separate Ollama processes (on different ports) each pinned to a different GPU, and then load-balance requests across them. Ollama’s logs (`ollama ps`) will indicate which device is used (the `PROCESSOR` column shows `gpu` vs `cpu`).  If only CPU is available, you might also run multiple Ollama instances on different CPU cores, but each will compete for RAM (each loads the model fully).



\*\*Resource Usage Monitoring:\*\* Ollama (via llama.cpp) generally loads a whole model into RAM/VRAM on startup. You can monitor this with system tools: on NVIDIA, `nvidia-smi` shows GPU memory and utilization; `top`/`htop` show CPU and RAM. For example, embedding a 7B model might use \\~6–8 GB VRAM and similar system RAM. Ollama’s `ollama ps` command is useful: if a model is loaded on GPU, it will show `gpu` under `PROCESSOR`, otherwise `cpu`. During embedding calls, GPU utilization (nvidia-smi) should spike if the model is on GPU; conversely, if you see near-0% GPU usage, the model may be falling back to CPU (possibly due to insufficient GPU memory or missing drivers). For automated monitoring, you could periodically poll `ollama list` or `ollama ps`, and capture system metrics.



\*\*Example Integration:\*\* A Python snippet using Ollama’s library demonstrates the request/response format:



```python

import ollama

response = ollama.embed(model="mxbai-embed-large", input="Sample text")

vector = response\["embeddings"]  # This is a 1024-dimensional list of floats

```



You could also call the OpenAI-compatible endpoint with the `openai` library by setting `base\_url="http://localhost:11434/v1"` and any `api\_key` (ignored by Ollama). In either case, handle exceptions and consider limiting parallel threads to avoid resource contention.



\*\*Summary:\*\* Ollama’s embedding API is straightforward JSON over HTTP. In high-load or production settings, combine this with standard engineering practices: use request queues and rate-limiters to avoid overload, batch requests judiciously for throughput, cache repeat results to cut costs, and monitor the Ollama process (with `ollama ps`, `nvidia-smi`, etc.) to ensure the model stays on GPU and within memory budgets.



\## \*\*Sources:\*\* Ollama documentation and user reports provide the details above.





Very slow embeddings : r/ollama

https://www.reddit.com/r/ollama/comments/1blx8lk/very\_slow\_embeddings/



Very slow embeddings : r/ollama

https://www.reddit.com/r/ollama/comments/1blx8lk/very\_slow\_embeddings/



H100 vs. RTX4090 performance question : r/MLQuestions

https://www.reddit.com/r/MLQuestions/comments/1d51fun/h100\_vs\_rtx4090\_performance\_question/



H100 vs. RTX4090 performance question : r/MLQuestions

https://www.reddit.com/r/MLQuestions/comments/1d51fun/h100\_vs\_rtx4090\_performance\_question/



How to Use Ollama (Complete Ollama Cheatsheet)

https://apidog.com/blog/how-to-use-ollama/



H100 vs. RTX4090 performance question : r/MLQuestions

https://www.reddit.com/r/MLQuestions/comments/1d51fun/h100\_vs\_rtx4090\_performance\_question/



API Endpoints | Open WebUI

https://docs.openwebui.com/getting-started/api-endpoints/



H100 vs. RTX4090 performance question : r/MLQuestions

https://www.reddit.com/r/MLQuestions/comments/1d51fun/h100\_vs\_rtx4090\_performance\_question/



H100 vs. RTX4090 performance question : r/MLQuestions

https://www.reddit.com/r/MLQuestions/comments/1d51fun/h100\_vs\_rtx4090\_performance\_question/



H100 vs. RTX4090 performance question : r/MLQuestions

https://www.reddit.com/r/MLQuestions/comments/1d51fun/h100\_vs\_rtx4090\_performance\_question/



embeddings models keep\_alive · Issue #6401 · ollama ... - GitHub

https://github.com/ollama/ollama/issues/6401



Embedding models · Ollama Blog

https://ollama.com/blog/embedding-models



Embedding models · Ollama Blog

https://ollama.com/blog/embedding-models



How to Use Ollama (Complete Ollama Cheatsheet)

https://apidog.com/blog/how-to-use-ollama/



How to Use Ollama (Complete Ollama Cheatsheet)

https://apidog.com/blog/how-to-use-ollama/



Ollama

https://docs.haystack.deepset.ai/reference/integrations-ollama



How to Use Ollama (Complete Ollama Cheatsheet)

https://apidog.com/blog/how-to-use-ollama/



How to Use Ollama (Complete Ollama Cheatsheet)

https://apidog.com/blog/how-to-use-ollama/



H100 vs. RTX4090 performance question : r/MLQuestions

https://www.reddit.com/r/MLQuestions/comments/1d51fun/h100\_vs\_rtx4090\_performance\_question/

