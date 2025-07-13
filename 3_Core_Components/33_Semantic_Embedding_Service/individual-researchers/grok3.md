

---



\# Schema Definition Engine - Detailed Design Report



The Embedding Service is a cornerstone of Globule, providing the semantic understanding necessary for connecting ideas and enhancing search capabilities. This report delves into the design, addressing key research areas to ensure the service is efficient, resilient, and user-friendly, aligning with Globule’s vision of seamless knowledge management.



\## Introduction

Globule aims to revolutionize personal knowledge management by integrating AI-driven semantic understanding into the filesystem, allowing users to capture thoughts effortlessly while AI organizes and connects them. The Embedding Service, part of the Dual Intelligence Services, captures the overall meaning, feeling, and relationships within user inputs, enabling tasks like semantic search and clustering. Given its dependency on previous components, we proceed systematically, ensuring each decision builds on established foundations.



\### Key Points

\- Research suggests using `mxbai-embed-large` for high accuracy, with alternatives like `nomic-embed-text` and `all-minilm` for lower resource needs.

\- It seems likely that embedding generation can meet the <200ms target with proper hardware, especially using GPUs.

\- The evidence leans toward integrating with Ollama for local model inference, with fallbacks like Hugging Face API if needed.

\- Handling different vector dimensions and preprocessing content, such as chunking large texts, is crucial for performance.



\### Embedding Service Design Overview

The Embedding Service is vital for Globule’s semantic understanding, enabling connections like recognizing "dog" and "puppy" as related or linking notes on "SQLite performance" to "database optimization." Here’s a clear, simple breakdown:



\#### Model Selection

Start with `mxbai-embed-large` for its high accuracy, suitable for detailed semantic tasks. For users with limited hardware, consider `nomic-embed-text` for balance or `all-minilm` for speed, each with different resource needs. Check \[Ollama’s model library](https://ollama.com/library) for options, ensuring multilingual support if needed.



\#### Performance

Embedding generation should be fast, ideally under 200ms, achievable with GPUs. Batch processing can speed things up, and using GPUs via Ollama helps, especially for larger models. For users without GPUs, smaller models might still meet needs on CPUs.



\#### Integration

Use Ollama’s local API for embeddings, accessible via REST or Python libraries, ensuring privacy and low latency. If Ollama fails, fall back to Hugging Face API, though it may be slower and cost more. Running models directly is another option but requires more setup.



\#### Storage and Preprocessing

Handle varying vector sizes (e.g., 1024 for `mxbai-embed-large`) by storing as BLOBs in SQLite. For large texts, chunk into 512-token pieces, ensuring no information loss, and preprocess by cleaning text (e.g., removing URLs).



\#### Quality and Efficiency

Ensure quality with benchmarks like MTEB, and cache embeddings for frequent use to save resources. For special content like images, consider future multimodal models, focusing on text for now.



This approach keeps Globule’s "understanding engine" fast, accurate, and adaptable to user needs.





\## Embedding Model Selection and Management

The choice of embedding model is critical for semantic accuracy and resource efficiency. Research suggests Ollama supports several models, including `mxbai-embed-large`, `nomic-embed-text`, and `all-minilm`, each with distinct trade-offs.



\- \*\*Available Models:\*\* From \[Ollama’s blog](https://ollama.com/blog/embedding-models), supported models include:

&nbsp; - `mxbai-embed-large`: 334M parameters, state-of-the-art (SOTA) on MTEB, suitable for high accuracy.

&nbsp; - `nomic-embed-text`: 137M parameters, supports long contexts (8192 tokens), and offers multimodal capabilities in v1.5.

&nbsp; - `all-minilm`: 23M parameters, lightweight, fast, but with slightly lower accuracy (e.g., 84-85% on STS-B vs. 87-88% for larger models).



\- \*\*Trade-offs:\*\* Larger models like `mxbai-embed-large` offer higher accuracy but require more disk space and memory, impacting users with limited hardware. `nomic-embed-text` balances performance and efficiency, while `all-minilm` is ideal for speed on resource-constrained devices.



\- \*\*Content Types:\*\* `mxbai-embed-large` excels in technical and general text, as per \[Mixedbread documentation](https://www.mixedbread.com/docs/embeddings/mxbai-embed-large-v1), while `nomic-embed-text`’s multimodal support is beneficial for diverse inputs. For creative writing, larger models capture nuances better.



\- \*\*Multilingual Support:\*\* Models like `nomic-embed-text` support multiple languages, crucial for users with multilingual notes, as noted in \[Ollama library](https://ollama.com/library).



\- \*\*Resource Requirements:\*\* `mxbai-embed-large` requires significant RAM (e.g., several GBs), while `all-minilm` is lighter, fitting devices with 8GB RAM, as per \[Sentence Transformers documentation](https://sbert.net/docs/package\_reference/sentence\_transformer/SentenceTransformer.html).



\## Performance and Optimization Strategies

Embedding generation must be fast to maintain user flow, with the HLD targeting sub-second response times for searches, implying embedding latency should be under 200ms for real-time processing.



\- \*\*Latency Benchmarks:\*\* Research shows open-source models on CPU can achieve latencies in tens of milliseconds, e.g., all-MiniLM-L6-v2 at 26ms P95 on CPU, improving to 12ms with quantization (\[Optimize Sentence Transformers](https://www.philschmid.de/optimize-sentence-transformers)). With GPU, latencies drop further, likely meeting the target.



\- \*\*Batch Processing:\*\* Processing multiple inputs in batches reduces per-item latency, especially on GPU, as per \[Sentence Transformers efficiency guide](https://sbert.net/docs/sentence\_transformer/usage/efficiency.html), improving throughput for large datasets.



\- \*\*GPU Acceleration:\*\* Ollama supports GPU acceleration, significantly reducing latency, as noted in Reddit discussions where GPU-loaded models outperform CPU (\[Reddit: Very slow embeddings](https://www.reddit.com/r/ollama/comments/1blx8lk/very\_slow\_embeddings/)).



\- \*\*Request Queuing:\*\* Implement queuing to manage load, preventing overload, especially on limited hardware, using settings like OLLAMA\_MAX\_QUEUE (default 512) from \[Ollama 0.2 concurrency](https://medium.com/@simeon.emanuilov/ollama-0-2-revolutionizing-local-model-management-with-concurrency-2318115ce961).



\## Integration Architecture with Ollama

Ollama provides a local API for running models, ensuring privacy and low latency, critical for Globule’s personal use case.



\- \*\*API Usage:\*\* The REST API endpoint is `\[invalid url, do not cite]`, with examples like `curl http://localhost:11434/api/embed -d '{ "model": "mxbai-embed-large", "input": "Llamas are members of the camelid family" }'` (\[Ollama blog](https://ollama.com/blog/embedding-models)). Python and JavaScript libraries simplify integration, e.g., `ollama.embed(model='mxbai-embed-large', input='Llamas are members of the camelid family')`.



\- \*\*Handling Unavailability:\*\* If Ollama fails, fall back to Hugging Face API, though latency can be higher (e.g., 2s for 35 embeddings on GPU, \[Stack Overflow](https://stackoverflow.com/questions/76877573/huggingface-inference-endpoints-extremely-slow-performance)). Alternatively, run models directly with sentence-transformers, requiring more setup.



\- \*\*Multiple Instances:\*\* Ollama supports running multiple instances for parallel processing, especially on multi-GPU setups, as per \[HelixML blog](https://blog.helix.ml/p/running-ollama-server-side), using HTTP proxies like Caddy for load balancing.



\- \*\*Monitoring:\*\* Monitor Ollama’s health via logs and resource usage, ensuring stability, with settings like OLLAMA\_MAX\_LOADED\_MODELS for concurrency management.



\## Fallback Strategies and Resilience

Ensuring resilience is crucial for uninterrupted service, especially for local deployments.



\- \*\*Hugging Face API:\*\* Offers a fallback, but latency (e.g., 2s for embeddings) and costs (based on usage, \[Hugging Face pricing](https://huggingface.co/pricing)) are higher, suitable for occasional use.



\- \*\*Direct Inference:\*\* Running sentence-transformers locally is an option, with latency benchmarks showing fast CPU performance (e.g., 26ms for all-MiniLM), but requires managing model loading and resources.



\- \*\*Consistency:\*\* Ensure consistency by mapping model outputs to standard formats, handling different dimensionalities (e.g., 1024 vs. 384) in storage.



\- \*\*Offline Operation:\*\* Cache model files locally for offline use, ensuring availability, as per \[Ollama setup](https://ollama.com/blog/embedding-models).



\## Vector Dimensionality and Storage

Vector dimensionality impacts storage and search performance, with the HLD mentioning 1024-dimensional vectors.



\- \*\*Dimensionality:\*\* Models vary, e.g., `mxbai-embed-large` outputs 1024 dimensions, `nomic-embed-text` 768, and `all-minilm` 384. Handle variability by storing as BLOBs in SQLite, with metadata for dimensions.



\- \*\*Dimension Reduction:\*\* Techniques like PCA or quantization can reduce storage, but for MVP, raw storage is sufficient, as per \[Mixedbread documentation](https://www.mixedbread.com/docs/embeddings/mxbai-embed-large-v1).



\- \*\*Storage Impact:\*\* Higher dimensions increase space (e.g., 1024 floats per vector), but modern storage handles this, with compression possible for future scalability.



\## Content Preprocessing

Preprocessing ensures inputs are suitable for embedding, especially for large or diverse content.



\- \*\*Chunking:\*\* Split large texts into chunks, e.g., 512 tokens for `mxbai-embed-large`, with overlap for context, as per \[Mixedbread sequence length](https://www.mixedbread.com/docs/embeddings/mxbai-embed-large-v1). Use sentence boundaries for natural splits.



\- \*\*Text Extraction:\*\* For PDFs or images, extract text using OCR or PDF readers, ensuring compatibility with embedding models, as per \[Hugging Face embeddings](https://huggingface.co/blog/getting-started-with-embeddings).



\- \*\*Cleaning:\*\* Normalize text by removing URLs, special characters, and standardizing formats, enhancing embedding quality, as per \[Sentence Transformers preprocessing](https://sbert.net/docs/package\_reference/sentence\_transformer/SentenceTransformer.html).



\## Quality Assurance and Monitoring

Ensuring embedding quality is vital for search and clustering accuracy.



\- \*\*Benchmarks:\*\* Use MTEB for evaluation, with `mxbai-embed-large` achieving SOTA, as per \[Hugging Face MTEB](https://huggingface.co/blog/mteb), ensuring reliability.



\- \*\*Similarity Testing:\*\* Implement tests with known pairs (e.g., "dog" and "puppy") to verify cosine similarity meets thresholds, detecting degraded embeddings.



\- \*\*Embedding Drift:\*\* Monitor for drift with model updates, ensuring consistency by re-embedding affected content, as per \[Pinecone embedding models](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/).



\- \*\*Metrics:\*\* Track latency, error rates, and similarity scores for monitoring, ensuring service health, as per \[Weaviate embedding guide](https://weaviate.io/blog/how-to-choose-an-embedding-model).



\## Caching and Incremental Updates

Efficiency is key for large datasets, reducing redundant computations.



\- \*\*Caching:\*\* Cache embeddings at chunk level for large documents, using in-memory stores for frequent access, improving performance, as per \[Zep blog](https://blog.getzep.com/text-embedding-latency-a-semi-scientific-look/).



\- \*\*Incremental Updates:\*\* For changes, re-embed only affected chunks, saving resources, especially for personal notes, as per \[Nixiesearch latency benchmarks](https://nixiesearch.substack.com/p/benchmarking-api-latency-of-embedding).



\- \*\*Interpolation:\*\* For minor edits, consider interpolation, though for MVP, full re-embedding is simpler, as per \[Milvus embedding insights](https://milvus.io/blog/we-benchmarked-20-embedding-apis-with-milvus-7-insights-that-will-surprise-you.md).



\## Special Content Types

Handling diverse content enhances Globule’s versatility, though focused on text for MVP.



\- \*\*Images:\*\* Use multimodal models like CLIP for image embeddings, aligning with text for search, as per \[Nomic Embed Vision](https://arxiv.org/html/2406.18587v1), deferred for future iterations.



\- \*\*Audio/Transcripts:\*\* Extract transcripts for embedding, using speech-to-text tools, as per \[Hugging Face tasks](https://huggingface.co/docs/inference-providers/index).



\- \*\*Code:\*\* Consider specialized models like CodeBERT for code snippets, enhancing semantic understanding, as per \[Sentence Transformers applications](https://medium.com/@rahultiwari065/unlocking-the-power-of-sentence-embeddings-with-all-minilm-l6-v2-7d6589a5f0aa).



\## Resource Management

Resource constraints impact model choice and performance, especially for local deployment.



\- \*\*RAM Requirements:\*\* `mxbai-embed-large` requires several GBs, while `all-minilm` fits devices with 8GB RAM, as per \[Ollama system requirements](https://blog.futuresmart.ai/running-multiple-open-source-llms-locally-with-ollama).



\- \*\*CPU/GPU Utilization:\*\* GPU acceleration reduces latency, with settings like OLLAMA\_NUM\_PARALLEL optimizing parallel requests, as per \[Ollama concurrency](https://medium.com/@simeon.emanuilov/ollama-0-2-revolutionizing-local-model-management-with-concurrency-2318115ce961).



\- \*\*Request Queuing:\*\* Manage requests to avoid overload, using queuing mechanisms, ensuring smooth operation on limited hardware, as per \[HelixML blog](https://blog.helix.ml/p/running-ollama-server-side).



\## Key Trade-offs

The design balances accuracy vs. speed (larger models are slower but more accurate), local vs. cloud (Ollama is private but requires hardware, Hugging Face is scalable but costly), and flexibility vs. simplicity (handling special content adds complexity).



\## Summary of Design Decisions

The following table summarizes key decisions:



| Aspect                     | Decision                                      | Rationale                                                                 |

|----------------------------|-----------------------------------------------|---------------------------------------------------------------------------|

| Model Selection            | Start with `mxbai-embed-large`, alternatives for resources | High accuracy, options for hardware constraints                          |

| Performance Target         | <200ms with GPU, batch processing            | Meets user flow needs, leverages hardware capabilities                    |

| Integration                | Ollama API, fallback to Hugging Face         | Local privacy, cloud resilience, latency trade-offs                      |

| Vector Dimensionality      | Handle varying sizes, store as BLOBs         | Flexibility for different models, efficient storage                      |

| Preprocessing              | Chunk at 512 tokens, clean text              | Ensures compatibility, enhances quality                                  |

| Quality Assurance          | MTEB benchmarks, similarity tests            | Ensures reliability, detects degradation                                 |

| Caching                    | Cache at chunk level, incremental updates    | Efficiency for large datasets, resource savings                          |

| Special Content            | Focus on text, plan for multimodal later     | MVP simplicity, future extensibility                                     |

| Resource Management        | Choose based on hardware, queue requests     | Adaptability, prevents overload                                          |



This design ensures a robust Embedding Service, balancing usability, performance, and scalability, setting the stage for Globule’s evolution.

