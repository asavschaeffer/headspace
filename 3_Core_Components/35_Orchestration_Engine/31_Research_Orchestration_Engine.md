# Research: Orchestration Engine

*Version: 1.0*
*Date: 2025-07-13*
*Status: Open Questions*

This document outlines the core research questions that must be addressed to create the Low-Level Design (LLD) for the Orchestration Engine. The engine's primary role is to act as the "conductor" for the dual intelligence services, ensuring that the Semantic Embedding Service and the Structural Parsing Service work in harmony.

## 1. Core Architectural Philosophy

1.  **What is the most suitable architectural pattern for the Orchestration Engine?**
    *   Is it a simple **Pipeline**, where input flows sequentially (e.g., Parse -> Enrich -> Embed)?
    *   Is it a **State Machine**, where the processing strategy changes based on the state of the input globule?
    *   Is it a **Strategy Pattern**, where different coordination strategies (e.g., "creative-writing-mode", "technical-analysis-mode") can be dynamically selected based on context or configuration?

2.  **Should the engine be stateless or stateful?**
    *   Does each `process_globule` call operate in isolation (stateless)?
    *   Or does the engine need to maintain state between calls to understand the context of a user's session (stateful)? For example, should it remember the last few globules to better inform the next parsing prompt?

3.  **How should the engine handle failures in its dependent services?**
    *   If the `Semantic Embedding Service` fails, should the `Structural Parsing Service` still proceed?
    *   What is the retry logic? Should it implement a circuit breaker pattern for services that are repeatedly failing?
    *   What constitutes a "successful" partial processing versus a total failure?

## 2. Dual-Track Coordination and Collaboration

The core innovation of Globule is the "collaborative intelligence" between the embedding and parsing services. The exact mechanism for this collaboration needs to be defined.

1.  **What is the precise flow of data between the two services?**
    *   **Option A (Parallel):** Both services run simultaneously. A third step then reconciles their outputs. This seems to be implied by the HLD.
    *   **Option B (Sequential - Parse First):** The `Structural Parsing Service` runs first. Its output (e.g., detected entities, schema type) is then used to enrich the input for the `Semantic Embedding Service`.
    *   **Option C (Sequential - Embed First):** The `Semantic Embedding Service` runs first. Its output (e.g., finding semantically similar globules) is used to build a richer, context-aware prompt for the `Structural Parsing Service`.
    *   **Option D (Iterative):** A multi-pass approach. A quick initial embedding finds neighbors, which informs parsing, which in turn informs a final, more context-aware embedding.

2.  **How is the "content-aware weight determination" implemented?**
    *   The HLD mentions a `ContentProfile` with `structure_score` and `creativity_score`. How are these scores calculated? Is it based on heuristics (e.g., presence of code, bullet points), or does it require another lightweight model?
    *   How do these weights practically influence the outcome? Do they affect the final file path generation, the data stored in metadata, or the priority of information displayed in the synthesis engine?

## 3. Disagreement and Nuance Handling

A key responsibility of the engine is to resolve or preserve disagreements between the two AI services (e.g., sarcasm).

1.  **How is a "disagreement" programmatically detected?**
    *   What is the quantitative measure? Is it a mismatch between the parser's sentiment analysis (e.g., `positive`) and the embedding's proximity to a known emotional cluster (e.g., `frustration`)?
    *   What is the threshold for flagging a disagreement?

2.  **What are the different categories of nuance to handle?**
    *   Sarcasm (conflicting sentiment/semantic meaning)
    *   Metaphor (literal vs. figurative meaning)
    *   Technical Jargon (a term's general meaning vs. its specific meaning in a domain)

3.  **What does it mean to "preserve both" interpretations?**
    *   Is a `nuance_detected: true` flag added to the globule's metadata?
    *   Are both interpretations (e.g., `literal_sentiment` and `semantic_sentiment`) stored?
    *   How is this preserved nuance used by other components, like the `Interactive Synthesis Engine`? Does it get a special UI treatment?

## 4. Integration and API Contracts

The Orchestration Engine sits at the center of the processing pipeline and must have clearly defined interfaces.

1.  **Input Contract:** What is the exact structure of the `EnrichedInput` object it receives from the `Adaptive Input Module`? What fields are guaranteed to be present?
2.  **Output Contract:** What is the exact structure of the `ProcessedGlobule` object it sends to the `Intelligent Storage Manager`?
3.  **Dependency Contracts:**
    *   What are the precise API calls (methods, arguments, return types) for the `Semantic Embedding Service` and `Structural Parsing Service`?
    *   Does the Orchestration Engine have direct access to the `Intelligent Storage Manager` to fetch semantic neighbors for context, or does it go through the `Query Engine`?

## 5. Performance and Scalability

The engine must be performant to meet the sub-500ms end-to-end processing target.

1.  **What is the latency budget for the engine's own logic, separate from the AI service calls?**
2.  How can the engine optimize its interactions with dependent services? Should it use batching if it receives multiple globules to process at once?
3.  What parts of the orchestration logic can be cached? For example, can a `ContentProfile` be cached for a given piece of text?

## 6. Configuration and Extensibility

The engine's behavior should be customizable by the user, in line with Globule's philosophy.

1.  **Which aspects of the orchestration logic should be configurable via the `Configuration System`?**
    *   The weights for parsing vs. embedding?
    *   The thresholds for disagreement detection?
    *   The default resolution strategy for disagreements?
2.  **How can users define new coordination strategies?**
    *   Could a `Schema Definition` include a section that specifies how the Orchestration Engine should process it (e.g., `orchestration_strategy: prioritize_parsing`)?
    *   This would allow users to create highly specialized workflows, turning the engine into a truly programmable component.