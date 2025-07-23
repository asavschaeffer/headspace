# Globule Project Wiki: Home

This wiki serves as the central knowledge base for the Globule project. It's designed to provide a comprehensive overview, from the highest-level vision down to the specific implementation details of each component.

## 1. What is Globule?

At its core, Globule is a system designed to reduce the friction between thought and digital organization. We currently spend significant mental overhead naming files, organizing folders, and trying to recall where we stored information. Globule proposes a different approach: a **local-first, semantic layer for your personal information.**

The name itself hints at the core function: to **glob** (globally search) for **blob**s (in this case, the vector embeddings that represent meaning). It's a system for finding the meaning in your data, not just the data itself.

The goal is to create a tool that understands the context and connections within your thoughts, allowing you to focus on creating while the system handles the organization.

## 2. Core Principles

The project is guided by several foundational principles that inform every architectural decision:

*   **Capture First, Organize Never:** Optimize for frictionless input; AI handles the organizational complexity.
*   **Semantic Understanding Over Hierarchical Storage:** Information is connected by intrinsic meaning, not rigid folder structures.
*   **AI as a Collaborative Partner:** The AI suggests, assists, and automates, but the user remains in control.
*   **Progressive Enhancement Architecture:** Build a simple, valuable core that can evolve without disruptive rewrites.
*   **Privacy-First, Hybrid-by-Choice:** Local processing by default; cloud features are an explicit opt-in.
*   **Modular and Pluggable Pipeline:** Components are designed as abstract interfaces for future extension.

## 3. How It Works: The Core Pipeline

Globule processes information in a simple, multi-stage pipeline:

1.  **Input**: A user captures a thought or piece of information (a "globule") through an interface like a CLI.
2.  **Processing**: The system processes the globule using a dual-track AI approach to understand it.
3.  **Storage**: The processed globule, along with its semantic metadata, is stored intelligently on the local filesystem.
4.  **Synthesis**: The user can then query, retrieve, and weave their captured thoughts into structured documents.

## 4. System Components

Globule's architecture is modular, with each component having a distinct responsibility.

*   **Orchestration Engine**: The central "conductor" that coordinates the various AI services. It decides how to best combine semantic and structural analysis for any given input.
*   **Adaptive Input Module**: The user's entry point to the system. It's a conversational gateway that validates input and applies the correct schema, clarifying with the user if the intent is ambiguous.
*   **Semantic Embedding Service**: This service uses models (e.g., `mxbai-embed-large`) to convert text into vector embeddings, capturing the "feeling" and semantic relationships of the content.
*   **Structural Parsing Service**: This service uses LLMs (e.g., `llama3.2:3b`) to extract specific, structured data from the text, such as entities, categories, and facts.
*   **Intelligent Storage Manager**: Manages all data persistence. Its key innovation is creating a semantic filesystem structure, generating human-readable paths and filenames based on the content's meaning. It uses SQLite for metadata and vector storage.
*   **Interactive Synthesis Engine**: Powers the user-facing drafting tool (`globule draft`). It provides a two-pane TUI (Textual User Interface) where users can browse, select, and combine their captured thoughts into a coherent document.
*   **Configuration System**: A three-tier cascade (System -> User -> Context) that allows for deep but progressive customization. It uses YAML files and is designed to be both powerful for advanced users and zero-config for beginners.
*   **Schema Definition Engine**: Allows users to define custom workflows and data structures using simple YAML files. This engine transpiles the user-friendly YAML into high-performance Pydantic models for runtime validation, effectively allowing users to encode their own logic into the system.

## 5. How to Use This Wiki

This wiki is organized to reflect the project's architecture.

*   **1_Foundations**: High-level documents covering the project's vision, strategy, and philosophy.
*   **2_System_Architecture**: Architectural diagrams and design documents that describe how the system fits together.
*   **3_Core_Components**: Detailed Low-Level Design (LLD) and research documents for each specific component.

To get a feel for the project, we recommend reading the following documents in order:

1.  `1_Foundations/11_Vision-and-Strategy.md`: Understand the "why" behind Globule.
2.  `2_System_Architecture/20_High-Level-Design.md`: Get a bird's-eye view of the system architecture.
3.  `2_System_Architecture/23_Component_Interaction_Flows.md`: See how the components work together in practice.
4.  Browse the `3_Core_Components` directory to dive into the specifics of any component that interests you.

You can navigate through the documents using the sidebar. The numbering is intended to provide a logical reading order, from abstract concepts to concrete implementations.

---

This wiki is a living document. As the project evolves, so will this knowledge base. Welcome to Globule.