# Gemini's Understanding of the Globule Project

This document reflects my current understanding of the Globule project based on the information available in this wiki. My purpose is to provide a comprehensive summary that can serve as a quick onboarding guide for new contributors and as a reference for the development team.

## 1. Core Vision: A Semantic Operating System

Globule is not just a note-taking application; it is an ambitious project to create a **semantic operating system layer**. The fundamental goal is to change how humans interact with computers by moving away from rigid, hierarchical file systems towards a more fluid, context-aware paradigm.

The system is designed to understand the **meaning and connections** between user inputs, rather than just storing raw data. It aims to create a personalized knowledge graph that mirrors the user's thought processes, making information retrieval intuitive and discovery-oriented.

## 2. Key Concepts

*   **Globule**: The atomic unit of information. A "globule" can be any piece of input—a thought, a note, a link, a voice memo, etc.
*   **Semantic Understanding**: The system's ability to grasp the context, meaning, and relationships of globules using AI technologies like embeddings and Large Language Models (LLMs).
*   **Dual-Track Processing**: A core architectural pattern where each globule is processed simultaneously by two AI systems:
    *   An **Embedding Engine** to capture semantic meaning and relationships.
    *   A **Parsing Engine** to extract structured data, entities, and facts.
*   **Progressive Discovery**: An intelligent retrieval mechanism that surfaces relevant information as the user explores their knowledge base, creating a "ripple effect" of discovery.
*   **Schema Definition Engine**: A powerful feature that allows users to define their own custom workflows and data structures using simple YAML files, transforming Globule into a personalized knowledge processing system.

## 3. System Architecture

Globule's architecture is designed as a multi-stage pipeline, ensuring modularity and progressive enhancement. The main layers are:

1.  **Input Layer**: Handles various forms of user input, starting with a CLI and TUI. It includes an **Adaptive Input Module** that can engage in a dialogue with the user to clarify intent.
2.  **Processing Pipeline**: The heart of the system, where the **Orchestration Engine** coordinates the dual-track processing of globules.
3.  **Storage Layer**: A local-first storage solution, using SQLite with vector support in the MVP. The **Intelligent Storage Manager** automatically organizes information into a semantic file structure.
4.  **Synthesis & Retrieval Layer**: The user-facing part of the system, where the **Query Engine** allows for natural language queries and the **Interactive Synthesis Engine** helps users weave their thoughts into polished documents.

## 4. Core Components

*   **Orchestration Engine**: The "conductor" of the AI services, ensuring that embedding and parsing work in harmony.
*   **Adaptive Input Module**: A conversational gateway that validates input and applies the correct schema.
*   **Dual Intelligence Services**: The Semantic Embedding Service and the Structural Parsing Service.
*   **Intelligent Storage Manager**: Creates a semantic filesystem, automatically organizing information.
*   **Interactive Synthesis Engine**: Powers the interactive drafting experience.
*   **Configuration System**: A three-tier cascade system (System -> User -> Context) that allows for deep customization.
*   **Schema Definition Engine**: Enables users to define custom workflows and data structures.

## 5. Development Roadmap

The project follows a staged development plan, with each stage building on the previous one:

*   **Stage 1: The Ollie (MVP)**: Focuses on the core capture, processing, storage, and retrieval loop.
*   **Stage 2: The Kickflip**: Enhances the platform with specialized processors for different input types (URLs, images, etc.) and introduces graph relationships.
*   **Stage 3: The Tre Flip**: Moves towards ambient intelligence with passive monitoring and event correlation.
*   **Stage 4: The 360 Flip**: The full realization of the semantic OS vision, with deep OS-level integration.

## 6. Design Philosophy

*   **Capture First, Organize Never**: The user should focus on capturing thoughts; the AI handles organization.
*   **Semantic > Hierarchical**: Meaning and context are more important than rigid folder structures.
*   **AI as a Collaborative Partner**: The AI assists, but the user is always in control.
*   **Privacy-First, Hybrid-by-Choice**: Local-first by default, with optional, secure cloud features.
*   **Modular and Pluggable**: The system is designed for extensibility.

## 7. Wiki Organization

This wiki is structured to mirror the component-oriented architecture of the Globule system. The documents are organized into categories like "Foundations," "System Architecture," and "Core Components," with a numerical prefix to ensure a logical reading order. This structure is intended to make the project's documentation as clear and navigable as the system it describes.