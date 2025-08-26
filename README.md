# Globule

> Capture your thoughts, find connections, and draft your next big idea.

![Project Status: Active Development](https://img.shields.io/badge/status-active_development-green)

Globule is a tool for thought management. It uses AI to help you find semantic connections between your notes, making it easy to synthesize new ideas and draft coherent documents.

## The Core Experience

Globule's magic is in its simplicity. Capture any thought, and let the AI handle the rest.

#### 1. Capture Instantly

No need to think about folders or filenames. Just capture the thought from anywhere - command line, WhatsApp, Telegram, or email.

```bash
$ globule add "The concept of 'progressive overload' in fitness could apply to creative stamina."

$ globule add "A core theme for my next post: discipline isn't about restriction, it's about freedom."
```

Or send a message to your personal WhatsApp bot and watch your thoughts flow seamlessly into your knowledge base.

#### 2. Synthesize with Ease

When you're ready to write, tell Globule what you're thinking about.

```bash
$ globule draft "my next blog post"
```

Globule's intelligent engine understands these thoughts are related and presents them in a clean, two-pane interface, ready for you to weave them together into a coherent first draft.

## Getting Started

```bash
# Clone and install from source
git clone https://github.com/asavschaeffer/globule
cd globule
pip install -e .

# Start capturing your thoughts
globule add "Your first thought here"

# Draft content from your captured thoughts
globule draft "your topic"
```

## Key Commands

- `globule add "<text>"`: Captures a new thought.
- `globule draft "<topic>"`: Opens an interactive TUI to synthesize a draft on a topic.
- `globule search "<query>"`: Performs a semantic search for related thoughts.
- `globule reconcile`: Reconciles the file system with the database.

### Messaging Integration

Capture thoughts from anywhere with messaging platform integration:

- `globule inputs setup-whatsapp`: Set up WhatsApp Business API integration
- `globule inputs setup-telegram`: Set up Telegram bot integration  
- `globule inputs webhook-server`: Run local webhook server for message processing
- `globule inputs test-message`: Test messaging integration with sample data

Send messages with text, images, or documents to your connected platforms and Globule will automatically process them into searchable thoughts in your knowledge base.

## Architecture

The Globule codebase has been refactored for simplicity, maintainability, and performance. It now follows a clean, three-layer architecture:

1.  **Command-Line Interface (CLI):** The user-facing entry point, built with `click`.
2.  **GlobuleAPI:** A clean, UI-agnostic API that exposes the core features of the application.
3.  **Core Logic:** The underlying orchestration engine, services, and storage managers that handle data processing and persistence.

This separation of concerns makes the system easier to understand, test, and extend.

## The Vision: Where We're Going

The initial version of Globule is focused on the core experience of capture and synthesis. But this is just the foundation for a much larger vision.

-   **Empowering Workflows:** Soon, you'll be able to teach Globule about *your* specific types of information (like `Recipes` or `Code Snippets`), enabling custom formatting and perfect integration with tools like Obsidian.
-   **Personalized Organization:** You will be able to tune Globule's brain, defining your own templates for how files and folders are named and organized, making the semantic filesystem truly your own.

Our ultimate goal is to build a new foundational layer for personal computing—one that understands context, not just commands.

## Contributing

This project is currently in a design-heavy phase. If you are interested in the architecture, design philosophy, and the future of semantic computing, we welcome you to explore our **[Project Wiki](https://github.com/asavschaeffer/globule/wiki)** where the system is being designed in the open.
