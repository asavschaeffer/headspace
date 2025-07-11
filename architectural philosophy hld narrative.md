## The Foundation: Core Processing Architecture

At the very base of our cathedral, we need the **Orchestration Engine**. Think of this as the conductor of an orchestra - it doesn't play any instruments itself, but it understands how to bring together different types of intelligence to create harmony. This module takes input from both the embedding system (which understands the feeling and relationships of ideas) and the parsing system (which extracts specific facts and entities), then makes intelligent decisions about how to combine their insights.

The Orchestration Engine needs to be built first because everything else depends on its decisions. When a user adds a thought about "meeting with Sarah about the new project," the orchestrator decides whether the important part is the semantic connection to other project-related thoughts (from embeddings) or the specific entity "Sarah" and the temporal marker "new" (from parsing). Most likely, it uses both, weighing them based on context.

## The Input Gateway: Conversational Understanding

Next, we need the **Adaptive Input Module** - this is like the cathedral's entrance, welcoming all who approach while gently guiding them to the right destination. This module does more than just accept text; it engages in a brief conversation with the user when needed. "I notice you're entering what looks like a recipe - should I use the cooking schema?" This conversational element transforms potential frustration into collaborative understanding.

The Input Module must be schema-aware from the beginning. Even in the MVP, it should ship with a few basic schemas (free text, link collection, task entry) while providing a simple way for users to define their own. A schema might be as simple as a YAML file that says "if the input contains a URL, extract the title and create a companion note."

## The Intelligence Layer: Dual Processing

The **Semantic Embedding Service** and **Structural Parsing Service** work as partners, not competitors. Think of them as two different types of scholars examining the same manuscript - one reads for meaning and emotion, the other catalogs facts and references.

The Embedding Service uses a model like mxbai-embed-large to transform text into high-dimensional vectors that capture meaning. When you write about "the melancholy of Sunday evenings," it understands this is semantically related to other thoughts about time, emotion, and weekly rhythms, even if those exact words never appear together elsewhere.

The Parsing Service uses a small, fast language model to extract structured information. It identifies that "Sunday evening" is a temporal marker, "melancholy" is an emotional state, and the overall statement is a personal reflection. This structured data becomes metadata that makes your thoughts queryable in ways pure text search could never achieve.

## The Storage Fabric: Semantic File System

The **Intelligent Storage Manager** is perhaps the most revolutionary component. Unlike traditional systems that make you decide where to put things, this module uses the combined intelligence from the orchestrator to automatically organize your thoughts into a meaningful file structure.

Here's where the magic happens: the filesystem itself becomes a form of pre-indexed search. A thought about "the progressive overload principle applied to creative work" might be stored at `~/globule/creativity/theories/progressive-overload-creative-work.md`. The path tells a story - it's about creativity, it's theoretical rather than practical, and it's building on an existing concept from another domain. Even without opening the file, you understand its context.

The Storage Manager must also handle metadata elegantly. Rather than cluttering filenames with dates (which the OS already tracks), it uses extended attributes or companion files to store rich metadata like embeddings, entities, and cross-references. This keeps the visible filesystem clean while maintaining a rich underground network of connections.

## The Synthesis Studio: Interactive Intelligence

The **Interactive Synthesis Engine** powers what happens when you type `globule draft`. This is where your cathedral's vaulted ceiling creates a soaring space for creativity. The engine presents two panes - on the left, your recent thoughts organized into semantic clusters (the Palette), and on the right, a canvas for weaving them into something new.

But here's what makes it special: the Palette isn't static. As you work, selecting and exploring different thoughts, the system quietly expands its search, bringing in related ideas from your entire history. It's like having a research assistant who knows everything you've ever thought and whispers relevant connections as you write.

The Synthesis Engine must support two distinct modes from the start. Build Mode (pressing Enter) simply adds selected thoughts to your draft. Explore Mode (pressing Tab) transforms the selected thought into a search query, revealing unexpected connections across your knowledge base. This modal design keeps the interface simple while enabling profound discovery.

## The Configuration Cascade: User Empowerment

Underlying all of these modules is the **Configuration System** - think of it as the architectural blueprints that can be modified even after the cathedral is built. This system operates at three levels: system defaults (rarely changed), user preferences (personal defaults), and context-specific overrides (for different projects or modes).

The beauty of this cascade is that new users can start with zero configuration and have a wonderful experience, while power users can customize every aspect of the system's behavior. Want the system to be chatty and explain everything while you're learning, but silent during focused writing sessions? The Configuration System makes this natural.

## The Schema Engine: Encoding Workflows

Finally, the **Schema Definition Engine** allows users to encode their own workflows into the system. A schema isn't just a template - it's a complete description of how certain types of information should flow through Globule.

For the MVP, schemas should be defined in simple YAML files that any user can understand and modify. A basic schema might say "when I add a link, fetch its title and create a reading note." A complex schema might define an entire workflow for processing meeting notes, extracting action items, and generating follow-up reminders.

## How These Modules Dance Together

The beauty of this modular design is that each component has a clear responsibility while contributing to a greater whole. When you add a thought, it flows through this pipeline:

The Input Module recognizes its type and applies the appropriate schema. The Orchestration Engine coordinates the Embedding and Parsing Services to understand the thought from multiple angles. The Storage Manager uses this combined intelligence to place the thought in a meaningful location. Later, when you're ready to create, the Synthesis Engine helps you rediscover and recombine these thoughts in ways that surprise and delight you.

Each module can be built incrementally, tested independently, and enhanced without breaking the others. This is how we build our cathedral - not all at once, but brick by brick, each piece making the next one possible.

The shopping list, then, isn't just a list of parts to build. It's a map of how human and machine intelligence can work together to create something neither could achieve alone. Start with the Orchestration Engine and basic Input Module, add the dual intelligence services, implement the revolutionary Storage Manager, and crown it all with the Interactive Synthesis Engine. Each piece is simple enough to build in a focused sprint, yet together they create a system that fundamentally changes how we think with computers.
