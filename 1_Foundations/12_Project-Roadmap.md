# Globule Development Rocket Trajectory 🚀

*Building a model rocket that can become a real rocket*

## Overview

Each stage builds on the previous one without requiring architectural rewrites. We're planning for the space station while building the model rocket.

---

## Stage 1: The Ollie (MVP - 6 weeks)
*Basic move that everything else builds on*

### Core Architecture
```
Input → Parallel Processing → Smart Storage → Semantic Retrieval
         ├─ Embedding (semantic understanding)
         └─ LLM Parser (structure extraction)
```

### What We Build
- **Input**: CLI/simple TUI for text input
- **Processing**: Dual-track (embeddings + parsing)
- **Storage**: SQLite with JSON metadata + vector columns
- **Retrieval**: Semantic search + temporal queries
- **Output**: Basic report generation from templates

### Key Design Decisions (Future-Proofing)
1. **Modular parser system** - Easy to add new domain schemas
2. **SQLite with JSON** - Can migrate to graph DB later
3. **Abstract storage layer** - Swap backends without rewriting
4. **Plugin-ready architecture** - Even if not exposed yet

### Success Criteria
- Can capture thoughts without friction
- Finds relevant content that keyword search would miss
- Generates useful daily summary
- All pieces are modular and testable

---

## Stage 2: The Kickflip (Enhanced Platform - 3 months)
*Adding style and complexity*

### Architecture Evolution
```
Input → Type Detection → Specialized Processing → Rich Storage → Multi-Modal Retrieval
         ├─ URLs: Crawl & Summarize
         ├─ Images: Vision + Alt Text
         ├─ Code: Diff Analysis
         └─ Voice: Transcription
```

### New Capabilities
- **Smart Input Router**: Detects input type automatically
- **Specialized Processors**:
  - Web crawler for links
  - Computer vision for images
  - Git integration for code
  - Voice transcription
- **Enhanced Storage**: 
  - Graph relationships between globules
  - Richer metadata schemas
  - Version tracking
- **Advanced Outputs**:
  - Blog post generation
  - Code diary formatting
  - Business dashboards
  - Custom report templates

### Technical Additions
- Background workers for async processing
- WebSocket API for real-time updates
- Plugin system goes live
- Cloud sync option (encrypted)

### Why This Isn't Scope Creep
Each processor is a **plugin** to the core system. The ollie architecture doesn't change - we just add new input adapters and output formatters.

---

## Stage 3: The Tre Flip (Ambient Intelligence - 6 months)
*Multiple inputs working in harmony*

### Architecture Evolution
```
Passive Monitoring → Event Stream → Semantic Layer → Proactive Insights
```

### New Capabilities
- **Passive Input Sources**:
  - File system monitoring
  - Browser activity
  - Clipboard monitoring
  - ActivityWatch integration
  - Calendar integration
  - Email monitoring (with permission)
  
- **Event Correlation**:
  - "You edited this file while reading these docs"
  - "This meeting relates to these code changes"
  - Pattern detection across sources

- **Proactive System**:
  - Notifications for patterns
  - Auto-categorization
  - Suggested connections
  - Anomaly detection

### Technical Additions
- Event streaming architecture (Kafka-lite)
- ML models for pattern detection
- Privacy-preserving analytics
- Federated learning prep

---

## Stage 4: The 360 Flip Down 10 Stairs (Semantic OS - 1+ year)
*The full vision realized*

### What This Becomes
- OS-level integration
- Universal semantic search across all computer activity
- Time travel through digital life
- Collaborative intelligence network
- Natural language computer control

### Why We Can Build This
Because every previous stage created the foundations:
- Stage 1: Semantic understanding
- Stage 2: Multi-modal processing  
- Stage 3: Ambient capture
- Stage 4: Just connecting it all

---

## Critical Path Dependencies

### What Must Be Perfect in Stage 1
1. **Embedding/Parser Duality** - This is core to everything
2. **Storage Abstraction** - Must handle future graph needs
3. **Plugin Architecture** - Even if hidden, must exist
4. **Performance Baseline** - Sub-100ms for operations

### What Can Wait
- Beautiful UI (CLI is fine)
- Multi-user support
- Advanced visualizations
- Cloud features

### What We Must Avoid
- Tight coupling between components
- Storage decisions that lock us in
- Over-engineering the MVP
- Feature creep in Stage 1

---

## Development Principles

1. **Each stage must provide standalone value**
   - Ollie: Replaces note-taking
   - Kickflip: Replaces multiple tools
   - Tre Flip: New capability (ambient capture)
   - 360: Paradigm shift

2. **No architectural rewrites between stages**
   - Plan the interfaces from day 1
   - Abstract the right things
   - Leave hooks for future features

3. **User value before technical elegance**
   - Ship the ollie even if tre flip isn't designed
   - Get feedback early and often
   - Let usage patterns guide development

4. **Build for extensibility, ship for simplicity**
   - Core must be plugin-ready
   - But MVP shouldn't expose complexity
   - Progressive disclosure of power

---

## Measuring Progress

### Stage 1 Milestones
- [ ] Week 1-2: Core capture + storage working
- [ ] Week 3-4: Semantic search beating keyword search  
- [ ] Week 5-6: Report generation providing value

### Stage 2 Milestones
- [ ] Month 1: Input router + 2 specialized processors
- [ ] Month 2: Graph relationships + rich queries
- [ ] Month 3: Plugin ecosystem + cloud option

### Go/No-Go Criteria
Before moving to next stage:
1. Current stage is daily-active useful
2. Architecture supports next stage without rewrites
3. Performance meets targets
4. Users are asking for next stage features

---

## The North Star

We're not building features. We're building toward a world where:
> "Computers understand what you mean, not just what you type"

Every stage moves us closer to this vision.