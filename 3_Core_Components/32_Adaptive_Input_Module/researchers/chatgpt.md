\# Adaptive Input Module – LLD Research Findings



\## 1. User Experience \& Interaction Design



\* \*\*MVP:\*\* Prompt users immediately when a known schema is detected. E.g. print something like `> It appears this input is a prompt. \[Press Enter to confirm, 'n' to correct, or wait 3 seconds…]`. Use a short timeout (∼3s) to auto-confirm by default, minimizing friction. Always allow manual override (e.g. pressing ‘n’ to reject) so the user stays in control.

\* Provide configurable verbosity for feedback. For example, offer “concise” vs “verbose” modes so beginners see explanations and experts see minimal text. Users should be able to set these in config (or via flags) per the Configuration System.

\* Use a friendly, human-centric tone and avoid jargon. Phrase prompts as questions (“It looks like a recipe; use the \*cooking\* schema?”) to feel conversational. Adhere to CLI empathy guidelines – the tool should “feel like it’s on the user’s side”. E.g. include helpful default answers (as Inquirer.js does) to reinforce a collaborative feel.

\* \*\*Future (Kickflip/Tre Flip):\*\* Expand beyond single-turn queries to multi-turn dialogues for complex inputs (several back-and-forth questions if needed). Consider voice or GUI interfaces for accessibility. Allow users to skim or “press Tab to see more info” if they want detail (analogous to help mode), and to execute the input in silent mode otherwise.



\## 2. Schema Detection \& Application



\* Ship with a set of basic schemas and allow user-defined ones in YAML. For MVP (Ollie), implement simple pattern triggers: e.g. if input contains `http://` treat it as a \*\*link\*\* schema, if it contains JSON syntax detect \*\*structured data\*\*, if it starts with a command word detect a \*\*prompt\*\* schema. Provide templates for common cases (URLs, tasks, quotes) that users can edit.

\* On text input, call the Schema Engine to \*detect\* matching schemas. For example, use `SchemaEngine.detect\_schema(text)` to find all triggers (like “contains URL”, regex patterns, keywords). If exactly one schema applies, proceed; if multiple match, either apply the highest-priority one or ask the user to choose.

\* When a schema is triggered, \*apply its actions\*. These can include automatic functions (e.g. fetching a link’s title/description) and user prompts for extra context. For example, the `link\_curation` schema might do `fetch\_title`, `extract\_description`, then `prompt\_for\_context: "Why save this link?"`. The gathered context is then inserted into the output format of the schema.

\* \*\*Future:\*\* Move toward ML/semantic detection in later phases. Use embeddings or lightweight classifiers to recognize domains beyond static patterns (e.g. distinguishing “meeting notes” vs “personal reflection”). Allow composite schemas (chains of actions) and conditional logic (like IFTTT). Support multilingual detection and optionally let the Orchestration Engine suggest schemas based on semantic similarity to past inputs.



\## 3. Configuration \& Customization



\* Employ a \*\*configuration cascade\*\* (system defaults, user prefs, project overrides) as described in the architecture. The Input Module should read settings like verbosity level, confirmation timeout, enabled schemas, etc. from this system. E.g. if a user sets “quiet mode”, skip verbose prompts; if “tutorial mode” is on, give extra explanation.

\* Expose all behavior via editable config. For MVP, schemas are defined as YAML files that any user can modify. Store these in a user config directory. Configuration options should include: `auto\_confirm\_delay`, `verbosity (auto/concise/verbose)`, list of active schemas/triggers, prompt templates, etc. Changing the config (or YAML files) should take effect without code changes.

\* \*\*MVP (Ollie):\*\* Provide simple defaults so zero-config users have a good experience. Include a few example schemas (free-text, link, task). Settings like “explain prompts” or “silent mode” let users dial how chatty the module is.

\* \*\*Future (Kickflip/Tre Flip):\*\* Add interactive configuration commands (e.g. `globule config set input.timeout 5`) or a TUI for tweaking options. Allow importing schema libraries or plugins. Support runtime reloading of schemas/config without restart. Permit context-specific overrides (e.g. a “writing project” might use different schema priorities) via project-local config files.



\## 4. Technical Architecture



\* Build the module as an async service in `input\_adapter.py` with clear interfaces. Key methods include `async def process\_input(text: str) -> EnrichedInput`, `async def detect\_schema(text)`, `async def gather\_additional\_context(text, schema)`, and `get\_confirmation\_prompt(detected\_type)`. The `EnrichedInput` result should contain the original text, the applied schema, any fetched data, and user-provided context.

\* Perform schema detection and user prompts in a non-blocking fashion. Use asynchronous timers so a 3-second auto-confirm doesn’t freeze the app. For example, spawn a background asyncio task for the timeout while listening for keypresses. Use a library like Rich for formatted CLI output and possibly \\\[promptui/Inquirer] for better interactive prompts.

\* \*\*Modularity:\*\* Integrate via abstract interfaces so new handlers can plug in. Use plugin classes for specialized inputs (e.g. an `URLProcessor` for link inputs), and for domain schemas (implementing a `DomainPlugin` interface). The module should rely on the Schema Engine component to load and manage these plugins or YAML schemas, rather than hard-coding logic.

\* Inter-component communication: The Input Module sits in the pipeline before Orchestration. After processing, it should produce a `ProcessedGlobule` or similar and pass it (with metadata) to `OrchestrationEngine.process\_globule`. It should also call into the Configuration System for settings (timeout, verbosity) so its behavior follows user preferences. Keep the code decoupled so the CLI, TUI or API front-ends can reuse the same input logic.



\## 5. Integration Points



\* \*\*Schema Engine:\*\* The Input Module uses the Schema Engine to match and apply schemas. For each input, it should query something like `schema\_engine.detect(text)` to get candidate schemas. It then executes the schema’s actions (via the Schema Engine or a set of callbacks), e.g. fetching titles or asking for context. The final enriched text and any metadata come from this process.

\* \*\*Orchestration Engine:\*\* Once input is validated and enriched, hand it off to the Orchestration Engine along with any context. For instance, call `orchestrator.process\_globule(enriched\_text, context\_dict)`. Any user-provided answers (like “Why save this link?”) should be included in the data the Orchestration Engine receives, so downstream embedding/parsing can factor them in.

\* \*\*Configuration System:\*\* Query config at runtime. For example, read `input.auto\_confirm\_timeout` or `input.verbosity` before prompting. The Input Module should register its config keys so that changes to the config cascade (user or project settings) automatically adjust its behavior. Also, schema definitions come from config files, so reloading the Config System may cause new schemas to be used.

\* \*\*Other:\*\* Bind into the existing input pipeline (e.g. invoked by the `globule add` CLI handler). If the user calls the CLI API or voice interface in future, those should route raw input through this module. Also integrate with the Input Validator stage (sanitization, rate-limiting) so only clean text reaches the schema logic.



\## 6. Edge Cases \& Special Scenarios



\* \*\*Ambiguous/Multiple Matches:\*\* If more than one schema seems applicable, ask the user to choose (in MVP) or apply a default “general” schema. The system should handle ambiguity gracefully by clarifying or doing nothing disruptive.

\* \*\*No Match:\*\* If no schema matches, simply treat the input as plain text (free-form globule) and proceed. Do not error out; just skip schema processing.

\* \*\*User Cancels/Timeouts:\*\* If the user declines a suggestion (presses ‘n’) or fails to respond, the module should either let them re-enter input or continue without that schema’s actions. For example, if the user doesn’t answer a follow-up question, proceed with empty context. (In MVP, simply continue; in future, maybe allow editing/retry.)

\* \*\*Action Failures:\*\* If a schema action fails (e.g. fetching URL metadata errors, or the user enters invalid JSON), log a warning and skip that action. The module should not crash; it should default to as much successful processing as possible.

\* \*\*Structured Inputs:\*\* If the user inputs JSON/CSV or Markdown, detect these (as “structured data”) and parse/validate accordingly. For example, if JSON is malformed, show an error to the user. If input is an image file path or other non-text, reject with a clear message.

\* \*\*False Positives:\*\* Avoid overly broad triggers. E.g. matching “\[www.”](http://www.”) might catch code. Always let the user correct a wrong guess (press ‘n’). The “press n to correct” prompt handles accidental detections.

\* \*\*Special Modes:\*\* If the user is in a project-specific mode (via config), apply only that context’s schemas. If the system is offline (no network), skip web-dependent actions.

\* \*Note:\* Extensibility means more edge cases later (e.g. mixed media inputs), but the core should degrade gracefully for missing features.



\## 7. Performance Requirements



\* \*\*Latency:\*\* The module’s feedback must be fast – ideally <50–100ms to detect schema and show a prompt. This ensures that CLI interactions feel instantaneous (target <100ms as per success criteria). Use async I/O and non-blocking calls to meet this.

\* \*\*Resource Use:\*\* Design for modest hardware (≥8GB RAM, dual-core). Use lightweight local models (e.g. 3B Llama or sentence-transformers) so embedding/parsing can run in-memory without GPU. All local processes should fit in RAM (e.g. avoid huge caches). For heavy tasks (like LLM parsing), allow async calls to external services if needed.

\* \*\*Local-first / Hybrid:\*\* By default everything should run on-device for privacy and offline capability. In future phases, support optional cloud/offloading (e.g. cloud LLM APIs) as plugins. For example, if a local model isn’t available, fall back to an open-source or cloud API.

\* \*\*Throughput:\*\* Although the CLI is single-user, design so that multiple `globule add` commands (or batched inputs) can be handled without locking. Use concurrency for independent tasks (e.g. URL fetch + local parse can run in parallel).

\* \*\*Scalability:\*\* The module should scale to large schema sets or many triggers by efficient matching (e.g. pre-compile regexes, index triggers). Maintain <100ms response even as user scripts invoke it repeatedly.

\* \*\*Metrics:\*\* Continuously monitor performance (e.g. log and profile response times). Performance tests should verify the targets in real environments.



\## 8. User Research Questions



\* \*\*MVP (Ollie):\*\* Is the single-turn schema prompt clear and helpful? For example, do users correctly interpret “Press Enter to confirm or ‘n’ to correct”? Is 3 seconds an acceptable timeout, or do they want a longer/shorter delay?

\* \*\*MVP:\*\* How do users feel about verbosity modes? Would they find a concise (no explanation) default better, or a chatty (explain-everything) default? How should they toggle this (flag, config)?

\* \*\*MVP:\*\* Which input types do users encounter most often, and which do they want automated? (E.g. saving URLs, creating tasks, quoting text.) Prioritize schemas based on this feedback.

\* \*\*Kickflip/Tre Flip:\*\* Would users want multi-turn conversation (follow-up questions) or is a one-shot suggestion sufficient? For instance, if input is ambiguous, do they want the system to ask a question, or just skip?

\* \*\*Future:\*\* How comfortable are users editing schemas in YAML? Would they prefer a GUI or a simpler DSL? What tools (templates, examples) would help them define new schemas?

\* \*\*Future:\*\* What default behaviors/integration would delight users? (e.g. automatic linking between related entries, undo/redo of input actions, etc.)



\## 9. Testing \& Validation Strategy



\* \*\*Unit Tests:\*\* Cover each piece of logic: schema trigger matching, prompt formatting, context gathering, confirmation timeout. Write tests for each built-in schema (e.g. link schema should trigger on “http\\://” and skip on plain text). Validate that YAML schemas load and produce expected outputs.

\* \*\*Conversation Simulation:\*\* Use CLI testing tools (like `pexpect` or Python’s built-in `pty`) to simulate user input and keypresses. Test scenarios: user presses Enter, presses ‘n’, or times out, and verify the module’s behavior and outputs (including exit codes).

\* \*\*Performance Testing:\*\* Measure detection and prompt latency on target hardware. Write benchmarks to ensure schema detection <100ms and overall input processing stays within targets (per Success Criteria). Automated CI tests should flag regressions.

\* \*\*Accuracy Metrics:\*\* Create a labeled dataset of test inputs (e.g. 100 examples of URLs, tasks, free text, etc.). Compute schema detection accuracy (target ≥90%). Investigate any misclassifications and refine rules or add exceptions.

\* \*\*Integration \& End-to-End:\*\* Test the full pipeline: entering input via CLI, module responses, and final handoff to Orchestration/Storage. For each config setting (verbosity, timeout), verify the visible behavior changes accordingly.

\* \*\*Edge Case Testing:\*\* Deliberately feed bad data: malformed URLs, enormous inputs, SQL injection attempts, unsupported commands. Ensure the module fails gracefully (e.g. error messages, no data corruption).

\* \*\*User Acceptance Testing:\*\* Have real users try common tasks with and without adaptive prompts. Collect qualitative feedback on clarity and usefulness. Use this feedback to tune messages and defaults.

\* \*\*Regression Safety:\*\* Include tests for any user corrections. For example, if a user corrects a schema suggestion (presses ‘n’ and re-enters), verify the new input is processed correctly.



\## Additional Insights



\* Leverage interactive CLI frameworks for richer prompts. Tools like \*\*promptui\*\* (Go) or \*\*Inquirer.js\*\* (Node) support default answers and multi-step questions. For instance, Inquirer’s “default” values let the user simply press ENTER to accept a suggestion, which mirrors our auto-confirm feature.

\* Provide immediate previews of schema results. E.g. when detecting a link, show the fetched title/description inline so the user can adjust context before confirming. This “preview” approach (common in GUI note-taking apps) gives confidence in automated actions.

\* Allow community or shareable schema packs. Users could import/export schema YAML sets (like a plugin), enabling workflows (e.g. blog-post schema, meeting notes schema) that others have refined.

\* Explore AI-assisted schema creation. For example, let a user describe a workflow in natural language and use an LLM to generate a draft YAML schema. This lowers the barrier to customization.

\* Integrate with external services: e.g. if input is an address or event, prompt to add to calendar; if it looks like a quote, offer to add to a “quotes” category. Similar to chatbots’ contextual actions (like Smart Replies), Globule could suggest relevant automations based on input content.



