<svg viewBox="0 0 1400 1000" xmlns="http://www.w3.org/2000/svg">
  <!-- Title -->
  <text x="700" y="30" text-anchor="middle" font-size="24" font-weight="bold">Globule Architecture: From Input to Insight</text>
  
  <!-- Input Layer -->
  <g id="input-layer">
    <rect x="50" y="70" width="200" height="60" fill="#E3F2FD" stroke="#1976D2" stroke-width="2" rx="5"/>
    <text x="150" y="105" text-anchor="middle" font-size="14" font-weight="bold">Voice Input</text>
    
    <rect x="50" y="150" width="200" height="60" fill="#E3F2FD" stroke="#1976D2" stroke-width="2" rx="5"/>
    <text x="150" y="185" text-anchor="middle" font-size="14" font-weight="bold">Text/CLI Input</text>
    
    <rect x="50" y="230" width="200" height="60" fill="#E3F2FD" stroke="#1976D2" stroke-width="2" rx="5"/>
    <text x="150" y="265" text-anchor="middle" font-size="14" font-weight="bold">URL/File Input</text>
  </g>
  
  <!-- Raw Globule Creation -->
  <rect x="350" y="120" width="180" height="120" fill="#FFF3E0" stroke="#F57C00" stroke-width="2" rx="5"/>
  <text x="440" y="145" text-anchor="middle" font-size="14" font-weight="bold">Raw Globule</text>
  <text x="440" y="165" text-anchor="middle" font-size="12">{</text>
  <text x="440" y="180" text-anchor="middle" font-size="12">id: uuid,</text>
  <text x="440" y="195" text-anchor="middle" font-size="12">content: "raw text",</text>
  <text x="440" y="210" text-anchor="middle" font-size="12">timestamp: now,</text>
  <text x="440" y="225" text-anchor="middle" font-size="12">type: "voice|text|url"</text>
  <text x="440" y="240" text-anchor="middle" font-size="12">}</text>
  
  <!-- Processing Pipeline -->
  <g id="processing">
    <!-- Embedding -->
    <rect x="620" y="70" width="200" height="80" fill="#E8F5E9" stroke="#388E3C" stroke-width="2" rx="5"/>
    <text x="720" y="95" text-anchor="middle" font-size="14" font-weight="bold">Embedding Engine</text>
    <text x="720" y="115" text-anchor="middle" font-size="12">sentence-transformers/</text>
    <text x="720" y="130" text-anchor="middle" font-size="12">all-MiniLM-L6-v2</text>
    <text x="720" y="145" text-anchor="middle" font-size="11" fill="#666">→ 384-dim vector</text>
    
    <!-- LLM Parser -->
    <rect x="620" y="180" width="200" height="120" fill="#F3E5F5" stroke="#7B1FA2" stroke-width="2" rx="5"/>
    <text x="720" y="205" text-anchor="middle" font-size="14" font-weight="bold">LLM Parser</text>
    <text x="720" y="225" text-anchor="middle" font-size="12">Domain Router</text>
    <text x="720" y="245" text-anchor="middle" font-size="11" fill="#666">↓</text>
    <text x="720" y="260" text-anchor="middle" font-size="11">Schema Selection</text>
    <text x="720" y="275" text-anchor="middle" font-size="11">Entity Extraction</text>
    <text x="720" y="290" text-anchor="middle" font-size="11">Structured Output</text>
  </g>
  
  <!-- Domain Schemas -->
  <g id="schemas">
    <rect x="900" y="160" width="160" height="60" fill="#FCE4EC" stroke="#C2185B" stroke-width="1" rx="3"/>
    <text x="980" y="180" text-anchor="middle" font-size="12" font-weight="bold">Valet Schema</text>
    <text x="980" y="195" text-anchor="middle" font-size="10">event, employee,</text>
    <text x="980" y="208" text-anchor="middle" font-size="10">customer, location</text>
    
    <rect x="900" y="230" width="160" height="60" fill="#FCE4EC" stroke="#C2185B" stroke-width="1" rx="3"/>
    <text x="980" y="250" text-anchor="middle" font-size="12" font-weight="bold">Research Schema</text>
    <text x="980" y="265" text-anchor="middle" font-size="10">topic, source,</text>
    <text x="980" y="278" text-anchor="middle" font-size="10">insight_type, tags</text>
    
    <rect x="900" y="300" width="160" height="60" fill="#FCE4EC" stroke="#C2185B" stroke-width="1" rx="3"/>
    <text x="980" y="320" text-anchor="middle" font-size="12" font-weight="bold">Generic Schema</text>
    <text x="980" y="335" text-anchor="middle" font-size="10">entities, actions,</text>
    <text x="980" y="348" text-anchor="middle" font-size="10">sentiment, intent</text>
  </g>
  
  <!-- Storage Layer -->
  <g id="storage">
    <rect x="350" y="400" width="180" height="100" fill="#E1F5FE" stroke="#0277BD" stroke-width="2" rx="5"/>
    <text x="440" y="425" text-anchor="middle" font-size="14" font-weight="bold">File Storage</text>
    <text x="440" y="445" text-anchor="middle" font-size="12">JSON Files</text>
    <text x="440" y="465" text-anchor="middle" font-size="11" fill="#666">• Raw content</text>
    <text x="440" y="480" text-anchor="middle" font-size="11" fill="#666">• Parsed data</text>
    <text x="440" y="495" text-anchor="middle" font-size="11" fill="#666">• Metadata</text>
    
    <rect x="620" y="400" width="180" height="100" fill="#E1F5FE" stroke="#0277BD" stroke-width="2" rx="5"/>
    <text x="710" y="425" text-anchor="middle" font-size="14" font-weight="bold">Vector DB</text>
    <text x="710" y="445" text-anchor="middle" font-size="12">ChromaDB</text>
    <text x="710" y="465" text-anchor="middle" font-size="11" fill="#666">• Embeddings</text>
    <text x="710" y="480" text-anchor="middle" font-size="11" fill="#666">• Metadata index</text>
    <text x="710" y="495" text-anchor="middle" font-size="11" fill="#666">• Similarity search</text>
  </g>
  
  <!-- Retrieval & Synthesis -->
  <g id="retrieval">
    <!-- Query Processing -->
    <rect x="150" y="600" width="200" height="80" fill="#EFEBE9" stroke="#5D4037" stroke-width="2" rx="5"/>
    <text x="250" y="625" text-anchor="middle" font-size="14" font-weight="bold">Query Engine</text>
    <text x="250" y="645" text-anchor="middle" font-size="12">"Show today's events"</text>
    <text x="250" y="660" text-anchor="middle" font-size="12">"Staff performance"</text>
    <text x="250" y="675" text-anchor="middle" font-size="12">"Damage incidents"</text>
    
    <!-- Retrieval Methods -->
    <rect x="450" y="580" width="180" height="50" fill="#F0F4C3" stroke="#827717" stroke-width="2" rx="5"/>
    <text x="540" y="610" text-anchor="middle" font-size="12" font-weight="bold">Semantic Search</text>
    
    <rect x="450" y="640" width="180" height="50" fill="#F0F4C3" stroke="#827717" stroke-width="2" rx="5"/>
    <text x="540" y="670" text-anchor="middle" font-size="12" font-weight="bold">Structured Query</text>
    
    <rect x="450" y="700" width="180" height="50" fill="#F0F4C3" stroke="#827717" stroke-width="2" rx="5"/>
    <text x="540" y="730" text-anchor="middle" font-size="12" font-weight="bold">Time-based Filter</text>
  </g>
  
  <!-- Output Generation -->
  <rect x="750" y="600" width="250" height="150" fill="#E8EAF6" stroke="#3F51B5" stroke-width="2" rx="5"/>
  <text x="875" y="625" text-anchor="middle" font-size="14" font-weight="bold">Report Generator</text>
  <text x="875" y="645" text-anchor="middle" font-size="12">Domain Template Engine</text>
  <line x1="770" y1="655" x2="980" y2="655" stroke="#666" stroke-width="1"/>
  <text x="875" y="675" text-anchor="middle" font-size="11">• Aggregate metrics</text>
  <text x="875" y="690" text-anchor="middle" font-size="11">• Narrative synthesis</text>
  <text x="875" y="705" text-anchor="middle" font-size="11">• Pattern detection</text>
  <text x="875" y="720" text-anchor="middle" font-size="11">• Recommendations</text>
  <text x="875" y="735" text-anchor="middle" font-size="11">• Formatted output</text>
  
  <!-- Final Output -->
  <rect x="1100" y="620" width="250" height="110" fill="#C8E6C9" stroke="#388E3C" stroke-width="2" rx="5"/>
  <text x="1225" y="645" text-anchor="middle" font-size="14" font-weight="bold">Daily Passdown Sheet</text>
  <text x="1225" y="665" text-anchor="middle" font-size="11">✓ Staff summary</text>
  <text x="1225" y="680" text-anchor="middle" font-size="11">✓ Key incidents</text>
  <text x="1225" y="695" text-anchor="middle" font-size="11">✓ Metrics & KPIs</text>
  <text x="1225" y="710" text-anchor="middle" font-size="11">✓ Tomorrow's actions</text>
  
  <!-- Arrows showing data flow -->
  <!-- Input to Raw Globule -->
  <path d="M 250 100 L 350 160" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 250 180 L 350 180" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 250 260 L 350 200" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Raw to Processing -->
  <path d="M 530 160 L 620 110" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 530 200 L 620 240" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Parser to Schemas -->
  <path d="M 820 240 L 900 190" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 820 240 L 900 260" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 820 240 L 900 330" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Processing to Storage -->
  <path d="M 720 150 L 710 400" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 720 300 L 440 400" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Query to Retrieval -->
  <path d="M 350 640 L 450 605" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 350 640 L 450 665" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 350 640 L 450 725" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Storage to Retrieval -->
  <path d="M 440 500 L 540 580" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 710 500 L 540 580" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Retrieval to Generator -->
  <path d="M 630 605 L 750 630" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 630 665 L 750 665" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  <path d="M 630 725 L 750 700" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Generator to Output -->
  <path d="M 1000 675 L 1100 675" stroke="#666" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
  
  <!-- Arrow marker definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
  
  <!-- Key architectural notes -->
  <text x="50" y="820" font-size="16" font-weight="bold">Key Architectural Principles:</text>
  <text x="50" y="845" font-size="14">1. Domain-agnostic core with pluggable schemas (valet, research, generic fallback)</text>
  <text x="50" y="865" font-size="14">2. Parallel processing: embedding and parsing happen concurrently</text>
  <text x="50" y="885" font-size="14">3. Hybrid retrieval: semantic search for concepts, structured queries for specific data</text>
  <text x="50" y="905" font-size="14">4. Template-driven output allows domain-specific report formats</text>
  <text x="50" y="925" font-size="14">5. All components are modular and can be extended without changing core flow</text>
</svg>
