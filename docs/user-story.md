the purpose of this software is to visualize and understand and reference infromation within the computer. this is useful for organizing and retrieval.

1. inputs are multimodal.
2. processed into text and coupled with original content
3. index file generated for each directory
    - 1-3 sentence or 1 paragraph description of each file within
4. embeddings generated for each file
    - OVERENGINEERED VERSION:
        - chunking
            - paragraph
            - codeblock
                - method, comment,
            - diagram
        - ordered(think conversation back and forth easily reconstructed from paragraph chunks)
            - document: string *name* ListOfFloat *embeddings* listofDocs *contents*
                - user string *name* ListOfFloat *embeddings* listofDocs *contents*
                    - paragraph1 string *name* ListOfFloat *embeddings* string *contents*
                    - paragraph2 string *name* ListOfFloat *embeddings* string *contents*
                - assistant string *name* ListOfFloat *embeddings* listofDocs *contents*
                - user string *name* ListOfFloat *embeddings* listofDocs *contents*
                - assistant string *name* ListOfFloat *embeddings* listofDocs *contents*
3. hbd clustering
4. data transference
    - give to llm to restructure directories
    - graph to 2d visualization
    - write tags?
5. querying
6. easy lightweight ingestion of new documents
    - multimodal files
    - links
    
