# chess
play chess with openai, uses embeddings to rank possible moves based on history. Unlike traditional chess engines that rely on predefined heuristics and complex algorithms, ChessBOT uses a unique method to rank and select moves, making it a challenging and unpredictable opponent.

Features
**OpenAI Integration:** ChessBOT uses OpenAI API
**Sentence Transformers:** ChessBOT uses Sentence Transformers to convert the state of the chess board and moves into embeddings, which are then used to calculate cosine similarities.
**Cosine Similarity:** The cosine similarity between the current state/move and previous states/moves is used to rank the potential moves.
**Novel Move Generation:** Instead of using traditional heuristics, ChessBOT ranks moves based on their cosine similarity scores and OpenAI evaluation.
