# chess

![chessgpt](https://github.com/EveryOneIsGross/chess/assets/23621140/1280658e-4af1-41bf-93ba-c5d6f6106b98)

I didn't have anyone to play chess with so I made openai play me. Uses embeddings to rank possible moves based on history. Unlike traditional chess engines that rely on predefined heuristics and complex algorithms, ChessBOT uses a unique method to rank and select moves, making it really bad at chess, but maybe it'll get good?

---

Features:

**OpenAI Integration:**
ChessBOT uses OpenAI API

**Sentence Transformers:**
ChessBOT uses Sentence Transformers to convert the state of the chess board and moves into embeddings, which are then used to calculate cosine similarities.

**Cosine Similarity:**
The cosine similarity between the current state/move and previous states/moves is used to rank the potential moves.

**Novel Move Generation:**
Instead of using traditional heuristics, ChessBOT ranks moves based on their cosine similarity scores and OpenAI evaluation.

**Memory:**
Stores embeddings for future recall as JSON.

```
r . b q k . . r
p p . p . p . .
B . . . p . p p
. . P . . . . .
Q . . P . . n .
. . . . P . . N
P . . . . P P P
R N B . K . . R
Enter your move:

```
