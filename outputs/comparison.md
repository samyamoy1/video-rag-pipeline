# Video question answering: results

Task: watch a short video (as pictures), answer a 4-option question.

## Score table (correct / questions)

| Version | What is moving | Where is it | What object | ALL |
|---|---|---|---|---|
| Random guessing | 1.2/5 | 1.2/5 | 1.2/5 | 3.8/15 (25%) |
| Search only (no LLM) | 0/5 | 0/5 | 3/5 | 3/15 (20%) |
| RAG + LLM | 2/5 | 1/5 | 3/5 | 6/15 (40%) |
| RAG + motion + LLM | 2/5 | 5/5 | 3/5 | 10/15 (67%) |

## Question by question

| Video | Type | Right answer | Search only | RAG + LLM | RAG + motion |
|---|---|---|---|---|---|
| v1_0 | What is moving | router | no | no | no |
| v1_1 | What is moving | router | no | yes | yes |
| v1_2 | What is moving | router | no | yes | yes |
| v1_3 | What is moving | dagger | no | no | no |
| v1_4 | What is moving | dagger | no | no | no |
| v2_0 | Where is it | front right | no | no | yes |
| v2_1 | Where is it | front right | no | yes | yes |
| v2_2 | Where is it | front right | no | no | yes |
| v2_3 | Where is it | back left | no | no | yes |
| v2_4 | Where is it | back left | no | no | yes |
| v3_0 | What object | blanket | yes | no | no |
| v3_1 | What object | blanket | yes | no | no |
| v3_2 | What object | blanket | yes | yes | yes |
| v3_3 | What object | dish | no | yes | yes |
| v3_4 | What object | dish | no | yes | yes |

## How each version works

- **Random guessing**: pick any option. About 25% right.
- **Search only (no LLM)**: write a note for each video frame, turn notes into numbers (embeddings), pick the option whose words are closest to the notes.
- **RAG + LLM**: same notes. Search finds the 5 notes closest to the question. A small language model reads them and picks an option.
- **RAG + motion + LLM**: same as above, plus a sentence from an OpenCV motion tracker (which way the object moves, where it sits in the frame).

## Things to remember

- Only 15 questions, and about 12 different videos (some videos are used by 2 or 3 questions). One question moves the score by about 7%. This is a good sign, not proof.
- The big gain is on the "Where is it" questions, because the motion tracker measures the answer directly.
- "What is moving" did not improve: the tracker says where and which way, but not what the object is.
