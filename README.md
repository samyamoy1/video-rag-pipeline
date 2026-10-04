# Video RAG: learning project

A small project to understand how a **RAG (Retrieval-Augmented Generation)** pipeline works, using video question answering as the example.

**Task:** watch a short video (as pictures) and answer a 4-option question, for example *"Where is the moving object located in the video?"*

This is a learning project, not a benchmark. It uses only 15 questions (about 12 different videos).

## How it works

```
Video
  -> frames (pictures)                step2
  -> caption each frame (Qwen-VL)     step3
  -> embeddings of captions (bge)     step4   <- the "index"
  -> search: top 5 captions for the   step5   <- the "R" in RAG
     question
  -> small LLM picks an option        step6   <- the "G" in RAG
  -> score

Extra tool (step 7): OpenCV motion tracking measures which way an
object moves and where it sits in the frame. This is passed to the LLM
as one extra sentence.
```

In plain words: a vision model writes a short note for each frame. Notes become numbers (embeddings). The question is also turned into numbers, and the 5 closest notes are found. A language model reads those notes and picks an answer.

## Results

Correct answers out of 5 per question type (15 total):

| Version | What is moving | Where is it | What object | ALL |
|---|---|---|---|---|
| Random guessing | 1.2/5 | 1.2/5 | 1.2/5 | 3.8/15 (25%) |
| Search only (no LLM) | 0/5 | 0/5 | 3/5 | 3/15 (20%) |
| RAG + LLM | 2/5 | 1/5 | 3/5 | 6/15 (40%) |
| RAG + motion + LLM | 2/5 | 5/5 | 3/5 | 10/15 (67%) |

Full question-by-question table: [outputs/comparison.md](outputs/comparison.md)

## What I learned

- **Retrieval worked.** Whenever the right answer was in the captions, search found it.
- **Captions were the weak part.** For 11 of 15 questions the answer word never appeared in any caption. Captions say what is in a picture, not what moves.
- **A tool fixed what text could not.** Motion tracking raised the "Where is it" questions from 1/5 to 5/5. The LLM mostly copies the measurement.
- **"What is moving" did not improve.** The tracker says where and which way, not what the object is.
- **Small test set.** One question is about 7% of the score. Treat the results as a direction, not proof. Some videos are reused by 2 or 3 questions.
- **Hardware matters.** Qwen2.5-VL-3B does not fit in 6 GB of GPU memory at full size, so it is loaded in 4-bit.

## Run it

Tested on Windows, Python 3.11, NVIDIA RTX 4050 (6 GB).

```bash
python -m venv venv
venv\Scripts\activate

# Install PyTorch first (command from https://pytorch.org/get-started/locally/)
pip install -r requirements.txt
pip install bitsandbytes
```

Run the steps in order:

```bash
python step1_download_data.py      # download 15 videos + questions
python step2_extract_frames.py     # video -> pictures
python step3_caption_frames.py     # picture -> short text note (slow, ~20 min)
python step4_build_index.py        # notes -> embeddings
python step5_retrieve.py           # question -> top 5 notes
python step6_answer.py             # LLM picks an answer, prints score
python step7_motion.py             # OpenCV motion tracking
python step7b_add_motion.py        # add motion sentence to questions
python step6_answer.py --input outputs/retrieved_motion.json --output outputs/results_rag_motion.json --label "RAG + motion"
python step8_compare.py            # one table with all versions
```

Settings (number of videos, models, frame rate) are all in `config.py`.

## Files

| File | What it does |
|---|---|
| `config.py` | Folders, model names, settings used by every script |
| `step1_download_data.py` | Downloads videos, questions, options, answers |
| `step2_extract_frames.py` | Saves frames: 4 per second for short clips, 1 per second for long ones |
| `step3_caption_frames.py` | Qwen2.5-VL-3B (4-bit) writes a note for each frame |
| `step4_build_index.py` | Turns notes into embeddings (bge-small-en-v1.5) |
| `step5_retrieve.py` | Finds the 5 closest notes per question, plus a baseline |
| `step6_answer.py` | Qwen2.5-1.5B reads the notes and picks an option, then scores |
| `step7_motion.py` | OpenCV frame differences give direction and position |
| `step7b_add_motion.py` | Writes the motion result as a sentence for each question |
| `step8_compare.py` | Builds the comparison table |

## Credits

- Dataset: [TaskMeAnything-v1-videoqa-random](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-videoqa-random)
- Models: Qwen2.5-VL-3B-Instruct, Qwen2.5-1.5B-Instruct, BAAI/bge-small-en-v1.5
