# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

### How real platforms do it (and what I'm doing instead)

Services like Spotify and YouTube combine many signals instead of relying on one formula. They look at **behavior** (what you play, skip, save, search for, share, or how long you watch), at **other users** (collaborative filtering: "people whose history looks like yours also liked this"), and at **the content itself** (genre, release date, audio characteristics). Spotify also mixes in human editors who pick a pool of candidate tracks that an algorithm then personalizes and orders for each listener.

This project has no listening history and no other users. It has 10 songs with descriptive attributes and one user who says what they like. That makes **content-based filtering** the right fit: compare each song's attributes with the user's stated preferences. The recommender will be a deterministic, hand-weighted scoring rule rather than a trained model, so every score can be traced back to the exact features that produced it.

### What a `Song` stores

Each row of `data/songs.csv` becomes a `Song` with:

- **Identity / metadata:** `id`, `title`, `artist`. These are used for display and tie-breaking, not scoring.
- **Categorical features:** `genre`, `mood`
- **Numeric features (0–1 scale):** `energy`, `valence`, `danceability`, `acousticness`
- **Numeric feature (BPM):** `tempo_bpm`

### What a `UserProfile` stores

A user describes their taste explicitly. No history is needed:

- `favorite_genre`: e.g. `"lofi"`
- `favorite_mood`: e.g. `"chill"`
- `target_energy`: a number from 0 to 1 for how energetic they want the music to be
- `likes_acoustic`: `True` or `False`

### Planned scoring rule (one song vs. one user)

Each song will earn points from four parts (max 11.0):

| Part | Points | How |
|---|---|---|
| Genre match | 4.0 | exact match with `favorite_genre`, otherwise 0 |
| Mood match | 3.0 | exact match with `favorite_mood`, otherwise 0 |
| Energy similarity | up to 3.0 | `3.0 × max(0, 1 − abs(song.energy − target_energy))` |
| Acoustic fit | up to 1.0 | `acousticness` if `likes_acoustic`, else `1 − acousticness` |

Genre gets the most weight because it's the broadest signal of taste. Someone who likes lofi usually won't want rock even when the mood matches. Energy is scored by **closeness**, not size, so a user who wants 0.40 energy will get more credit from a 0.42 song than a 0.90 song. Tempo, valence, and danceability will stay in the data for later experiments but won't be part of the first version of the score.

For example, *Midnight Coding* (lofi, chill, energy 0.42, acousticness 0.71) scored for a user who wants lofi / chill / 0.40 / likes acoustic would earn 4.0 + 3.0 + 2.94 + 0.71 = **10.65**. The explanation would say something like: *"matches your favorite genre (lofi), matches your mood (chill), energy 0.42 is close to your target 0.40, fairly acoustic (0.71)."*

### Planned ranking rule (all songs → top K)

1. Score every song in the catalog with the scoring rule.
2. Sort by score, highest first. Ties are broken by lower `id`, so results are always the same.
3. Return the top `k` songs, each with its score and a short explanation.

Scoring and ranking are kept separate on purpose. Scoring answers "how well does *this one* song fit?", and ranking answers "which songs are *best overall*?" Keeping them apart means the scoring math can be tested on a single song, and the ranking can change (for example, adding variety) without touching the scoring.

```
songs.csv ──► Song objects ──┐
                             ├──► score each song ──► sort (score ↓, id ↑) ──► top K + explanations
UserProfile ─────────────────┘   (genre, mood, energy, acoustic)
```

More detail on the research and design decisions is in [docs/phase1_notes.md](docs/phase1_notes.md).

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



