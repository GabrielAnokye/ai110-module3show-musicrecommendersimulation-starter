# Phase 1 Notes: Understanding the Problem

Supporting notes for the "How The System Works" section of the README. Nothing here is implemented yet. This is the design that Phase 2 will build.

---

## 1. Research summary

### Collaborative filtering
Collaborative filtering recommends items using **similarities between users and items at the same time**. If two people's histories overlap, things one of them liked become candidates for the other. Google's ML course describes it as using "similarities between users and items simultaneously."

- **Feedback can be explicit** (ratings, thumbs up/down) or **implicit** (the system infers interest from a play, a save, or watch time).
- On music platforms, likes, saves, playlist adds, and long listens push a prediction up. Skips, dislikes, and "not interested" push it down.
- **Example:** Ana and Ben both play a lot of lofi and jazz and both saved *Library Rain*. Ben also replays *Coffee Shop Stories*, which Ana has never heard. Because their histories overlap, the system predicts Ana will probably like it too. No song attributes are needed.
- **Limitations:**
  - **Cold start:** a brand-new song or user has no interactions, so the model has nothing to compare. Google's course notes a model "can't create an embedding" for an item not seen during training.
  - It needs a lot of interaction data from many users.
  - Popular items get more interactions, so they can get recommended even more.

### Content-based filtering
Content-based filtering compares **a song's attributes** with **one user's preferences**. It doesn't need other users.

| Feature | Role |
|---|---|
| Genre | broad style category; the strongest single taste signal |
| Mood | the emotional label a listener expects |
| Energy | intensity/activity (calm ↔ intense) |
| Tempo | speed in BPM |
| Valence | musical positivity (sad/dark ↔ happy/bright) |
| Danceability | rhythmic suitability for dancing |
| Acousticness | acoustic vs. electronic/produced sound |

- **Example:** a user who likes chill, low-energy music (mood `chill`, target energy ≈ 0.35) would see songs like *Library Rain* (lofi, chill, 0.35) and *Spacewalk Thoughts* (ambient, chill, 0.28) rise to the top. *Gym Hero* (0.93 energy, intense) would sink.
- **Why it fits this project:** we have 10 songs, 1 user, and no interaction history. Collaborative filtering would have no data to work with, while content-based scoring only needs the attributes we already have. Google's course also notes it can serve niche tastes. Its main weakness is that it can't easily expand beyond what the user already likes.

### Real systems vs. our simulation
Real platforms **blend signals** rather than using one formula:

- **Spotify** says it uses your *taste profile* (searching, listening, skipping, saving), information you provide and context (age, follows, device, language, location), general *trends* across users, *content attributes* (genre, release date, etc.), safety rules, and human **editorial** curation. Its "Algotorial" playlists use editors to build a candidate pool, then algorithms select and order tracks per listener. Spotify also discloses a commercial signal (Discovery Mode) that can raise the chance a song is recommended.
- **YouTube** says it uses clicks, watch time, survey responses ("valued watchtime"), shares, likes, and dislikes. It notes that clicks alone are misleading, and that it "doesn't follow a fixed formula." It also says responsibility goals can override engagement signals.

These terms will come up throughout the design, so here is what each one means:

| Term | Meaning | In this project? |
|---|---|---|
| User behavior | what someone actually did (plays, skips, saves) | ❌ not available |
| Song attributes | facts about each song (genre, energy, …) | ✅ `songs.csv` |
| User preferences | what someone says they want | ✅ `UserProfile` |
| Similarity scoring | how close a song is to the preferences | ✅ planned scoring rule |
| Ranking | ordering all scored songs and picking the top K | ✅ planned ranking rule |

**Why a deterministic weighted rule instead of ML:** there's no training data (no behavior, no labels), the catalog is tiny, and the assignment values explainability. A hand-weighted rule gives the same output every time, can be unit-tested exactly, and lets every point in a score be explained to the user.

> I did not find, and the notes don't claim, any proprietary details of Spotify's or YouTube's actual models or weights. Everything above comes from the companies' own public pages linked below.

---

## 2. Dataset analysis (`data/songs.csv`)

**Shape:** 10 rows × 10 columns. No missing values, no duplicate IDs, no whitespace or line-ending problems. IDs run from 1 to 10.

| Column | Type | Range / values | Use |
|---|---|---|---|
| `id` | int | 1–10 | identity, tie-breaker |
| `title` | str | unique | display only |
| `artist` | str | 8 artists (Neon Echo ×2, LoRoom ×2) | display only (possible diversity rule later) |
| `genre` | str | lofi 3, pop 2, rock, ambient, jazz, synthwave, indie pop | **primary** |
| `mood` | str | chill 3, happy 2, intense 2, relaxed, moody, focused | **primary** |
| `energy` | float | 0.28–0.93 (mean 0.60) | **primary numeric** |
| `tempo_bpm` | float | 60–152 (mean 101.6) | secondary (needs normalizing) |
| `valence` | float | 0.48–0.84 (mean 0.65) | secondary |
| `danceability` | float | 0.41–0.88 (mean 0.66) | secondary |
| `acousticness` | float | 0.05–0.92 (mean 0.51) | **secondary (used)** |

### Feature notes

- **Genre** is the broadest predictor of taste and should be a **primary** feature. It is an exact string match, though, so `"indie pop"` gets no credit from a `"pop"` fan. Five of the seven genres have only one song.
- **Mood** is the listener's intent ("I want something chill"), which makes it **primary**. Its labels overlap: chill, relaxed, and focused are close neighbors, but exact matching treats them as completely different.
- **Energy** separates workout tracks from study tracks well, and it has the widest spread of any feature, so it is **primary**. It overlaps with tempo and danceability.
- **Acousticness** captures the acoustic vs. electronic "texture" that the existing `likes_acoustic` field asks about, so it is **secondary**. It is strongly anti-correlated with energy in this data: every song with acousticness ≥ 0.71 has energy ≤ 0.42.
- **Tempo** helps tell a fast workout track from a slow ballad, but it is mostly redundant with energy here, so it is **secondary (future)**. It is in BPM rather than 0–1, so it would need normalizing first, e.g. `(bpm − 60) / (152 − 60)`.
- **Valence** separates happy from dark songs at the same energy, making it **secondary (future)**. Its range is narrow (0.48–0.84), so nothing in the catalog is truly sad.
- **Danceability** is useful for party or workout profiles, but it is **secondary (future)** because it overlaps with energy and tempo.

### Mood vs. energy vs. valence
They're related but **not interchangeable**:
- **Energy** is *intensity* (calm ↔ intense).
- **Valence** is *positivity* (dark ↔ bright).
- **Mood** is a human-readable label that roughly combines the two, plus context like "focused."

The data shows why one can't replace another:
- *Storm Runner* and *Gym Hero* are both `intense` with energy 0.91 / 0.93, but their valence is 0.48 vs. 0.77 (dark vs. euphoric).
- *Night Drive Loop* (`moody`, energy 0.75) and *Rooftop Lights* (`happy`, energy 0.76) have nearly identical energy. Only valence (0.49 vs. 0.81) and the mood label tell them apart.

### Dataset limitations
- Only 10 songs, and 5 of the 7 genres have a single song. Top-5 lists will always include weak matches.
- There is no hip-hop, R&B, classical, country, metal, Latin, or non-English music. The catalog reflects a narrow "study/chill + pop" slice of taste.
- Two artists account for 4 of the 10 songs.
- No song has low valence, so a "sad" listener can't be served well.
- Labels are single-valued: a song can't be both `chill` and `focused`.

---

## 3. Algorithm Recipe

**Inputs:** one `Song`, one `UserProfile`.

```
genre_score    = 4.0  if song.genre == user.favorite_genre  else 0
mood_score     = 3.0  if song.mood  == user.favorite_mood   else 0
energy_score   = 3.0 * max(0, 1 - abs(song.energy - user.target_energy))
acoustic_score = 1.0 * (song.acousticness       if user.likes_acoustic
                        else 1 - song.acousticness)

score = genre_score + mood_score + energy_score + acoustic_score      # range 0 – 11
```

- **Why genre (4) > mood (3):** genre reflects long-term taste, while mood is more situational. A lofi fan in an "intense" mood still probably wants lofi more than they want rock. Combined, genre and mood (7 points) outweigh perfect energy plus acoustic fit (4 points), so labels dominate and the numeric features act as fine-tuning.
- **Why similarity, not magnitude:** rewarding high energy would push *Gym Hero* to the top for everyone. Using distance from the target means a 0.40 target gives 0.42 → 2.94 points and 0.90 → 1.50 points.
- **Acoustic tradeoff:** the boolean `likes_acoustic` becomes a continuous value through `acousticness`.
  - The recommended version treats `False` as "prefers produced/electronic" and gives `1 − acousticness`. It never subtracts points, so scores stay in the 0–11 range and are easy to read.
  - The alternative is to *penalize* acoustic songs (e.g. `−acousticness`) for `False`. That expresses dislike more strongly, but it allows negative contributions and makes scores harder to explain.
  - A third option treats `False` as "no preference" and gives 0. That's safer if a user simply didn't answer, but it makes the field one-directional.
- **What changing weights does:** raise mood above genre and *Rooftop Lights* (indie pop, happy) would beat *Gym Hero* (pop, intense) for a pop/happy user. Raise energy to 5 and numeric closeness could override a genre mismatch. These are the Phase 4 experiments.
- **Kept for later:** tempo (normalized), valence, and danceability can be added as extra `w × (1 − distance)` terms without changing the structure.

### Worked example (from `songs.csv`)
User: `favorite_genre="lofi"`, `favorite_mood="chill"`, `target_energy=0.40`, `likes_acoustic=True`
Song: **#2 *Midnight Coding*** (lofi, chill, energy 0.42, acousticness 0.71)

| Part | Calculation | Points |
|---|---|---|
| Genre | lofi = lofi | 4.00 |
| Mood | chill = chill | 3.00 |
| Energy | 3.0 × (1 − 0.02) | 2.94 |
| Acoustic | 0.71 | 0.71 |
| **Total** | | **10.65** |

**Explanation text:** *"Matches your favorite genre (lofi) and mood (chill). Its energy (0.42) is very close to your target (0.40), and it's fairly acoustic (0.71), which you said you like."*

**Design check (hand-calculated, not program output):** *Library Rain* (lofi, chill, 0.35, acousticness 0.86) would score 10.71 and narrowly beat *Midnight Coding*. Its higher acousticness outweighs its slightly worse energy match. It's worth watching whether the 1-point acoustic term should be allowed to decide near-ties like this.

The starter test's two fixture songs would score 10.8 (pop/happy) vs. 1.9 (lofi/chill) for the test's pop/happy/0.8/non-acoustic user, so this design satisfies the existing test's expectation.

---

## 4. Scoring rule vs. ranking rule

- **Scoring rule:** one song + one user → a number (plus reasons). It knows nothing about the other songs.
- **Ranking rule:** score *every* song, sort by `(score descending, id ascending)`, and return the first `k`.
  - **Deterministic tie-break:** lower `id` wins.
  - If `k` is larger than the catalog, return every song. If `k ≤ 0`, return an empty list.

Keeping them separate means the math can be unit-tested on a single song. It also lets later ranking changes, like artist or genre diversity, happen without editing the scoring.

---

## 5. Conceptual architecture

```
CSV ─► load_songs() ─► list of song dicts / Song objects
                                │
UserProfile / user_prefs ───────┤
                                ▼
                   score_song(): feature comparison ─► weighted score + reasons
                                │
                                ▼
       recommend_songs() / Recommender.recommend(): sort ─► top K
                                │
                                ▼
          explanations (Recommender.explain_recommendation / reasons)
```

- **`Song`** (existing dataclass): `id`, `title`, and `artist` are identity. `genre`, `mood`, `energy`, `tempo_bpm`, `valence`, `danceability`, and `acousticness` are features.
- **`UserProfile`** (existing dataclass): `favorite_genre`, `favorite_mood`, `target_energy`, `likes_acoustic`. These are explicit preferences, so no history is required.
- **`Recommender(songs)`**: `recommend(user, k)` returns `List[Song]`, and `explain_recommendation(user, song)` returns `str`.
- **Functional API**: `load_songs(path) -> List[Dict]`, `score_song(user_prefs, song) -> (score, reasons)`, and `recommend_songs(user_prefs, songs, k) -> [(song_dict, score, explanation)]`.

**Avoiding duplicated logic:** the weights will live as module-level constants, and the formula will be written **once**, in a single scoring function. The OOP methods will convert `Song`/`UserProfile` into the same inputs (e.g. `dataclasses.asdict(song)`) and delegate to it. `explain_recommendation` will reuse the scoring function's `reasons` list, so the explanation can never disagree with the score.

---

## 6. Limitations and filter-bubble concerns

- **Filter bubble:** genre plus mood are worth 7 of 11 points, so users will mostly see songs they already know they like. Content-based filtering has no built-in way to suggest something new.
- **Exact string matching:** "indie pop" ≠ "pop", and "relaxed" ≠ "chill." Near-misses get zero credit.
- **Small, skewed catalog:** lofi and chill listeners get three strong matches. Rock, jazz, and ambient fans get one at most, so the system may look less fair to them.
- **Weights are assumptions,** not learned or validated.
- **Static profile:** no feedback loop, so the system can't learn from skips the way real platforms do.

---

## 7. Starter-code observations (for Phase 2, not changed now)

- `src/main.py` uses `from recommender import ...`, but the README runs `python -m src.main`. That currently fails with `ModuleNotFoundError: No module named 'recommender'`.
- `main.py`'s profile dict uses keys `genre`, `mood`, `energy` and has no acoustic key. `UserProfile` uses `favorite_genre`, `favorite_mood`, `target_energy`, `likes_acoustic`. The functional API will need to handle that mapping.
- `score_song` currently returns `[]` instead of the documented `(score, reasons)` tuple.
- Running bare `pytest` fails at collection (`No module named 'src'`). `python -m pytest` runs and both tests pass, but only because the stub returns `songs[:k]` and the pop fixture happens to be first, not because ranking works.

---

## 8. Decisions to confirm

1. **Meaning of `likes_acoustic=False`:** "prefers electronic" (reward `1 − acousticness`, recommended), "dislikes acoustic" (penalty), or "no preference" (0)?
2. **Starting weights:** 4 / 3 / 3 / 1 (labels dominate). OK as the Phase 2 baseline?
3. **Missing acoustic key in `main.py` dicts:** treat as no preference (0 points), or add `likes_acoustic` to the example profile?
4. **Tie-break:** lower `id` first. Would alphabetical title be preferable?
5. **Case-insensitive matching:** normalize genre/mood strings with `.strip().lower()` before comparing? (The current data is already lowercase.)

---

## Sources

- Spotify, "Understanding recommendations": https://www.spotify.com/uk/safetyandprivacy/understanding-recommendations/
- Spotify Engineering, "Humans + Machines: A Look Behind Spotify's Algotorial Playlists" (2023): https://engineering.atspotify.com/2023/4/humans-machines-a-look-behind-spotifys-algotorial-playlists/
- YouTube Official Blog, "On YouTube's recommendation system": https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/
- Google for Developers, Recommendation Systems course: [Collaborative filtering basics](https://developers.google.com/machine-learning/recommendation/collaborative/basics), [Collaborative filtering summary](https://developers.google.com/machine-learning/recommendation/collaborative/summary), [Content-based filtering summary](https://developers.google.com/machine-learning/recommendation/content-based/summary)
