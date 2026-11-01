# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

I used Claude Code, an AI coding agent that reads files, edits code, and runs commands in my project, for Phases 4–5 and all four bonus challenges. The main example for this section is **Bonus Challenge 1**: adding five song attributes to the dataset and an optional advanced score that uses them. The agent had to change the CSV, the loader, the scoring, the CLI, the tests, and the docs consistently, without breaking anything built earlier.

### A. Prompt examples

These are the actual prompts from this project's agent session, quoted in short excerpts. Prompts for Phases 1–3 came from earlier sessions that aren't recorded here.

- **Phase 4:** *"Your task is to evaluate the existing system using diverse user profiles, investigate unexpected recommendations, conduct one controlled scoring experiment, identify limitations and biases, and document the findings in `README.md` and `model_card.md`."*
- **Phase 4 follow-up:** *"I would add two or three sentences to the README's Experiments section. […] do just this in readme"*
- **Phase 5:** *"Your task is to complete Phase 5: Reflection and Model Card. Focus on accurate documentation, thoughtful reflection, and final project quality. Do not redesign the recommender or make unnecessary implementation changes."*
- **Bonus Challenges 2–4:** quoted in their own sections below.
- **Bonus Challenge 1 (this extension):** *"Add these columns to `data/songs.csv`: `popularity`, `release_decade`, `instrumentalness`, `vocal_intensity`, `mood_tags`. Populate all 18 songs. […] Introduce an optional advanced-feature scoring switch, such as `advanced=True`, that works with the existing ranking modes. Retain the original four-feature scoring formula when advanced mode is disabled. […] Do not alter fabricated metadata solely to force a desired ranking."*

### B. AI-generated changes (Bonus Challenge 1)

- **CSV expansion:** five columns appended to all 18 rows of `data/songs.csv`. A script appended them as text, so the original ten values couldn't be reformatted, then checked that those values were unchanged per song. The new values were chosen from each song's existing genre, mood, and energy, *before* any advanced ranking was run.
- **Validation:** `load_songs` treats the five columns as all-or-none. A file with the original ten columns still loads, but a file with only some advanced columns is rejected. It checks:
  - popularity is a whole number from 0 to 100
  - the decade is a multiple of 10 from 1900 to 2020
  - instrumentalness and vocal intensity are between 0 and 1
  - mood tags come from a fixed list of nine, with no blanks or repeats
- **Advanced scoring:** optional fields on `Song` and `UserProfile` that default to `None`, and `_advanced_points()`, which adds the five contributions (max 3.0) on top of the existing score. An `advanced=False` switch was added to `score_song`, `recommend_songs`, `recommend_diverse_songs`, and `Recommender`. There's also `max_possible_score(mode, advanced)`.
- **CLI:** the `--advanced` flag, example advanced preferences for the four demo profiles (stored separately from the basic profiles), and a printed line explaining where the extra points come from.
- **Tests:** `tests/test_advanced.py` adds 56 tests. One existing assertion was changed; see section E.
- **Docs:** the README "Advanced Features" section, and new paragraphs in model card §3 and §4.

### C. Design decisions

- **Synthetic metadata.** The songs are fictional, so real popularity or release data doesn't exist for them. Inventing values and *labeling them as invented* is the honest option. It also keeps the project free of any service's data.
- **Five extra features.** The rubric asks for at least five. Each one measures something different: how popular a song is, its era, instrumental vs. vocal sound, how strong the vocals are, and more detailed moods.
- **Optional advanced scoring.** The basic 4/3/3/1 score, its tests, and every documented result stay exactly the same unless `--advanced` is used.
- **Backward compatibility.** New dataclass fields have defaults, every new argument is optional, and the old CSV format still loads. Missing advanced data **raises an error** instead of being silently treated as zero.
- **Human-readable explanations.** Each advanced contribution adds its own reason, such as `Mood tags 2/2 matched: uplifting, euphoric (+1.00)`, so a score can still be checked by hand.

### D. Verification the agent ran (actual commands and results)

| Command | Result |
|---|---|
| `.venv/bin/pytest -q` | `186 passed` |
| `.venv/bin/python -m pytest -q` | `186 passed` |
| `python -m src.main` compared with the original Phase 3 output | identical |
| 10 earlier flag combinations (`--all-profiles`, `--mode …`, `--diverse …`) compared with snapshots taken before this change | all identical |
| `python -m src.main --table` and `--all-profiles --diverse --table` compared with snapshots | identical |
| `python -m experiments.weight_experiment` compared with its snapshot | identical |
| `--advanced`, `--all-profiles --advanced`, `--mode mood-first --advanced`, `--advanced --diverse --table`, and three more combinations | all ran; 5 or 20 "Mood tags" reasons, as expected |
| `git diff --check` | clean |
| Script comparing the original and expanded CSV | 18 songs, original ten columns unchanged, exactly five new columns |

**What did you verify or fix manually?** See *My manual verification notes* at the end of this section.

### E. AI collaboration: accepted, questioned, and revised

**Accepted as specified in my prompts:**

- the 4/3/3/1, 2/4/3/1, and 2/3/6/1 weights
- the 1.5 and 2.0 diversity penalties
- the five advanced formulas and their 3.0-point maximum

**Raised by the agent for me to decide:**

- **Phase 5:** `requirements.txt` lists `pandas` and `streamlit`, but no code uses them. The agent documented this in the README and left the file unchanged.
- **Challenges 2–4:** each one was built on top of earlier, uncommitted work. The agent pointed this out and suggested one combined commit.
- **This challenge:** adding columns broke the exact 10-key dictionary check in `test_load_songs_converts_types`. Instead of deleting the check, the agent narrowed it to the ten original columns (whose values are unchanged) and added a check for the new keys. It's the only earlier test that was edited.

**Mistakes the agent caught and fixed in its own work:**

- **Phase 4:**
  - The first README output blocks were in the wrong order and left out the `=====` banners, so they weren't exact copies of the output. They were regenerated from real CLI output and checked programmatically.
  - The model card limitations paragraph was 7 sentences, but 3–5 were requested. It was cut to 5.
  - A docstring said `patch.object`, but the code used `patch.multiple`. The docstring was corrected.
- **Phase 5:** model card wording like "I implemented…" was made neutral, because the agent wrote that code.
- **Challenge 3:** the penalty defaults were fixed when the function was defined, so changing the constants later would have had no effect. They were changed to be read at call time.
- **Challenge 4:** two of the agent's first tests were wrong, and the tests were fixed, not the code. Awkwardly wrapped headers led to a wider score column.
- **This challenge:**
  - **Analysis script:** the agent's first before/after script forgot `advanced=True` and wrongly reported "no change." The comparison was rerun correctly.
  - **README wording:** the first README draft said the Chill Lofi order held because "gaps are too large." Library Rain actually leads by only 0.06; the real reason is that all five songs gain similar advanced points. The wording was corrected.
  - **Test patterns:** three tests expected error messages without the `mood_tags:` prefix. The test patterns were fixed.

**Revised by me** (visible in the project history):

- **Phase 4 opinion:** I added my own three-sentence opinion to the README's Interesting Results.
- **Phase 4 and 5 commits:** I committed them with my own messages (`test: evaluate recommendation profiles and weight sensitivity`, `docs: complete recommender model card and reflection`) instead of the suggested ones.

### My manual verification notes

> ✍️ *To be completed by me. Everything above was produced or run by the AI agent; nothing here has been confirmed by me yet.*

- [ ] I read the five new columns in `data/songs.csv` and the values seem reasonable for each fictional song:
- [ ] I ran `python -m src.main --advanced` and checked one song's five advanced points by hand:
- [ ] I ran `pytest` myself and saw the result:
- [ ] My opinion on whether the advanced features made the recommendations better, worse, or just different:

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

**Strategy pattern**, used for Bonus Challenge 2 (multiple ranking modes). Each ranking mode is an interchangeable strategy that supplies weights to one shared scoring formula.

**Example AI prompt used**

I gave Claude Code (an AI coding agent) a written task description. Key excerpts:

> Introduce at least three user-selectable recommendation modes with genuinely different scoring priorities, using a simple modular Strategy design pattern.
>
> Design a small, maintainable Strategy implementation, such as an immutable weights configuration and a registry of named scoring strategies. Preserve one shared scoring formula. Do not copy and paste scoring logic for different modes. […] Ensure the default mode still uses the module-level constants so the existing `experiments/weight_experiment.py` patching mechanism continues to work.

The prompt specified the three modes and their weights: baseline 4/3/3/1, mood-first 2/4/3/1, and energy-focused 2/3/6/1.

**How did AI help you brainstorm or implement it?**

The agent read the existing recommender, CLI, tests, and Phase 4 experiment, then wrote the implementation:

- `ScoringWeights`: a frozen dataclass holding the four weights, with a `max_score` property
- `ScoringStrategy`: a frozen dataclass holding a mode's name, description, and how to get its weights
- `STRATEGIES`: a registry of the three modes, and `get_strategy(mode)`, which raises a clear error for unknown modes
- an optional `mode` argument on `score_song`, `recommend_songs`, and `Recommender(songs, mode=...)`, defaulting to `"baseline"`
- a `--mode` CLI option using argparse `choices`
- 28 new tests in `tests/test_modes.py` (86 tests in total)

It also ran every profile in every mode and wrote the comparison in the README's "Optional Extensions — Ranking Modes" section.

**AI design suggestions accepted or revised**

These ideas came from my prompt and were used as written:

- immutable weights plus a named registry
- one shared formula
- an optional `mode` argument so existing calls keep working
- baseline as the default
- argparse `choices`

These are design decisions the agent made itself. They're listed here for me to review:

- **The baseline reads the module constants each time it's used**, instead of storing a frozen copy of 4/3/3/1. A frozen copy would have stopped the Phase 4 experiment's patching from having any effect.
- **It decided against one subclass per mode with its own `score` method.** Since the modes differ only in weights, that would have meant copying the formula three times.
- **Modes are selected by string name** (`"mood-first"`), not by passing strategy objects, so the same names work in Python and as CLI choices.
- **The `Ranking mode: ...` line is printed only for non-default modes.** As a result, `python -m src.main` and `--mode baseline` give exactly the original output.
- **`energy-focused` reuses the Phase 4 experiment's 2/3/6/1 weights**, and a test confirms both produce identical rankings.

**How does the pattern appear in your final code?**

- `src/recommender.py`: `ScoringWeights`, `ScoringStrategy`, `STRATEGIES`, and `get_strategy()`. `score_song()` calls `get_strategy(mode).weights()` and applies the same formula for every mode.
- `src/main.py`: the `--mode` option and `describe_mode()`.
- `tests/test_modes.py`: tests for the modes.

**My manual verification notes**

> ✍️ *To be completed by me. The AI agent ran the automated checks listed in the README; nothing below has been confirmed by me yet.*

- [ ] I ran `python -m src.main` and confirmed the default output is unchanged:
- [ ] I tried `--mode mood-first` and `--mode energy-focused` and reviewed the rankings:
- [ ] I read through `ScoringWeights` / `STRATEGIES` / `score_song` and understand how a mode is selected:
- [ ] My own notes on whether the mode rankings make musical sense:

---

## Diversity-Aware Ranking (Bonus Challenge 3)

**Example AI prompt used**

I gave Claude Code (an AI coding agent) a written task description. Key excerpts:

> Implement an optional diversity-aware reranking mechanism that reduces repeated genres and artists in the top-K recommendations. It must work with all existing scoring modes and preserve the original recommendation behavior when diversity is disabled.
>
> Adjusted score = original relevance score − genre repetition penalty − artist repetition penalty. […] Genre repetition penalty: 1.5 points for every previously selected song from the same genre. Artist repetition penalty: 2.0 points if the artist has already appeared in the selected recommendations. […] Never present an adjusted score as though it were the unmodified relevance score.

The prompt specified the greedy approach, the formula, the 1.5 and 2.0 penalties, the `--diverse` flag, and the list of tests.

**What the AI contributed**

The agent checked the catalog and found that two artists repeat (LoRoom and Neon Echo), so the artist penalty could be shown with real data instead of only synthetic test songs. It then implemented the feature, ran it on all four profiles in every mode, and wrote the README and model card sections from the actual output.

**Implementation summary**

- `src/recommender.py`:
  - `GENRE_REPEAT_PENALTY = 1.5` and `ARTIST_REPEAT_PENALTY = 2.0`
  - a frozen `DiversePick` record that keeps the base relevance score, each penalty, and the adjusted score separate, with a `penalty_explanation()` method
  - `diversity_rerank(scored, k)`: the greedy reranker
  - `recommend_diverse_songs(user_prefs, songs, k, mode)`
  - The scoring loop moved into a small shared helper (`_score_all`), so `recommend_songs` and the reranker use the same scores.
- `src/main.py`: the `--diverse` flag. Diverse output labels "Relevance score" and "Adjusted for diversity" separately and adds a "Diversity:" line explaining any penalty.
- `tests/test_diversity.py`: 24 new tests (110 in total).

**Decisions the AI made for me to review**

- The penalty defaults are read **when the function is called**, not when it's defined, so changing or patching the constants takes effect.
- The reranker returns new `DiversePick` records instead of reusing the `(song, score, explanation)` tuples. That way an adjusted score can't be mistaken for a relevance score.
- Diversity was added as a separate function and CLI flag, not as a new ranking mode, so it can combine with any mode.
- `recommend_songs` and the `Recommender` class are unchanged. The class has no diversity option, because the challenge didn't require one.
- It noted, but didn't change, the trade-off that Chill Lofi loses Focus Flow (relevance 7.78) and gains Paper Boats (3.55).

**My manual verification notes**

> ✍️ *To be completed by me. The AI agent ran the automated checks; nothing below has been confirmed by me yet.*

- [ ] I ran `python -m src.main --all-profiles --diverse` and compared it with the default output:
- [ ] I checked one penalty by hand (e.g. Focus Flow: 7.78 − 3.00 − 2.00 = 2.78):
- [ ] My opinion on whether the diverse lists are better, worse, or just different:

---

## Readable Terminal Table (Bonus Challenge 4)

**Example AI prompt used**

I gave Claude Code (an AI coding agent) a written task description. Key excerpts:

> Implement a well-formatted ASCII table showing ranked songs, scores, and the exact reasons for each recommendation. Use only the Python standard library if practical. […] For diversity mode, distinguish the base relevance score and adjusted ranking score. Support wrapping long explanation text so the output remains readable. Avoid clipping away important scoring information. […] Separate formatting logic from recommendation calculations.

**What was generated**

- `src/table.py` (new, standard library only via `textwrap`):
  - `wrap_cell`: wraps text on spaces, indents continuation lines, and splits words too long for a column instead of clipping them
  - `render_table`: draws fixed-width borders and allows multi-line cells
  - `recommendations_table` and `diverse_table`: turn engine results into table rows
- `src/main.py`: the `--table` flag. When it's set, the same results that would be printed as a list are printed as a table.
- `tests/test_table.py`: 20 new tests (130 in total). Most of them parse the rendered table back into columns to check titles, scores, and every reason.
- The README "Optional Extensions — Table Output" section. Its output blocks were generated from the real CLI and checked to match it exactly.

**Design decisions the AI made for me to review**

- **One reason per line**, split from `score_song`'s own explanation string, so the table can't reword or drop a reason.
- **Fixed widths fitting a 100-column terminal**, instead of detecting the terminal size. This keeps output identical across machines and tests. The long "Acousticness …" reason always wraps to a second line.
- **The score header shows the mode's maximum** ("Score (max 11)", "(max 10)", "(max 12)"), so scores from different modes aren't read as the same scale.
- **In diverse mode, the score cell shows relevance on one line and "-> adjusted" on the next.** The full penalty explanation is kept as one wrapped `Diversity:` entry, never as a plain "Score".
- **The default output and every non-table output are unchanged.** All ten earlier flag combinations were compared with saved snapshots.
- Before the fixes, two of the agent's first tests were wrong (counting header rules, and reading wrapped text with borders included). The tests were fixed, not the code.

**My manual verification notes**

> ✍️ *To be completed by me. The AI agent ran the automated checks; nothing below has been confirmed by me yet.*

- [ ] I ran `python -m src.main --table` and `--all-profiles --diverse --table` in my own terminal and the table fits and reads well:
- [ ] I compared one row's reasons and score with the plain (non-table) output:
- [ ] My opinion on whether the table is easier to read than the list:
