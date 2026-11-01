# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**VibeFinder 1.0 — Content-Based Music Recommender**

VibeFinder is a small Python program that ranks songs from an 18-song catalog according to a listener's stated preferences.

---

## 2. Intended Use  

### Goal / task

VibeFinder takes a profile with four preferences: a favorite genre, a favorite mood, a target energy level from 0 to 1, and whether the listener likes acoustic-sounding music. It compares that profile with every song in the catalog, gives each song a relevance score, and returns the five highest-scoring songs. Each song comes with an explanation of where its points came from.

The score estimates how well a song fits what the listener *said* they want. It doesn't predict what they will actually play next. It also doesn't learn: there are no likes, skips, or listening history, so the same profile always gets the same list.

### Intended use

VibeFinder is an educational simulation built on a small synthetic dataset for a classroom project. It's meant for:

- learning how content-based recommendation works
- showing how feature weights and ranking rules shape the results
- exploring how changing a listener's preferences changes the recommendations
- studying bias and limitations in a simplified recommender

### Not intended for

- a production music recommendation service
- reliably predicting whether a real person will enjoy a song
- making any important decision about people
- drawing conclusions about the musical preferences of real populations
- representing all music genres and cultures fairly (the catalog has 13 genres with one to three songs each)

---

## 3. How the Model Works  

Every song gets points for how well it matches the listener's profile:

| What's compared | Points |
|---|---|
| Song's genre is the listener's favorite genre | 4 |
| Song's mood is the listener's favorite mood | 3 |
| Song's energy is close to the listener's target energy | up to 3 |
| Song's acoustic sound fits the listener's acoustic preference | up to 1 |
| **Maximum possible score** | **11** |

Genre and mood are all-or-nothing. Before comparing, letter case and extra spaces are ignored, but the words must otherwise match exactly. Energy rewards **closeness, not loudness**. A song exactly at the target energy gets all 3 points, and the farther away it is, the fewer points it gets. For a listener who wants energy 0.40, a song at 0.42 earns 2.94 points, but a much more energetic song at 0.91 earns only 1.47. For acousticness, a listener who likes acoustic music gets more points for more acoustic songs, and a listener who doesn't gets more points for less acoustic (more produced) songs.

The four parts are added up, the songs are sorted from highest to lowest score, and the top five are shown. If two songs have exactly the same score, the one with the lower ID number goes first. There is no randomness, so the same profile always produces the same list.

**This project does not train a machine-learning model. It simulates recommendation behavior using a weighted content-based algorithm.** The weights were chosen by hand in Phase 2, after comparing three options against the catalog.

**Changes from the starter:** the starter's recommendation functions were placeholders, and its runner script used profile keys and an import path that didn't match the rest of the code. This version implements CSV loading with validation, the scoring rule, ranking with a fixed tie-break, and explanations, and it fixes the runner so it uses the real profile fields.

**Optional ranking modes (bonus extension):** besides the baseline described above, two optional modes use the same formula with different weights. *Mood-first* uses genre 2, mood 4, energy up to 3, and acoustic up to 1. *Energy-focused* uses genre 2, mood 3, energy up to 6, and acoustic up to 1. The baseline is still the default, and everything else in this model card (data, strengths, limitations, and evaluation) describes the **baseline** mode only. The alternative modes change what the system prioritizes, but they haven't been shown to give better recommendations (see the README's [Ranking Modes](README.md#optional-extensions--ranking-modes) section).

**Optional diversity-aware reranking (bonus extension):** with `--diverse`, the top five are chosen one at a time. A song loses 1.5 points for each song of the same genre already picked, and 2 points if its artist was already picked. The adjusted score only decides the order; each song's real relevance score is still shown next to it. This can make a list less repetitive. With the baseline weights, it raised the number of different genres from 3 to 4 for Chill Lofi and the Adversarial Listener, and replaced a second LoRoom song in both lists. It didn't change the Intense Rock list at all, and it only reordered High-Energy Pop. The extra variety came at a cost: Chill Lofi lost Focus Flow (relevance 7.78) and gained Paper Boats (3.55), a sad folk song that matches neither the listener's genre nor mood. So the feature didn't clearly *improve* the test profiles; it traded relevance for variety. More variety is also not the same as fairness. The penalties only look at the five songs on screen, not at which genres, artists, or cultures the catalog represents or which listeners are served well. The system still has no feedback loop to show whether it reduces a filter bubble over time, and the 1.5 and 2.0 penalties are untested design choices. The feature is off by default, and everything else in this card describes the system without it.

**Optional advanced scoring (bonus extension):** with `--advanced`, up to 3 extra points are added on top of the chosen mode's score. These points come from five extra song attributes (see §4): 0.5 for closeness to a target popularity, 0.5 for matching a preferred release decade, 0.5 each for closeness in instrumentalness and vocal intensity, and up to 1.0 for the share of the listener's preferred mood tags the song has. The maximum becomes 14 with the baseline weights (13 for mood-first, 15 for energy-focused). The original four-part score isn't changed, just added to. Advanced scoring only runs when it's explicitly requested and every song and preference has the needed data; missing data raises an error instead of being treated as zero. It's an educational demonstration of adding features to a content-based score. In the test profiles, it swapped #2 and #3 for High-Energy Pop and #1 and #2 for Intense Rock. For the Adversarial Listener, it replaced Storm Runner with a 0.14-energy classical song. Because the features and preferences are invented, these changes show the formula's *sensitivity* to new attributes, not better recommendations. The rest of this card describes the basic score.

---

## 4. Data  

The catalog (`data/songs.csv`) has **18 fictional songs**: 10 that came with the starter project and 8 added for this project (generated with AI help) to cover more genres and moods. The songs and artists aren't real releases, and the feature values were assigned for this project, not measured from audio by any streaming service.

Each song has an ID, title, artist, genre, mood, energy, tempo (BPM), valence, danceability, and acousticness.

- **13 genres:** lofi (3 songs), rock (3), pop (2), and one song each for ambient, jazz, synthwave, indie pop, afrobeats, hip-hop, r&b, folk, classical, and reggae
- **9 moods:** happy, chill, intense (3 each); relaxed, moody, sad (2 each); focused, confident, romantic (1 each)
- **Ranges:** energy 0.14–0.93, acousticness 0.05–0.98, tempo 60–152 BPM

The score uses only **genre, mood, energy, and acousticness**. Tempo, valence, and danceability are kept in the file but don't affect the ranking.

**Expanded dataset (bonus extension):** five more columns were added to all 18 songs, without changing any original value:

- `popularity` (whole number, 0–100)
- `release_decade` (1990s–2020s in this catalog)
- `instrumentalness` and `vocal_intensity` (0–1)
- `mood_tags` (one to three tags from a fixed list of nine, such as *nostalgic*, *uplifting*, or *melancholic*)

They're used only by optional advanced scoring. Like everything else in the catalog, they're **synthetic**: the values were chosen with AI help to fit each fictional song's genre, mood, and energy. They are not real popularity figures or release dates. Using real ones would bring its own problems. Popularity rewards music that's already popular, release dates can favor recent music, and both reflect the habits of the platform that measured them. The tags also partly repeat the existing mood labels.

**Limits of the data:**

- It's a very small, synthetic catalog.
- Genres are unevenly represented: ten genres have only one song.
- Genre and mood labels are judgment calls (is a song "chill" or "relaxed"?).
- There is no real listening behavior, and there are no ratings showing whether anyone is actually satisfied with a recommendation.

---

## 5. Strengths

- **It works best when a listener's preferences line up with the catalog.** High-Energy Pop, Chill Lofi, and Intense Rock each get a #1 song that matches all four preferences: Sunrise City (10.76), Library Rain (10.71), and Storm Runner (10.87).
- **It responds to preferences.** Changing the profile changes the list a lot: across the four test profiles, any two top-five lists share at most three songs, and three of the six pairs share none.
- **Every score can be explained.** Each recommendation lists the points it got for genre, mood, energy, and acousticness, and those points add up exactly to the score.
- **It's predictable and testable.** It has no randomness, ties are broken by ID, and invalid profiles or CSV rows are rejected with clear errors instead of being silently guessed.

---

## 6. Limitations and Bias 

The biggest weakness the experiments revealed is that an exact genre match outweighs everything else, even when the catalog can't back it up. The "Adversarial Listener" asked for lofi, sad, high-energy (0.90) music. The three lofi songs, with energy only 0.35–0.42, still took the top three places on the genre bonus alone (5.49–5.85 out of 11), while Storm Runner, an almost perfect energy match, came fifth with 3.87. Uneven data makes this worse: lofi and rock have three songs each, pop has two, ten genres have only one, and no song with acousticness 0.5 or higher has energy above 0.42, so some combinations of preferences can't be satisfied at all. The system still fills all five slots with confident-looking scores instead of saying the catalog has no good match.

More specific evidence:

- **Exact matching:** a pop listener gets 0 genre points for Rooftop Lights because it's labeled *indie pop*, so it ranks below Gym Hero, a pop song with the wrong mood.
- **Correlated features:** in this catalog, energy and acousticness have a correlation of about −0.96, so energetic acoustic songs are missing. That's a property of this small synthetic dataset, not a general rule about music.
- **Possible filter bubble:** 7 of the 11 points come from exact genre and mood matches, which tends to keep recommending familiar categories. A true long-term filter bubble hasn't been shown, though, because the simulation has no repeated feedback loop: it never learns from what a listener chose.

The README's [Limitations and Risks](README.md#limitations-and-risks) section has the full list.

---

## 7. Evaluation  

**Profiles tested.** I tried four made-up listeners (not real people). *High-Energy Pop* (pop, happy, energy 0.80, prefers produced sound) is the default case. *Chill Lofi* (lofi, chill, 0.40, likes acoustic) and *Intense Rock* (rock, intense, 0.90, prefers produced sound) sit at opposite ends of the catalog, which checks that changing preferences really changes the results. The *Adversarial Listener* (lofi, sad, 0.90, prefers produced sound) asks for things that don't exist together in the catalog, which shows which preference the system values most.

**How I tested.** I ran all four profiles through the real program (`python -m src.main --all-profiles`), compared their top-five lists, checked that each explanation adds up to its score, and recalculated several scores by hand from the song data.

**What I saw.** Every profile got five songs, ranked correctly, with explanations that add up to the scores. Sunrise City was #1 for High-Energy Pop (10.76), Library Rain for Chill Lofi (10.71), and Storm Runner for Intense Rock (10.87). The Adversarial Listener got three low-energy lofi songs ahead of the high-energy options it asked for. The lists were very different: the four top-fives share almost nothing, and a song appears in more than one list only when the listeners share a genre, a mood, or an energy level.

- *Pop vs. Lofi:* no shared songs. One wants loud, produced music and the other quiet, acoustic music, and in this catalog those are opposite ends.
- *Pop vs. Rock:* they share Gym Hero and Concrete Crown. Gym Hero gets in for different reasons: it's pop for the pop fan and intense for the rock fan.
- *Pop vs. Adversarial:* no shared songs. Both want high energy, but the adversarial listener's "lofi" bonus pulls quiet songs to the top.
- *Lofi vs. Rock:* no shared songs. They disagree on everything.
- *Lofi vs. Adversarial:* they share all three lofi songs, because they share the genre. The order changes because the adversarial listener wants loud, non-acoustic music, which hurts the quietest, most acoustic lofi song (Library Rain) the most. It drops from #1 to #3.
- *Rock vs. Adversarial:* they share only Storm Runner, which is #1 for rock but only #5 for the adversarial listener because it isn't lofi.

**Surprises.** For the pop fan, Gym Hero (pop, but *intense*, not happy) ranks #2, ahead of two happy songs. Being the right genre is worth 4 points and being the right mood is worth 3, so genre wins. For the rock fan, Gym Hero (a pop song) ties a rock song at exactly 6.86 and is shown first only because of its ID number, not because it fits better.

**Weight experiment.** I changed the weights from 4 / 3 / 3 / 1 (genre / mood / energy / acoustic) to 2 / 3 / 6 / 1, which halves genre and doubles energy. Then I reran the same four profiles with the same code. Some songs moved a lot. For pop, lofi, and rock, the same songs mostly stayed in the top five and only swapped places: happy songs moved above Gym Hero for the pop fan, an ambient song moved above a lofi song for the lofi fan, and only the rock list's fifth song changed. For the adversarial listener, four of the five songs were replaced by loud rock, pop, and hip-hop songs. That list matched the target energy much better (the average gap dropped from 0.434 to 0.050), but none of its songs were lofi or sad, so the genre and mood fit got worse. Cross-genre discovery didn't consistently improve: the number of other-genre songs in the top five changed only for the adversarial profile. The number of different songs recommended across all four lists actually fell from 14 to 13. Neither version was better for every profile, so I kept the original weights.

**What this shows and what it doesn't.** The scoring rule behaves exactly as designed, and its results can be explained to the point. Where a profile's preferences line up with the catalog, it produces sensible lists. The project's 58 automated tests pass, but they only check that the program does what it's supposed to do: correct math, ranking, and error handling. They don't check whether the music is a good fit. I don't know whether real listeners would like these songs: there's no listening history, no feedback, and no "correct answer" to compare against, and the test profiles are ones I invented. Passing tests is not the same as accurate recommendations.

---

## 8. Future Work  

1. **More representative data.** Add a larger, more balanced catalog with more genres, moods, energy levels, and combinations like energetic acoustic music. Right now a folk fan has exactly one folk song to choose from, and the Adversarial Listener's request (high-energy sad lofi) matches nothing well. More candidates would give the scoring rule real options instead of forcing it to fill the list with weak matches.
2. **More flexible preferences and matching.** Let listeners name several favorite genres and moods, and give partial credit for related labels. For example, *indie pop* could earn some genre points for a pop fan. Real user feedback (likes and skips) could later adjust the weights.
3. **Better discovery and relevance controls.** Add a minimum score so that weak matches aren't shown as if they were good ones, rerank the list to keep it from being too repetitive, or occasionally include an exploration pick. That would stop the system from always returning five suggestions even when the fifth one (like Concrete Crown at 3.91 for the pop fan) matches no label at all.

These were written as ideas for a future version. Since then, one part of idea 3 has been added as an optional bonus feature: reranking to reduce repetition (see §3). The minimum-score threshold, exploration picks, and ideas 1–2 are still not implemented.

---

## 9. Personal Reflection  

My main reflection is in the [Reflection](README.md#reflection) section of the README. In short: this project showed me that a ranking can be mathematically correct and still not be a good recommendation, and that the weights I choose decide what the system thinks matters.
