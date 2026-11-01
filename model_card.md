# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---

## 6. Limitations and Bias 

The biggest weakness the experiments revealed is that an exact genre match outweighs everything else, even when the catalog can't back it up. The "Adversarial Listener" asked for lofi, sad, high-energy (0.90) music. The three lofi songs, with energy only 0.35–0.42, still took the top three places on the genre bonus alone (5.49–5.85 out of 11), while Storm Runner, an almost perfect energy match, came fifth with 3.87. Uneven data makes this worse: lofi and rock have three songs each, pop has two, ten genres have only one, and no song with acousticness 0.5 or higher has energy above 0.42, so some combinations of preferences can't be satisfied at all. The system still fills all five slots with confident-looking scores instead of saying the catalog has no good match.

---

## 7. Evaluation  

**Profiles tested.** I tried four made-up listeners (not real people). *High-Energy Pop* (pop, happy, energy 0.80, prefers produced sound) is the default case. *Chill Lofi* (lofi, chill, 0.40, likes acoustic) and *Intense Rock* (rock, intense, 0.90, prefers produced sound) sit at opposite ends of the catalog, which checks that changing preferences really changes the results. The *Adversarial Listener* (lofi, sad, 0.90, prefers produced sound) asks for things that don't exist together in the catalog, which shows which preference the system values most.

**What I saw.** Every profile got five songs, ranked correctly, with explanations that add up to the scores. The lists were very different: the four top-fives share almost nothing, and a song appears in more than one list only when the listeners share a genre, a mood, or an energy level.

- *Pop vs. Lofi:* no shared songs. One wants loud, produced music and the other quiet, acoustic music, and in this catalog those are opposite ends.
- *Pop vs. Rock:* they share Gym Hero and Concrete Crown. Gym Hero gets in for different reasons: it's pop for the pop fan and intense for the rock fan.
- *Pop vs. Adversarial:* no shared songs. Both want high energy, but the adversarial listener's "lofi" bonus pulls quiet songs to the top.
- *Lofi vs. Rock:* no shared songs. They disagree on everything.
- *Lofi vs. Adversarial:* they share all three lofi songs, because they share the genre. The order changes because the adversarial listener wants loud, non-acoustic music, which hurts the quietest, most acoustic lofi song (Library Rain) the most. It drops from #1 to #3.
- *Rock vs. Adversarial:* they share only Storm Runner, which is #1 for rock but only #5 for the adversarial listener because it isn't lofi.

**Surprises.** For the pop fan, Gym Hero (pop, but *intense*, not happy) ranks #2, ahead of two happy songs. Being the right genre is worth 4 points and being the right mood is worth 3, so genre wins. For the rock fan, Gym Hero (a pop song) ties a rock song at exactly 6.86 and is shown first only because of its ID number, not because it fits better.

**Weight experiment.** I halved the genre weight (4 → 2) and doubled the energy weight (3 → 6), then reran the same four profiles with the same code. For pop, lofi, and rock, the same songs mostly stayed in the top five and only swapped places. Happy songs moved above Gym Hero for the pop fan, an ambient song moved above a lofi song for the lofi fan, and only the rock list's fifth song changed. For the adversarial listener, four of the five songs were replaced by loud rock, pop, and hip-hop songs, none of them lofi or sad. So the weights matter most when genre and energy disagree, and moving weight between them trades one wrong answer for another. I kept the original weights.

**What this shows and what it doesn't.** The scoring rule behaves exactly as designed, and its results can be explained to the point. Where a profile's preferences line up with the catalog, it produces sensible lists. These checks only show the math is *consistent*, though. I don't know whether real listeners would like these songs: there's no listening history, no feedback, and no "correct answer" to compare against, and the test profiles are ones I invented. Passing tests is not the same as accurate recommendations.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
