## *["This is the moment I've been training for," said the pun-generating AI](https://paulcalhoun.substack.com/p/this-is-the-moment-ive-been-training)*

### In 2022 Robert Gonsalves [demonstrated](https://towardsdatascience.com/i-once-trained-an-ai-to-rhyme-and-it-took-gpt-j-a-long-time-de1f98925e17) that GPT-J-6B could be fine tuned for limerick generation. This is an interesting data point, historically speaking, for a few reasons:
* GPT-J-6B was over a year old when this happened
* It’s ~50x smaller than GPT3
* Generating coherent and amusing jokes [is considered computationally difficult](https://hdsr.mitpress.mit.edu/pub/wi9yky5c/release/3)
   * Note: Google’s PaLM LLM [already managed this task](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html), albeit at 100x scale
* Robert Gonsalves did this as a fun personal project, using readily available cloud tools

### I’m currently trying to fine tune the same model to make puns. Some unique (I think) output examples so far:
* **Two guys argued about a painting. There was a rupture in the peace.**
   * Peace => Piece (painting)
* **When the townspeople found out the cow was giving birth, it was quite a cow to have to deal with.**
   * I like this one because it’s still a pun, despite not being remotely funny.
* **A musician went to the hospital because he swallowed a trombone. The doctor told him to have a tube inserted and he would be playing on his own soon.**
   * This is a mediocre pun, but the setup requires a large amount of real-world knowledge.
* **Two electricians had such different tastes, they went to a waffle outlet for a discussion.**
   * This one appears to be a double-pun (electricians => outlet, and waffle-food => waffle-to change opinions)
* **“I love kiwis,” said Tom kiwwisely.**
   * They’re not all zingers.
* **To be able to go back to boarding school and pass all her subjects meant that she had learnt her lesson.**
   * So much worldbuilding for such an anticlimactic payoff.
* **The story of a boy who was born with one eye in the wrong place was told from an unexpected angle.**
   * This one is probably the most impressive to date, after ~12000 fine tuning steps (and poring through maybe 800 non-pun or unfunny pun inferences).
* **Old pianists never die they just get tuned away.**
   * This format (“Old [specialist]s never die, they just [death euphemism]”) is featured many times in the training data. However, the above pun is not on Google anywhere, so I assume it’s new.
* **I like to have a fire lit in my chimney, said Tom light-heartedly.**
   * Heart=>Hearth
* **Old gardeners never die they just turn green**
* **He didn't wear his house shoes to work because he's such a homeboy.**
* **Old mathematicians never die, they just have to multiply.**
* **A young man sitting at a table with a pot of stew was very busy keeping a lid on his appetite.**
* **Drumlines are always being beat up.**
* **"There's no shortage of water," said Tom rationally.**
   * Water rations.
* **My new job as a gem cutter is fascinating because I am so deeply engaging.**
   * Gems => engagement rings.

--------------

Note: HF doesn't support running models trained in 8-bit, so in the included example transformers gets patched before the model is loaded directly via torch.

Also, make sure you use the package versions requirements.txt. At least the bnb and transformer versions need to match what's there.
