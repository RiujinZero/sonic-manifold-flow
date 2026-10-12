# Design implementation progress

This ledger maps repository evidence to the Qwen testbed
[`design-v0.3.md`](design-v0.3.md), the successor
[`ManifoldFlow_Sonic_Language_World_Model_Design_v0.4.md`](ManifoldFlow_Sonic_Language_World_Model_Design_v0.4.md)
(which replaces v0.3's narrower Phase 7 "Sonic portability" with training
a ~100M-parameter Sonic *with* ManifoldFlow as native state from the
first optimizer step - a much larger scope, still unstarted), and
preserves the completed and pending image research from
[`design-v0.2.md`](design-v0.2.md). Update it whenever a milestone
changes the status of a design requirement or phase exit condition.

Status meanings:

- **Complete:** implemented and directly verified at the intended current scale.
- **Core complete:** correctness implementation and focused tests exist; the
  required real experiment or systems gate is still pending.
- **In progress:** active milestone with partial evidence.
- **Not started:** no implementation evidence yet.
- **Deferred:** explicitly outside the current MVP entry path.

## Current position

The digit-target half of the generalization investigation sharpens (and
partially revises) the noun result's headline. Frozen digit decode -
unlike frozen noun decode - generalizes substantially: 49.0% held-out
accuracy (real retrieval, 10k steps, converged 85-95% training-vocab
band) versus a ~8.3% chance level, retaining 52% of its own training
ceiling with Sonic never unfrozen. Staged unfreeze still helps further
(oracle retrieval, 94.0% held-out, retaining 94% of its 100%
training-vocab ceiling) but is no longer the dividing line between
"generalizes at all" and "doesn't" the way it was for nouns (0%->51%).
The likely real variable: whether the decode target's output space is
small and already fully covered regardless of split (digits, 1-12) vs.
large/closed and requiring genuinely novel output tokens at eval time
(nouns like "hamster," never produced as output during training) - the
former generalizes fine frozen, the latter needs Sonic itself to adapt.
Practical implication for the production recipe: staged unfreeze is the
safer default for large/open output vocabularies, but frozen training's
generalization gap for small bounded outputs may be acceptable
depending on tolerance - not a categorical "frozen always memorizes"
rule as the noun-only result first suggested
(`results/sonic-digit-decode-generalization.md`).

The generalization test reframes everything else this session found:
frozen training's high in-distribution accuracy was hiding complete
memorization. Two noun-decode runs (`--mode decode`, oracle retrieval),
both genuinely converged before a final held-out pass on
`EVAL_NAMES`/`EVAL_NOUNS` vocabulary never seen during training - frozen
(6000 steps, stable 70-82% band for its last 1400 steps) scored **0/100
(0.0%)** on held-out episodes despite 73.3% on the training vocabulary;
staged unfreeze (perfect 100.0% for its last 1400 steps) scored **51/100
(51.0%)** held-out - a real drop from its own ceiling, but ~8-9x above
the ~5.9% chance level, genuine partial transfer. Frozen training
produced a lookup table over the closed training vocabulary with zero
generalization; unfreezing produced something that actually works on
novel names and nouns, even though neither this session's four
mechanistic ablations nor anything else has explained *why*. Per the
user's own framing, that mechanism may matter less than this result:
a production recipe built on "frozen reaches high accuracy" alone would,
on this evidence, likely fail completely at real inference on unseen
entities. Next: does this held-out gap hold for the digit target and for
real (non-oracle) retrieval, not yet tested
(`results/sonic-noun-decode-generalization.md`).

The digit-decode-under-unfreeze problem is confirmed digit-specific, not
a general property of real retrieval - the genuine apples-to-apples test
against the digit investigation's own `--mode retrieval` (new `--mode
retrieval-noun`, same recency bonus + accuracy-gated unfreeze recipe,
real non-oracle `bank.read()`). Frozen reaches 68.3% by step 3000;
staged unfreeze fires at step 700 and climbs to a sustained, perfect
**100%** by step 2500 - decode roughly octuples within 500 post-unfreeze
steps (11.7%->90.0%), the same clean step-function signature every
successful unfreeze in this project has shown. This lands in essentially
the same place as the oracle-retrieval noun result (100%) and nothing
like the real-retrieval digit result (flat ~9%, never breaks out even at
10k steps) - despite sharing the identical "real, occasionally-wrong
bank.read()" mechanism the digit investigation spent four ablations
blaming without ever pinning down. Digits remain the one hard case; noun
decode works whether retrieval is oracle or real
(`results/sonic-noun-retrieval-decode.md`). Next, per the user's own
framing (the underlying cause may not matter if the practical recipe is
known): does this accuracy reflect genuine generalization or training-
vocabulary memorization - testing directly via the new
`--held-out-eval-episodes` mechanism, not yet run.

Target type turns out to matter enormously for both frozen ceiling and
unfreeze behavior - the first result from a new probe
(`sonic_noun_target_probe.py`, forked from the reverse-query probe this
session) built specifically to test this. Decoding a noun ("guitars")
instead of a digit or a name, under the same oracle-fed `--mode decode`
setup the name-decode comparison used: frozen-only reaches 76.7% by
step 4000 (already far above name's 11.7% at the same budget, and
comparable to digit's own 10k-step ceiling reached in a quarter of the
steps), and staged unfreeze produces the same clean step-function jump
every successful unfreeze in this project has shown - 73.3% ->86.7% at
the unfreeze step, climbing to a **sustained, perfect 100%** from step
2300 onward (17 consecutive checkpoints, 1700 steps). Noun decode
behaves like name decode (unfreeze helps, cleanly), not like digit
decode in retrieval mode (unfreeze doesn't help) - directly confirming
target type, not "decode under unfreezing" in general, is what
separates this session's failures from its successes. Important
caveat: this used oracle retrieval, not the real `bank.read()` that's
where every digit-decode failure actually happened - an earlier
retrieval-mode attempt at this comparison was killed after discovering
its accuracy metric still measured quantity decode, not noun decode;
the real-retrieval version of this comparison remains unbuilt
(`results/sonic-noun-target-decode.md`).

Extending the population-fixed staged-unfreeze run to 10,000 steps (5x
the step budget) resolves the ambiguity the 3000-step version left open:
the plateau was real, not "still early." Post-unfreeze value accuracy
averaged 9.3% across all 91 post-unfreeze checkpoints (steps
1000-10000), never exceeding 16.7% - a flat line, not a slow climb that
ran out of budget, even while retrieval accuracy saturated at 95-100%.
Compare against the frozen-only 10k run reaching 81.7% (peak 95.0%) over
the identical step count. **Practical recommendation for retrieval mode
specifically, now well-established across four staged-unfreeze attempts
this session (three declining on the biased population, one flat at
both 3000 and 10,000 steps on the corrected population): keep Sonic
frozen, don't stage an unfreeze.** This is the opposite of what worked
cleanly in the correction probe, the batch-lane probe, and this same
file's own decode mode - the mechanistic reason for that split remains
unexplained after three refuted hypotheses (wrong retrieved content,
key_projection still training, retrieval-loss gradient competition), but
the practical question (which recipe to use) is answered regardless of
the missing mechanism. One honest caveat: the frozen-only 10k comparison
used the population *before* the bias fix, so it isn't a perfectly
population-matched comparison - the practical conclusion doesn't depend
on closing that gap, but it remains open
(`results/sonic-reverse-query-unfreeze-population-bias-10k.md`).

The population-bias fix breaks the three-time decline pattern, but only
partially resolves the investigation. An investigative agent's
structural comparison against the correction probe's own successful
unfreeze found that `_build_episode`'s reverse-answer-uniqueness
rejection was being applied even in `--mode retrieval`, which never
trains on the reverse answer at all - silently biasing that mode's
training/eval population (though an empirical check found the bias
smaller than theorized: ~28% rejection rate, <1% mean-statement-count
shift). Made conditional on `args.mode`; retrieval mode now trains and
evaluates on the full, unfiltered population. Re-running the identical
staged-unfreeze recipe that failed three times: for the first time,
value accuracy does not decline post-unfreeze relative to its own
frozen baseline (~7.1% pre-unfreeze avg -> ~10.2% post-unfreeze avg,
steps 100-3000) - but the honest read is a flat, noisy plateau, not a
climb (step 1000 and step 3000 both sit at 8.3%, no visible trend across
2000 post-unfreeze steps). Given the 10k-step frozen run's own precedent
(a slow, low, noisy band through step 4000 that only breaks into an
80-95% ceiling around step 5000-6000), this plateau is consistent with
"still early" rather than "stalled" - genuinely undetermined without
extending this run's budget to match. This closes out the session's
"why does unfreezing hurt retrieval-mode decode" investigation with a
partial answer (population bias was real and worth fixing, and removes
the active harm) rather than a full one (still unclear whether
unfreezing helps, is neutral, or simply hasn't had enough steps to show
either way) (`results/sonic-reverse-query-unfreeze-population-bias.md`).

A dedicated 10,000-step frozen-only run (no unfreeze at all, recency
bonus on) settles how far retrieval-mode decode goes if just given
enough time: it reaches an 80-95% sustained band from step ~5000-6000
onward, ending at 81.7% (peak 95.0%) - directly matching the correction
probe's own best frozen-phase result (83.3%), reached by a completely
different route (5x the steps, no unfreeze, real non-oracle retrieval
the whole way). The earlier accidental frozen runs (2000 steps -> 25.0%,
3000 steps -> 50.0%, still rising) undersold how much headroom was
actually there - decode was nowhere near done at either checkpoint
(`results/sonic-reverse-query-recency-10k.md`).

Three consecutive, targeted hypotheses for why unfreezing Sonic hurts
retrieval-mode decode have all been refuted. Freezing `key_projection`
at the unfreeze moment: same decline. Corrupting 20% of decode mode's
oracle content to mimic real retrieval's failure mode: decode mode's
unfreeze win survived intact on both clean and corrupted subsets - rules
out wrong content generically. Zeroing the retrieval loss's weight the
moment Sonic unfreezes (removing its gradient pull on Sonic's shared
hidden states entirely): still the same decline, if anything slightly
worse (~14.3%->~7.2% average across 7 post-unfreeze checkpoints). None
of the concrete, testable theories about retrieval mode's *training
dynamics* explain it. The question that remains genuinely open: what's
actually different between this probe's failing retrieval-mode unfreeze
and the correction probe's own successful one, given both do real
(non-oracle) digit decode from real `bank.read()` retrieval at a
comparable ~80%+ accuracy before staging - a careful side-by-side code
comparison between the two probes' training loops is the next step,
not yet done (`results/sonic-reverse-query-gradient-competition-ablation.md`).

The corruption ablation refutes the leading theory for why unfreezing
Sonic hurts retrieval-mode decode: it isn't wrong content. Decode mode's
clean staged-unfreeze win (8.3%->50% at the original scale) was
re-tested with 20% of episodes fed a random *wrong* statement instead of
the oracle-correct one (`--corruption-rate 0.2`, mimicking real
retrieval's actual failure mode) - the win survived intact on both
subsets: clean-episode accuracy jumped from ~7% to a 40-53% band at
exactly step 2000, and corrupted-episode accuracy climbed right
alongside it (often higher, up to 78.6%). Since wrong-but-plausible
content doesn't break the mechanism, retrieval mode's decline must come
from something else about training on the real retrieval task - the
leading remaining candidate is gradient competition: retrieval mode's
loss includes an InfoNCE retrieval term computed from Sonic's own hidden
states, which still pulls on Sonic's shared representations once
unfrozen even with `key_projection` itself frozen (the earlier failed
fix), pursuing a decode-irrelevant separability objective decode mode
never has at all. Next ablation, not yet run: staged unfreeze in
retrieval mode with `--retrieval-loss-weight 0` post-unfreeze, isolating
whether removing that competing objective is what actually rescues
decode (`results/sonic-reverse-query-corruption-ablation.md`).

The freeze-retrieval-on-unfreeze hypothesis is refuted, not just
untested: re-run with the gate corrected to fire reliably (threshold
0.75, same one that already worked) plus `--freeze-retrieval-on-unfreeze`
enabled produced the same decline as the original no-freeze run, almost
to the point - value accuracy averaged ~14.3% pre-unfreeze and ~10.0%
post (6 checkpoints, steps 1100-1600, no recovery), against the earlier
no-freeze run's ~14.8%->~9.8%. Freezing `key_projection` at the unfreeze
moment left retrieval's own post-unfreeze quality fine (touched 95.0%
twice) but did nothing for decode - ruling out "retrieval still moving"
as the mechanism behind the decline. Two independent, comparable tests
now agree unfreezing Sonic costs retrieval-mode decode ~4-5 points on
average; the one run that never unfroze at all (gate too strict, stayed
frozen 3000 steps) remains this line's best result by a wide margin,
climbing to a 20-50% band and peaking 50.0% at step 2900. Next: a clean
10,000-step frozen-only run (no unfreeze at all) to see how far that
climb goes, launched now
(`results/sonic-reverse-query-retrieval-freeze-corrected.md`).

Two follow-ups to the recency-bonus fix produced a genuinely surprising
result: unfreezing Sonic in retrieval mode has not helped decode once so
far, and simply extending the frozen phase has. A gated unfreeze at
threshold 0.75 (retrieval left trainable) fired cleanly at step 1000 and
retrieval climbed further post-unfreeze (78-95% band), but forward value
accuracy declined (~14.8% pre-unfreeze avg -> ~9.8% post, killed at step
1700, no recovery). A second run testing the user's freeze-retrieval-
on-unfreeze hypothesis (freeze `key_projection` the moment Sonic
unfreezes) used too strict a gate (0.85) and never actually unfroze
across the full 3000 steps - `actual_unfreeze_step: null` - so it
accidentally became "recency-bonus retrieval mode, frozen, extended to
3000 steps" instead. That run produced this line's best decode result by
a wide margin: value accuracy climbed to a 20-50% band from step 1700
onward, peaking 50.0% at step 2900 (final 31.7%), roughly double the
original 2000-step recency-only run's 25% and still not obviously
plateaued. The freeze-on-unfreeze hypothesis remains untested, not
refuted - next run needs a threshold that reliably fires (0.75-0.80)
now that it's clear decode may not need unfreezing to keep improving at
all (`results/sonic-reverse-query-retrieval-freeze.md`).

The reverse-query probe's forward-retrieval ceiling was diagnosable, not
a new limitation: it was the exact ~60% plateau
`sonic_correction_memory_probe.py` also saw before adding a write-order
recency bonus to retrieval re-ranking, a fix this newer probe simply
never inherited. Porting the identical bonus
(`--recency-weight 0.3 --recency-decay 0.7`, same values already
validated there) broke the ceiling the same way here: forward retrieval
53.3%->81.7%, in the same climb-then-sustained-band shape the correction
probe showed. Forward value (decode) accuracy nearly doubled as a direct
result (13.3%->25.0%) and, unlike every earlier frozen run of this
probe, was still climbing rather than flat at the final checkpoint - the
same "retrieval saturates first, decode lags a few hundred steps, then
takes off" two-phase shape `sonic-correction-quantity-pooling-fix.md`
already documented, just not yet finished within this run's 2000-step
budget. Reverse (zero-shot) retrieval improved only modestly and noisily
(56.7%->61.7%) despite the identical re-ranking applied to both reads -
unexplained. Next: combine this with the accuracy-gated unfreeze (this
is the first retrieval-mode run where that gate's precondition - a
retrieval signal reliable enough to be worth leaning on - might actually
hold), not yet run (`results/sonic-reverse-query-recency-bonus.md`).

Staged unfreeze reproduces a third time on the reverse-query probe's
decode mode - name accuracy jumps from a frozen 0-8.3% band straight to
50.0% at the exact step the unfreeze fires (step 2000 of 4000), holding
46.7-58.3% through the end, the same immediate step-function signature
as the correction and batch-lane probes' wins. But it does not
generalize as a blanket fix: retrieval mode's own forward decode (value
accuracy) got slightly *worse* under the identical recipe (12.9%->8.25%
pre/post-unfreeze average), because that run's forward retrieval never
left a noisy 50-65% ceiling either before or after unfreezing - decode
mode's win came from Sonic having an oracle-perfect signal to lean on
throughout, a precondition retrieval mode's own joint training never
reached. One genuine surprise: unfreezing lifted zero-shot reverse
retrieval by ~8 points on average (58.2%->66.3%, peaking at 73.3%) even
though forward retrieval itself barely moved - an observed, unexplained
effect, not a noise artifact given its consistency across ~20
post-unfreeze checkpoints (`results/sonic-reverse-query-staged-unfreeze.md`).

Reverse-query retrieval generalizes from forward-only training with zero
reverse-specific gradient steps: the same `key_projection`, trained only
on "How many guitars does Kevin have?"-style forward questions, resolves
"Who has 8 guitars?"-style reverse questions to the correct basin at a
rate (56.7% final) that tracks forward retrieval (53.3% final) step for
step from the first eval checkpoint - direct support for the reverse-
query design the user proposed to avoid needing the target
deterministically placed. Both decode paths (forward digit accuracy in
retrieval mode, reverse name accuracy in decode mode) stayed near chance
after 2000 steps, but neither run's frozen retrieval had reached the
stable, high band the other two staged-unfreeze results converged from
before unfreezing - consistent with the user's own read that decode is
waiting on a reliable retrieval signal to lean on, not failing
independently. Staged unfreeze, once retrieval itself plateaus, is the
natural next run, not yet tried on this probe
(`results/sonic-reverse-query-probe.md`).

A more mature base Sonic (40.8B pretraining tokens vs the project's
long-standing 33.7B-token default, same architecture, pulled from the
live production run's Drive checkpoints) shifts the staged-unfreeze
batch-lane numbers modestly upward - current-episode 87.6%->90.7%,
prior-episode 71.0%->74.2% - but with a real and initially
counter-intuitive twist: the *frozen*-phase adapter convergence is
dramatically slower with the more-trained checkpoint (18.4%/14.0%
current/prior at step 1200 vs the 33.7B run's 76.7%/62.0% at the same
step, same seeds), even though both checkpoints reach a similar high
band within 200 steps of unfreezing and the more-trained one ends up
slightly ahead. The memory mechanism's ceiling responds to continued
Sonic pretraining, but not in a simple monotonic way across the frozen
and unfrozen phases of this training recipe - one comparison, not yet
replicated at intermediate checkpoints or repeated seeds
(`results/sonic-batch-lane-40B-checkpoint.md`).

Staged unfreezing transfers from the single-episode correction task to
the harder cross-optimizer-step persistence property the batch-lane
probe line exists to measure: prior-episode (genuine cross-step) value
accuracy rises from the frozen-only run's ~58% to **71.0%** with the same
`--unfreeze-sonic-at-step` design (unfreeze at step 2000 of 4000, past
the frozen phase's own convergence point), a real, checkpoint-consistent
~11-13-point gain, not a single lucky checkpoint - the jump to 77.5% is
visible within 200 steps of unfreezing and every post-unfreeze checkpoint
sits at 60.5% or higher, where the pre-unfreeze band regularly dipped to
48%. Current-episode accuracy stayed flat (87.6% vs 91.1%, within this
metric's usual noise). Smaller in absolute terms than the correction
probe's 83.3%->91.7% gain, consistent with cross-step persistence being a
genuinely harder problem - staging lets Sonic's decode readout adapt to
what the adapter retrieves, but can't manufacture information a bank
lookup didn't actually surface (`results/sonic-batch-lane-staged-unfreeze.md`).

Staging the unfreeze - the user's own next suggestion after the
immediate-unfreeze collapse ("try the staged approach, if that fails too
we may need to train them separate") - resolves the collapse entirely
and beats the frozen ceiling by a wide margin: **91.7% value accuracy**,
up from the frozen run's 83.3%, with decode loss an order of magnitude
lower (0.068 vs never below ~0.28 frozen). Letting the port/key/value
adapter converge to a real, working state with Sonic frozen first (past
the canonical run's own convergence point, step 2100 of 4000 total),
*then* unfreezing, produces an immediate, dramatic, and sustained
improvement - value accuracy jumps from 76.7% to 98.3% within 200 steps
of unfreezing and holds a 90-98% band for the remaining 1800 steps, the
opposite of the same learning rate's behavior when applied from a
cold-started adapter. `--unfreeze-sonic` generalized into
`--unfreeze-sonic-at-step` to make this a first-class, reusable mode
(`results/sonic-correction-staged-unfreeze.md`). This is the strongest
single result in the correction-probe line and directly validates
unfreezing Sonic as a genuine next step in the mechanisms -> persistence
-> full training progression, provided it's staged rather than
simultaneous.

Unfreezing Sonic for joint training (the user's own next-in-sequence
step after all three persistence tiers) produced a real, substantial
regression, not the hoped-for improvement: value accuracy collapsed from
the quantity-pooling fix's 83.3% (frozen) to 10.0% (unfrozen,
`--sonic-learning-rate 1e-5`), while retrieval stayed strong or even
slightly higher (90.0% vs 83.3%) - unfreezing broke decode, not
retrieval. Two curve anomalies explain why rather than leaving it a
mystery: retrieval was already near-saturated at the very first
checkpoint (86.7% at step 100, no learning curve at all, unlike every
frozen-Sonic run's real climb), and decode loss stayed suspiciously flat
and low (~0.29-0.30) despite far worse exact-match accuracy than the
frozen run's higher-loss-but-higher-accuracy result. Most plausible
explanation: an unfrozen Sonic can cheapen both metrics' proxies (more
separable-looking representations, lower average per-token loss on the
mostly-deterministic template scaffolding) without solving the harder
problem either was meant to measure - specifically, decoding the one
quantity digit. Naive joint training (one combined loss, Sonic and a
cold-start adapter updated simultaneously) is not a safe default next
step even at a conservative learning rate; a staged approach (converge
the adapter with Sonic frozen first, unfreeze second) is the natural
next experiment, not yet tried (`results/sonic-correction-unfreeze-sonic.md`).

Re-running the real-corpus probe with `--context-pooling last` (a
correction to how this rerun was first framed: the commit that added the
flag attributed the original null result primarily to pooling dilution,
generalizing loosely from the synthetic-probe fix, but re-reading
`sonic-real-corpus-memory-probe.md` itself surfaces its own
already-cross-validated diagnosis - the null result was structural,
*"a random real-corpus continuation window is frequently not about the
preceding context segment at all,"* independently confirmed the same day
by the multi-query generator probe's large positive effect on a fix that
guarantees the dependency, not one that touches pooling) gives a real
but modest result, consistent with that structural diagnosis rather than
overturning it: both the aggregate LM loss reduction (0.0024 -> 0.0034)
and the hard-token-bucket-specific reduction (0.0109 -> 0.0152) grew by
about 40% under last-token pooling versus the original mean pooling -
the same direction and similar relative margin in two independent
metrics, not a coincidental fluctuation, but nowhere near the order of
magnitude the correction probe's identical-in-spirit fix produced,
because pooling precision only matters once a real context-continuation
dependency exists, and most random corpus windows don't have one
(`results/sonic-real-corpus-context-pooling-last.md`).

Scaling the batch-lane persistence probe from its 16-lane stand-in to
96 lanes (closer to the real production batch composition) preserves the
mechanism's behavior essentially unchanged - 84.4% current-episode /
59.2% prior-episode value accuracy at just 800 steps, matching the
16-lane run's eventual 85.8%/59.5% plateau but reached in under half the
optimizer steps (more lanes means more gradient signal per step,
compressing the convergence curve). Two real findings along the way:
`--compile layers`'s exhaustive autotune search is not worth its 47+
minute one-time cost for a short probe run (confirmed via `nvidia-smi`/
`ps`: pure compile-phase CPU work, zero training progress) - the user's
own framing, "worth it for a production run that runs for weeks but not
for a short probe," is exactly right; and even without compile, GPU
utilization stayed at ~9-10% at 96 lanes, confirming the real bottleneck
is the per-lane Python loop generating `distractor_reasoning` episodes
live (76,800 generations for this one run) - the same class of overhead
this project's whole batching effort was built to remove from the *GPU*
side, still present on the *CPU* side. The fix mirrors the real corpus's
own pretokenized-shards approach: pregenerate a pool of episodes once
instead of calling the generator every step - not yet built, but 258GB of
USB storage is already set aside for it
(`results/sonic-batch-lane-persistence-96-lanes.md`).

Doubling the batch-lane persistence probe's training budget (2000 ->
4000 steps, same 16-lane design) answers the open question its own
extended-run entry left hanging: the ~59% prior-episode (genuine
cross-optimizer-step) accuracy is a **real, noisy plateau**, not a curve
still climbing. Current-episode accuracy kept improving (85.8% -> 91.1%),
but prior-episode accuracy settled into a 50-81% band from step ~1200
onward with no further sustained movement by step 4000 (58.1%,
statistically indistinguishable from step 2000's 59.5%). This run also
used the new batched+compiled implementation at real scale for the first
time, doubling as a correctness check on that refactor
(`results/sonic-batch-lane-persistence-4000-steps.md`).

Genuine cross-optimizer-step persistence is confirmed, not just
promising: the batch-lane persistence probe's extended run (2000 steps,
4x the eval sample of its first run) shows a fact written into a lane's
bank during one training step - gradient graph fully torn down, bank
contents detached before the next step - correctly retrieved in a
*separate, later* optimizer step **59.5% of the time, against an 8.1%
no-memory floor** (both n=37, a stable, credible margin, unlike the first
run's n=4-11 noise). The learning curve is smooth and monotonic-ish
across 20 checkpoints, current-episode and prior-episode accuracy track
each other closely throughout, and the `no_memory` control stays flat in
a noisy but low 3.6-13% band the whole run. This is the last of the three
tiers this project set out to validate (single-episode, cross-episode
session, cross-optimizer-step) now having a real, statistically credible
positive result with the same mechanism and the same quantity-token
pooling fix (`results/sonic-batch-lane-persistence-extended.md`).

Getting the production execution machinery (`model.compile_layers`, bf16
autocast, `packed_projections`, `chunked_cross_entropy`) actually
exercised, per direct user request, produced a real, measured answer to
"is training fast enough on this hardware": **~50k tok/s** at a safe
physical shape (batch=16/seq_len=2048 or any shape with the same
~32,768 tokens/step), ~87-88% of the ~57k target, with the memory
mechanism's own overhead measured directly at a small **1.5%** (49,859
tok/s vanilla vs 49,100 tok/s with a 16/96-row memory injection,
matched 10-step runs) - confirming the basins are not the primary
obstacle to reaching the target; the remaining gap is in base Sonic
throughput. Two real engineering findings along the way:
`compile_layers`'s `fullgraph=True, dynamic=False` needs every call to
share one fixed shape (fixed via padding, the same prerequisite real
batching needs), and the naive `logits_from_hidden` + `F.cross_entropy`
pattern is unusable at any realistic scale (a full `[batch, seq, vocab]`
logits tensor is ~157GB at production shape) - `model.chunked_cross_entropy`,
already built into the vendored model, fixes this with no new
dependency. Doubling the physical shape to ~65k tokens/step didn't
cleanly OOM but collapsed throughput 18x (memory-swapping, not a crash)
- production's real path to its 96-sequence effective batch is
gradient accumulation over smaller physical micro-batches (confirmed),
not one large tensor, so the measured ~50k tok/s should be the number
that actually governs training wall-clock time
(`results/sonic-throughput-benchmark.md`).

The quantity-token pooling fix transfers directly to the harder
cross-episode session setting it was flagged as untested for two entries
ago: re-running the never-reset-bank session probe with
`--value-source quantity` took value accuracy from a flat ~15.8% to
**59.2%** and answer accuracy from 15.0% to 56.7%, with retrieval
statistically unchanged (60.0% both times, as expected - the fix only
touches the value path). Same two-phase learning-curve shape as the
single-episode fix (retrieval plateaus immediately, value/answer accuracy
stays at the no-memory floor through step 900, then climbs steadily to a
55-68% band). This run also produced the first clean, monotonic retention
curve in the whole project (design-v0.1.md SS38): 78.6% value accuracy at
0-2 intervening statements, down to 55.6% at 6+ - visible now only because
value accuracy climbed high enough overall for the bucket-level split to
rise above noise (`results/sonic-session-correction-quantity-pooling.md`).
The batch-lane persistence probe (genuine cross-optimizer-step
persistence, not yet run beyond a CPU smoke test) has had the same fix
ported ahead of its first full-scale run, defaulting to
`--value-source quantity` directly since it has no prior canonical run to
stay comparable with.

The write-side pooling dilution `sonic-vanilla-correction-baseline.md`
pointed at was the whole bottleneck. Pooling only the quantity digit's own
hidden state into the value vector - instead of mean-pooling the entire
rendered statement sentence, which had stayed unchanged through every
earlier fix attempt (key/value split, recency bonus, both together) -
took value accuracy from a flat ~6.7% (chance-level, unmoved across 20
checkpoints in every prior run) to **83.3%**, with answer accuracy
following it almost exactly to 80.0%. This is more than 4x above vanilla
Sonic's own 18.7% native in-context capability at the identical task, and
the `no_memory` control in the same run stayed pinned at exactly 6.67% the
entire time - ruling out a generic training-run fluke. The learning curve
has a clear two-phase shape: retrieval saturates to its usual ~80% plateau
by step 800-900 (matching the recency-bonus behavior already established),
while value/answer accuracy stays flat at chance through step 1200, then
climbs steadily to 80-83% by step 2000, consistent with the split-only
run's still-decreasing decode loss finally having a clean enough signal to
use (`results/sonic-correction-quantity-pooling-fix.md`). This is now the
best result in the whole correction/mutation probe line, and directly
resolves what the previous two entries below identified as the central
open bottleneck.

In parallel, the first test of genuine cross-episode persistence - several
independent correction episodes writing into one bank that is never reset
between them, each query facing the fully cluttered bank rather than only
what preceded it - showed retrieval survives the session structure at a
real, stable 60% plateau (below the single-episode probe's 83%, consistent
with a harder, more cluttered candidate pool rather than a broken
mechanism), while value accuracy stayed flat in the same pre-fix ~9-18%
band (this run predates the pooling fix above and still uses whole-sentence
value pooling). Re-running this session probe with the quantity-pooling fix
is now the most direct, highest-priority next step
(`results/sonic-session-correction-probe.md`).

A different baseline than every earlier `no_memory` condition -
"maybe Sonic is just dumb at digits" checked directly by giving Sonic
the *same* correction-chain task with no memory system at all, one
continuous in-context sequence, pure frozen-checkpoint inference, no
training. Result: vanilla Sonic's value accuracy (18.7%, 300 episodes)
is meaningfully above the 8.3% chance rate and, notably, **higher than
the trained memory injection's 6.7%** on the identical task. This rules
out "Sonic can't do digits regardless of pathway" as the primary
explanation for the flat value accuracy in the previous entry - if that
were true, the in-context baseline (which gets no help from anything
built in this project) would be expected to sit at chance too, and it
doesn't. The likelier explanation, now with direct supporting evidence:
`value_projection`'s write-side mean pool over the *whole* rendered
statement dilutes the one digit's signal among ~8-10 other tokens - the
same dilution pattern already confirmed once in this project for raw
real-corpus windows - while Sonic's own in-context processing never
pools anything away. The mechanism appears to be discarding information
Sonic could otherwise use, not compensating for something Sonic
inherently lacks (`results/sonic-vanilla-correction-baseline.md`). The
quantity-token-focused pooling fix this motivates is the direct next
experiment, in progress.

A direct challenge to the previous entry's "decode bottleneck" framing -
was flat answer accuracy just an artifact of the unforgiving all-tokens
exact-match metric, with Sonic actually getting closer than it looked? -
led to `eval_value_accuracy` (scored only on the quantity digits'
own token span) and a re-run of the split+recency configuration. The
isolated metric did not reveal a hidden higher number: `memory`'s value
accuracy stayed flat in the same chance-level band as `no_memory`'s
constant 6.67% (matching 1/12 random guessing) at *every one* of 20
checkpoints, even as retrieval accuracy climbed from 40% to 83% over the
same run. `answer_accuracy` and `value_accuracy` were the same number at
nearly every checkpoint for the memory condition too, meaning the
surrounding scaffolding text was never the bottleneck - the digit itself
is where accuracy actually lives, and it is completely decoupled from
retrieval quality. This sharpens, not softens, the earlier finding:
retrieval correctly identifies the right fact 83% of the time, and the
injected representation has no measurable effect on that one target
token's prediction. Candidate causes identified but not yet tested:
write-side mean-pooling over the whole statement sentence may dilute the
quantity token's signal the same way raw-corpus windows were diluted
earlier in this project (`sonic-real-corpus-memory-probe.md`); the
port's rank-64 injection may be too weak to override Sonic's own
confident frozen prior at that position; recovering one specific integer
via a frozen, untrained-for-this LM head may simply be a harder read-out
problem than routing to the right basin
(`results/sonic-correction-value-accuracy-addendum.md`).

Two follow-ups to the correction/mutation probe, tested in isolation:
a key/value split (separate `SonicKeyProjection` instances for the
retrieval key and injected value, rather than one vector serving both)
and an explicit write-order recency bonus (adapting
`persistent_context.py`'s decay idea - built for a query sequence, not
applicable as-is here - to this task's actual structure: known write
order). The recency bonus is a clean, substantial win: retrieval
accuracy climbed from the canonical run's ~60% plateau to a sustained
75-85% band from step 700 onward (83.3% final), and the double-edged
risk flagged before running it - that a naive "most recent write wins"
bonus would reward treating a `decoy_observation` sighting as
authoritative, since those are deliberately placed after the real
correction to test exactly this - was real in principle but didn't
manifest as serious damage in practice (trap-case accuracy tracked
close to overall throughout, no consistent large gap). The key/value
split alone left retrieval and answer accuracy both statistically
unchanged, though decode loss kept decreasing through the full
2000-step budget rather than plateauing early like the canonical run -
a real, partial signal, not a fix within this budget.

The more important finding: **answer accuracy did not move in any of
the three conditions** - still noisy in roughly the same 3-10% band
whether retrieval was ~60% or ~83%. Finding the right fact dramatically
more often did not translate into answering correctly more often, which
rules out retrieval quality as the primary constraint on answer accuracy
for this task. The bottleneck is squarely in decoding a specific,
correct value from a retrieved representation, not in finding the right
one to retrieve - not yet pinned down further
(`results/sonic-correction-key-value-split-and-recency-bonus.md`).

Per-fact writes fixed retrieval on the multi-query generator task
completely - 98.6% retrieval accuracy, saturating almost immediately
(94.96% at the first eval checkpoint) - and in doing so isolated a
sharper, more specific problem underneath: answer accuracy *dropped* to
5.0% (from 17.6% on the flawed pooled-context design it replaced), and
decode loss plateaus early (by step 700) far short of usable, despite
retrieval being essentially solved. Diagnosis: `SonicKeyProjection`'s
output serves two roles at once - the retrieval key *and*, via
`bank.centers[...]`, the injected value - a simplification documented
since the first training probe in this project, harmless until now. The
contrastive loss optimizes purely for discriminability, with nothing
keeping the resulting vector from collapsing toward "which fact this
is" at the cost of "what value this fact asserted." This maps directly
onto something design-v0.4 already specifies and this project has
simplified away since the start: SS3.1/SS6.2 call for **separate**
key and value encodings (`D_key=128`, `D_value=256`), not one vector
serving both. Concrete next step, not yet built: split key and value
into two distinct projections
(`results/sonic-distractor-generator-per-fact-memory-probe.md`).

The user's explicit priority after this: basic mechanisms (overwriting
a value that changes over time) before cross-episode persistence -
retrieval among distinct facts is the easier, now twice-validated
problem; a real world model needs mutation, which no probe had tested
until this point. `sonic_correction_memory_probe.py` uses the real,
vendored `distractor_reasoning.correction` generator (initial value,
one or more corrections, decoys placed *after* the real correction to
defeat "grab the last number" shortcuts, and critically a
`decoy_observation` - a mere sighting that must never override the
tracked value) with each statement in a chain getting its own basin,
and the retrieval target deliberately the *most recent* matching
statement, not just "the one relevant fact" - directly testing whether
recency is learnable at all, which is a genuinely open question:
`BasinMemory` has no representation of write order or time, so any
recency preference could only come from a learnable correlate (e.g.
correction-phrased text looking different from initial assertions), not
an explicit ordering signal. Smoke test found a real, meaningful
starting difference from the discrimination probes: 0% retrieval
accuracy pre-training here, versus 70-80%+ pre-training when facts are
topically distinct - near-duplicate statements about the same fact
don't naturally separate the way different facts do.

Result: real, fast learning that plateaus well short of ceiling, not
unlearnable and not perfect. Retrieval climbed from 0% to ~58% within
the *first 100 steps*, then held a noisy 55-67% band for the remaining
1900 steps with no further trend (66.7% final, within that band's
noise, not a sustained new high). A specific, plausible cause is visible
in the generator's own construction: each correction's phrasing is
chosen independently from a 3-template pool, giving a reliable
"correction-phrased vs. initial" signal (plausibly most of the fast
early climb) but nothing to distinguish a level-5 episode's 2nd
correction from its 3rd, final one, since both draw from the same random
pool - a real, explicable ceiling on the learnable correlate, not a
training failure. Answer accuracy stayed low throughout (3.3% -> mostly
6.7%, never climbing further) with decode loss dropping once early then
plateauing - the same signature as the per-fact probe's key/value
tension finding, now confirmed independently on a structurally different
task: two probes agreeing is stronger evidence than either alone
(`results/sonic-correction-memory-probe.md`).

The raw-corpus probe's own result landed (`results/sonic-real-corpus-memory-probe.md`):
a clean null, not an inconclusive one. `no_memory` 0.7172 vs `memory`
0.7148 final LM loss - a 0.0023 difference, noise. A difficulty-bucketed
breakdown (splitting all 11,460 eval tokens by how hard `no_memory`
itself found them, using that as a proxy for "needed real binding")
still found nothing: 2.6308 vs 2.6199 on the hardest quartile, 0.0793 vs
0.0798 (memory slightly *worse*) on the rest. Read together with the
multi-query probe's large positive effect on the same task family and
checkpoint, this pins the cause precisely: a random real-corpus
continuation window is frequently not *about* the context that precedes
it (the corpus mixes distractor tasks, chat, and topic switches loosely
in one stream), so a token being hard for `no_memory` reflects general
surprisingness, not a missed dependency on that specific context - the
multi-query fix works because it *guarantees* the dependency, not merely
because it adds more questions. Also ran this probe on GPU for the first
time as a live cross-check of the new WSL environment: CPU
(`76m25.655s`) and GPU (`23m16.987s`, ~3.3x faster) produced
statistically identical results, confirming the GPU migration changed
only speed, not behavior.

GPU is now the default way these probes run: `~/gtr_venv` in WSL
(torch+cuda 12.8, transformers, Triton 3.6.0, `fla-core`, pointed at the
same RTX 5080) has `manifoldflow` installed editable, all Sonic-relevant
tests pass there (the vision-track tests fail on an unrelated
torchvision/torch build mismatch, not a regression), and a real GPU
training run confirmed correct, matching CPU behavior at ~3.8x the
speed with the current unbatched, one-episode-at-a-time script design -
the full ~57k tok/s production figure needs the Triton HGRN scan
backend and batched (not sequential) processing, neither wired into
these probe scripts yet, deliberately deferred until "more serious
runs" per explicit direction (`project_sonic_gpu_env.md`).

Multi-query episodes against the real, vendored `distractor_reasoning`
generator fixed a real dilution problem the raw-corpus probe ran into:
that corpus randomizes names/values to force binding but is otherwise
repetitive scaffolding, so uniform mean LM loss over a random window
diluted the signal on the few binding-relevant positions. Fix: generate
one `World` once, ask it 6 heterogeneous questions (the generator's own
primary query + 3 extra direct queries + 2 reverse "who has N noun?"
lookups built directly against `World.direct_having_facts()`), track
each answer's token span, score loss/accuracy on answer content only.
Result: answer loss 5.674 -> **0.535** (frozen Sonic, `no_memory`
perfectly flat/always-wrong at every checkpoint, confirming the task is
genuinely unsolvable without memory), answer accuracy 0% -> **17.6%**
exact match, converging fast (stable by step 300, ~6x the gradient
signal per episode versus one query). Accuracy plateaus well below
saturation for a specific, actionable reason, not a vague "harder task":
every one of those 6 questions is answered from **one pooled
representation of the whole world**, since the write side still pools
context into a single vector - the discrimination/learned-retrieval
probes already showed the fix (one basin per fact, not per episode), just
not yet applied here (`results/sonic-distractor-generator-memory-probe.md`).

This project moved to its own repository (`RiujinZero/Sonic`, this
worktree) to keep everything under the company account; `docs/design-progress.md`'s
history before this point lived in `RiujinZero/Manifold-Flow`, still
reachable from the `claude/design-doc-log-review-4819a4` worktree there.
`vendor/GTR-sonic-ssm/` was also added here - a wholesale (not
line-by-line) copy of `gator/` and `gator_linear/`, including the real
Triton HGRN kernel source, kept as reference material for the GPU
training infrastructure work that's still on hold (see
`vendor/GTR-sonic-ssm/README.md`).

A trainable retrieval key (`SonicKeyProjection`, trained via a separate
InfoNCE contrastive loss over in-episode negatives, sharing an optimizer
with `SonicMemoryPort`) closed the discrimination probe's ~83.6%
retrieval ceiling completely: **100.0% retrieval accuracy from the very
first eval checkpoint**, sustained for the entire 2000-step run, and
**100.0% final answer accuracy** (up from 86.7%). The discrimination
probe's diagnosis - retrieval was a fixed function of an untrained key
space, not a fundamental limit - was exactly right, and the standard fix
(one linear layer, one contrastive loss term) worked completely
(`results/sonic-pretrained-memory-learned-retrieval-probe.md`).

Direct prerequisite for the user's stated next goal - "does ManifoldFlow
help Sonic converge faster than vanilla Sonic" - cleared: every probe
before this kept Sonic frozen, so none had tested whether jointly
fine-tuning Sonic with the memory components stays stable.
`sonic_finetune_stability_probe.py` unfreezes Sonic (separate Adam
learning rates: `2e-5` for Sonic's pretrained weights, `1e-3` for the
freshly-initialized port/key-projection) and trains for 250 steps,
watching for NaN/Inf, gradient-norm blowup, and held-out
ordinary-language loss drift. Result: no NaN/Inf at any step, gradient
norms trended *down* over the run, task loss and accuracy improved
cleanly (decode loss 9.30 -> under 1.0, answer accuracy to 50-75% by step
160). One real, honestly-reported caveat: held-out LM loss rose ~10%
relative in the first 50 steps, then plateaued rather than diverging -
plausibly just the expected cost of fine-tuning on a narrow synthetic
task with no ordinary-language data mixed in (design-v0.4's own Stage 6
curriculum exists to prevent exactly this), not evidence that the
architecture combination itself is unstable
(`results/sonic-finetune-stability-probe.md`). Open: the actual
convergence-speed comparison still needs (a) a data design where
information recurs across many training steps, so a vanilla Sonic could
in principle also learn it parametrically - the current per-episode
isolated-random-facts task gives vanilla Sonic nothing to race against -
which the user is thinking through separately, and (b) a genuine
vanilla-Sonic control condition, which no probe in this project has run.

Multi-fact discrimination, the natural next gap after the pretrained
probe's single-basin (trivially-retrieved) result: `facts_per_episode=4`
competing facts written into the same per-episode bank, `retrieval_accuracy`
(did `bank.read()` return the right basin - genuinely non-trivial now)
scored separately from `answer_accuracy` (did decoding then recover the
right object), since these can fail independently. Same frozen-checkpoint,
trained-port-only design, same 2000-step budget. Result: **86.7% final
answer accuracy** vs **0.0%** for `no_memory` (25% chance floor for 4-way
discrimination). `no_memory`'s loss (8.2-9.0) again sits well above
`ln(4)=1.39`, the same confident-wrong-completion pattern as the
single-fact probe, not a weak baseline. Retrieval accuracy is a fixed
ceiling this setup does not train (mean 83.6%, range 75.0-88.3%, no
improving trend across 2000 steps) - expected, since only the port's
parameters train and `bank.read()`'s routing depends entirely on the
frozen checkpoint's own mean-pooled representations, untouched by the
optimizer. `answer_accuracy` tracks `retrieval_accuracy` closely at every
checkpoint (occasionally slightly above it, from lucky object collisions
between competing facts, not from decoding beating retrieval
structurally), plateauing together by step ~700 - the ~13-point gap from
the single-fact probe's 100% is attributable almost entirely to retrieval,
not decoding, a precisely located bottleneck rather than a vague
"harder task, lower score"
(`results/sonic-pretrained-memory-discrimination-probe.md`). Open: a
trainable projection before the write/read step (distinct from
`SonicMemoryPort`, which never influences what gets retrieved) is a real
candidate fix for the retrieval ceiling, untested; so is a longer-lived
bank persisting across episodes rather than reset fresh each time - the
actual "persistent state" claim design-v0.4 makes, which no training
probe in this project has attempted yet.

Starting from the real checkpoint fixes the cold-start problem the first
training probe identified: user direction was explicit that recall and
persistent state actually working matters more than staying spec-pure on
the write path, so rather than pursuing an auxiliary-loss/warmup bridge
for SS6.3's detached-write design, `sonic_pretrained_memory_probe.py`
loads Sonic from the real checkpoint (`sonic_ssm_128k_steps.pt`, 6.1B
training tokens) and freezes it, training only `SonicMemoryPort`'s own
parameters. A debug check first confirmed the premise: the frozen
checkpoint's fact representations spread out substantially more than the
random-init model's had (0.60-0.91 cosine similarity across different
facts, versus a near-flat 0.91-0.98 band regardless of content before) -
object identity present as a real secondary signal beneath the dominant
entity-identity axis. Canonical result, same task/budget as the first
probe: **100.0% final accuracy** for the memory condition (frozen Sonic +
trained port) versus **0.0%** for `no_memory` (frozen Sonic, no port,
structurally cannot see the fact). The climb was smooth and monotonic
(14% -> 39% -> 73% -> 87% -> 100% at steps 100/200/300/400/500), loss
falling cleanly from `ln(8)=2.08` to 0.0048 by step 2000, no instability -
unlike the first probe's transient batch=1 spike that collapsed back to
chance. `no_memory`'s loss (8.3-9.2) sat well *above* `ln(8)`, meaning the
frozen pretrained model confidently commits to some other completion
rather than hedging uniformly - a genuinely adversarial floor. Combined
with the first probe's finding that the injection/gate/retrieval
mechanism trains correctly given any usable signal, this is now a
complete, positive answer to "does training work with a Sonic and
basins": yes, when the write side has legible structure to lean on, which
a pretrained checkpoint supplies and a random init does not
(`results/sonic-pretrained-memory-probe.md`). Open: whether fine-tuning
Sonic (not freezing it) changes this, and whether the result holds beyond
the single-fact-per-episode simplification both probes share.

First real training-viability attempt at design-v0.4's core hypothesis
("does training work with a Sonic and basins"), not another
inference-only probe: `SonicMemoryPort` (SS6.2's gated low-rank residual,
9 unit tests) wired to a real `BasinMemory` write/read cycle around a
small, freshly-initialized Sonic on a synthetic two-forward-pass recall
task (fact and query/answer run as independent `forward_hidden` calls
with fresh state, so external memory is the only channel that can carry
the binding - an earlier single-sequence version was uninformative,
since a plain SSM solves in-context recall for free). Canonical result:
flat at the chance floor (12.0% vs 12.5% floor) for the SS6.3-faithful
condition (retrieved memory value treated as a detached constant) across
2000 gradient-accumulated steps, `lr=1e-3`, gate opened at init. A
direct instrumentation check ruled out a wiring bug - the port's
`up`/`down`/`gate` weights move substantially during training. A
non-spec diagnostic (`--no-detach-write`, gradient allowed through the
write-side representation) isolates the actual cause: same architecture,
same task, same budget, **52.0% accuracy**, loss `2.08 -> 0.80`. The
injection/gate/retrieval mechanism itself is not broken - given any
gradient signal shaping what the fact representation encodes, it learns
the task cleanly and fast. What doesn't yet work at this scale is SS6.3's
literal detached-write design when training completely from scratch: with
no pretrained model supplying already-legible representations, and no
gradient pressure making the write-side representation legible either,
the small port cannot learn to decode object identity from an untrained
model's near-inseparable hidden states (>0.96 cosine similarity between
different facts at init) within budget. A real, actionable gap for
design-v0.4, not a dead end: SS6.3's detachment is well-motivated for the
target system (many unrelated writes aggregating into one basin over
time) but has no answer yet for the cold-start problem it creates when
there is no pretrained model to lean on
(`results/sonic-memory-training-probe.md`).

Sonic's Phase 3 (persistent context, the centerpiece claim) passed on the
*first* attempt - no hard-negative-dataset detour needed, because Sonic's
own Phase 2 result already showed a 91.7% stateless floor (not Qwen's
100% ceiling) on the original 24-fact dataset, so the precondition that
made Qwen's first attempt uninformative didn't obviously hold here, and
didn't in fact hold: `alpha=0.5, decay=0.95` reached 100% at N=2/5,
declining toward but staying at/above the 91.7% floor through N=20;
`decay=0.90` at the same `alpha` actually dipped *below* the floor by
N=20 (87.5%), an honest small-sample-noise-caveated observation, not
over-claimed. `phase_3_exit_gate_passed: true`, `collapse_detected: false`
everywhere (`results/sonic-phase3-persistent-context.md`). The
decay-rate ordering (0.95 outlasting 0.90 at matched alpha) is now the
same finding in three independent settings - the original vision-domain
probe, Qwen's hard-negative result, and Sonic here - three unrelated
representations (a frozen ResNet, a ~4B transformer, a 100M SSM)
agreeing on which decay rate holds up longer. `PersistentBasinContext`
was reused with zero code changes for the third time. Sonic's Phase 2
(static bank) result mirrors Qwen's structurally: real
write/read retrieval mostly holds (91.7%, 22/24 - the first time any
Phase 2 retrieval check in this project has missed a query, Qwen
included) but growth control fails the same way Qwen's did - no single
global distance threshold separates paraphrase-reinforcement from
genuinely-different-fact, worst-case same-fact distance (0.1813)
exceeding best-case cross-fact distance (0.0927)
(`results/sonic-phase2-static-bank.md`). Two independent architectures
sharing no code beyond `BasinMemory` itself now show the identical
structural limitation - stronger evidence it is a property of the
global-threshold mechanism, not a Qwen-specific artifact.
`phase_2_exit_gate_passed: false` for both models. Sonic's own Phase 0
and Phase 1 (design-v0.3.md's protocol, repeated
against Sonic exactly as it was run against Qwen) both pass. Phase 0:
bit-identical D128 keys across a full process restart
(`results/sonic-phase0-deterministic-replay.md`) - notably on CPU (Sonic's
default `scan_backend="doubling"` is pure PyTorch, no Triton), unlike
Qwen's GPU run, so the two bit-exact results aren't evidence for quite
the same reason. Phase 1: the identical sweep protocol (same dataset,
same depth fractions, same pooling rules) found the *same* qualitative
pattern Qwen's Phase 1 found - mean pooling dominates last-token pooling
by 25-50 points at every depth, dwarfing the largest depth-only gap
(under 9 points) - on an architecture that shares nothing with Qwen's
transformer except being a causal LM
(`results/sonic-phase1-representation-sweep.md`). Selected
`layer_fraction=0.75` (a genuine near-tie with 0.5, broken only by
iteration order - unlike Qwen's clean win), `top1=91.7%`, clearing both
controls; `sonic_adapter.py` now hooks that layer directly rather than
using `forward_hidden`'s own final-norm output, which the sweep never
tested as a distinct, fairly-compared condition. Sonic's numbers sit
below Qwen's at the same protocol (100M vs ~4B params, 6.1B training
tokens), consistent with, not proof of, Sonic being an earlier-stage
model. Phase 7 (Sonic portability, v0.3's original narrower framing) is
unblocked: Sonic is `SonicSSMLanguageModel`
(`src/manifoldflow/sonic_ssm.py`, vendored byte-identical from the
separate `GTR-sonic-ssm` repo - a genuine diagonal state-space model,
architecturally unrelated to that repo's other "Gator" graph-routing
models), and a real production checkpoint
(`sonic_ssm_128k_steps.pt`, update 125,000, 6.14B tokens seen) loads
cleanly with every one of its 100,608,101 parameters matching exactly.
`SonicRepresentationAdapter` (`sonic_adapter.py`) implements the same
`RepresentationAdapter` interface `QwenRepresentationAdapter` already
does, reusing `pool_hidden_states`/`FixedSeededProjection` unchanged - a
second, architecturally unrelated model now proven under the same
modality-independent primitives. A real forward pass on the real
checkpoint produced grammatically coherent top-5 continuations for "The
capital of France is" (not noise), confirming genuine trained language
competence; greedy `generate()` degenerates into a repetition loop over
longer spans, consistent with this model's much smaller scale and
earlier training stage relative to Qwen3.5-4B, not a bug
(`results/sonic-phase7-checkpoint-loaded.md`). This satisfies v0.3's own
Phase 7 exit gate; design-v0.4's actual vision - joint training, not
inference-only probing - remains unstarted and is a substantially larger
undertaking. Phase 6 hardening happened first this session. Consolidation now runs as a genuine
background "sleep state" rather than only synchronously:
`consolidate_in_background()` spawns a daemon thread; `write()` rejects
outright rather than blocking or silently losing data while a
consolidation holds the write lock; `read()` is untouched by that lock
and keeps serving correct results throughout, verified with a real
concurrency test (build the bank, start background consolidation, hammer
`read()` from the main thread while it runs) rather than only checking
before/after boundaries as the original Phase 6 benchmark did. That test
surfaced a real bug on its first run: `write()`'s first-basin bootstrap
and `from_bulk()` both created basins with `count=0` (correct for
`BasinState.from_centers()`'s actual "seeded, not yet observed" purpose,
wrong for a basin that represents one completed real observation),
silently masked in every earlier Phase 6 test because `consolidate()`
always prunes before merging and prune's own rebuild path happened to
floor every count to >=1 whenever anything was actually expired - which
every earlier test setup arranged, and this new one deliberately didn't.
Fixed at the root via `BasinState.from_statistics(..., count=1)`, the
same primitive already used for every basin after the first
(`results/qwen-phase6b-background-consolidation.md`). Phase 6 (versioned
consolidation and scale) was already core complete before this:
`prune_expired_basins` (rebuild-based, `O(n)`) and
`merge_compatible_recent_basins` (exact parallel-Welford merge, only ever
touching `"recent"`-tier basins) both hold 100% retrieval accuracy and
full foreground read availability through a real synthetic 10K-basin
test, with prune and merge timing measured separately - 7ms->99ms for
prune (10x data, ~14x time, near-linear) versus 0.82s->77.7s for merge
(10x data, ~95x time, `O(n^2)`) - an honest, load-bearing limit found
before it caused a problem: a first implementation materialized one full
`n_recent x n_recent` distance matrix (40GB at 100K recent basins),
caught in code review and fixed to chunk before any real run, though the
underlying time complexity is unchanged and merge past ~10K basins was
not attempted for that reason (`results/qwen-phase6-consolidation-scale.md`).
Split is explicitly not implemented - SS5.2 itself says it needs
subcentroid/sketch infrastructure this project doesn't have, so
attempting it would be guessing, not implementing the design. Phase 5
(continual write lifecycle) added trust-tier gating,
repetition-based promotion, expiration, and supersession to `BasinMemory`,
then tested the exit gate's three clauses for real on Qwen3.5-4B: new
memories were immediately usable (100%) and corrections cleared their old
record cleanly (100%), but the correction's new paraphrase and unrelated
(uncorrected) entities' retrieval both landed at 86.7% (13/15) - short of
a clean pass, not a collapse (`results/qwen-phase5-write-lifecycle.md`).
Both shortfalls landed at the identical rate, which the writeup flags
rather than explains away: it might be one shared cause or unrelated
per-entity difficulty (matching Phase 3b's own 86.7% floor on a similarly
constructed dataset), and this run cannot distinguish the two without
per-entity logging it did not capture. Promotion, expiration, and
trust-tier rejection are covered by unit tests
(`test_continuous_memory.py`) but were not exercised by this real-data
run, since nothing in the stream repeated identical content. Phase 4
(adaptive compatibility gate) is concluded, with a genuinely
counter-intuitive result worth remembering: three real evaluations, not
one. The first (`results/qwen-phase4-adaptive-gate.md`) found
`AdaptiveBasinContext`'s binary top-1 agreement check fixes false
stale-memory carryover decisively (13.3% -> 83.3% current-input accuracy
on an immediate topic switch) at a real cost of 3-13 retention points,
and diagnosed the cause as SS6.3's own explicit warning against a plain
top-1/cosine check. Two attempts to build the distributional comparison
SS6.3 recommends instead - one over the full retrieval-width candidate
union, one narrowed to each side's own top few candidates
(`results/qwen-phase4b-fixed-agreement.md`,
`results/qwen-phase4c-narrowed-agreement.md`) - both made *both* metrics
measurably worse than the binary version, not better: at real 60-basin
scale, graded (not sharply bimodal) RBF scores diluted the overlap signal
enough that genuine topic switches stopped being recognized, collapsing
retention to 38-40% by N=20 and barely moving carryover. The binary check
was reverted to and kept, documented in `adaptive_context.py`'s own
docstring as the empirically best of three tested variants despite not
matching the doc's theoretical preference - a real, load-bearing lesson
that a theoretically-motivated fix is not automatically a better one
until it is actually run. Phase 3's centerpiece exit gate
(reproduce the vision-domain graded forgetting curve in language) is
**complete**, but only after an
important detour. The first attempt, on the Phase 1/2 dataset, was
uninformative: every config, *including the `alpha=1.0` stateless
control*, scored a perfect 1.000/1.000 at every distraction depth
(`results/qwen-phase3-persistent-context.md`) - those facts are separable
enough that fresh, context-free routing never fails even under heavy
distraction, leaving no floor for persistence to lift above (the vision
probe's "the stateless control matters as much as any other row" lesson,
inverted: there the floor was low enough to leave room, here there was no
room to begin with). A second dataset, built around SS8.2's own
"same-type hard negatives" (60 facts as 30 same-entity minimal pairs, e.g.
"X owns the copper glider" vs "X owns the iron key"), produced a genuine
86.7% stateless floor - and every persistent-context config exceeded it,
with `decay=0.95` extending the advantage further than `decay=0.9` at
matched `alpha`, exactly matching the vision-domain curve's decay-rate
ordering (`results/qwen-phase3b-hard-negatives.md`). `PersistentBasinContext`
was reused with zero code changes for this - first evidence the
vision-domain persistence result is a property of the mechanism, not an
accident of ImageNet's geometry. Phase 2's exit gate was mixed, reported honestly
rather than rounded up: real payload retrieval through the actual
write/read path is 100% correct, but the *other* half of the exit gate -
"does not show uncontrolled basin growth" - does not hold under a single
global distance threshold, a genuine finding, not a bug
(`results/qwen-phase2-static-bank.md`). Phase 1's exit gate passed cleanly:
on a real frozen Qwen3.5-4B, mean pooling at 75% depth reached 100%
held-out paraphrase retrieval versus 75% lexical and 0% random controls
(`results/qwen-phase1-representation-sweep.md`). Phase 0's exit gate passed
before that: bit-identical D128 keys across a full process restart
(`results/qwen-phase0-deterministic-replay.md`). The preceding vision track
(below) established the RBF-default decision, local-update lifecycle, first
HNSW integration, and the basin-traversal persistence mechanism; those
components are inputs to the language testbed, not evidence that persistent
language memory works.

### Vision-track history (image domain, preceding v0.3)

The project has completed the **Phase C — Flow and Local Learning** decision
gate and is in a systems-focused **Phase C/Week 3 pivot**. Matched online
controls show that a simpler top-4 RBF prototype memory wins at equal core
vector storage: 85.97% shifted accuracy versus 82.97% for local novelty. The
§10.3 kill criterion therefore fires for the current attractor formulation.
ANN routing has now been pushed to systems scale on a synthetic bank
(`ann-routing-scale.md`): build time and lookup+rerank latency stay well
inside budget through 1M basins for both targets. A settings sweep
(`ann_scale_sweep.py`, which reuses one exact reference across many HNSW
configurations instead of recomputing it per run) closes the recall gap for
Euclidean/RBF routing at both 100k and 1M using only a higher `ef_search` -
no index rebuild needed - and closes it for anisotropic-energy routing at
100k. It does **not** close at 1M for the energy target: pushing `ef_search`,
graph degree, and construction effort all higher only reaches 93.6% with
sharply diminishing returns, consistent with a representational ceiling
(HNSW's shared Euclidean graph cannot fully capture a per-basin anisotropic
metric) rather than a search-budget shortfall. Since Euclidean/RBF is already
the design's preferred scorer per the matched-online-controls result below,
this does not block the systems path that matters; it is tracked as an open
research question specific to the attractor geometry.

A real (non-synthetic) 10k bank was then built and routed on Fashion-MNIST
(`real-10k-ann-routing.md`). Euclidean/RBF transferred cleanly from the
synthetic study (99.8% recall at default settings). Anisotropic-energy
routing did not: that bank needed roughly 8x the candidate width the
synthetic bank needed at the same 10k scale to clear 95% recall, which was
initially read as "real data is harder for energy routing than synthetic
suggested." A second real 10k bank, built and routed from real ImageNet-1K
latents (`first-imagenet-bank.md`), corrected that: it clears 95% energy
recall at the *synthetic* study's original candidate width (M=128), not the
wider one Fashion-MNIST needed, despite comparably wide basin rate ranges on
both banks. The Fashion-MNIST result was a property of that specific bank,
not a general real-vs-synthetic effect - energy's required candidate width
is bank-specific and not yet predictable without measuring the bank, while
Euclidean/RBF has now cleared 95% recall on every bank tested (synthetic,
real Fashion-MNIST, real ImageNet) with no exceptions. ImageNet-1K remains
scoped as a matched sparse-memory baseline study routed on Euclidean/RBF
unless revised geometry first clears the learning gate.

**Scope caveat on the §10.3 kill criterion:** the comparative-value gate in
`matched-online-controls.md` is measured only on Fashion-MNIST `d=128` image
latents. It is not yet evidence that analytic local-flow learning
underperforms simpler online memories in general - the design's own §0 scopes
the MVP to small-image domains and reserves event-stream, sensor, and
language-state memory for later. Latent geometry, class structure, and drift
differ enough across domains that the comparative outcome could change. Treat
the kill criterion as vision-scoped, not domain-general, until a matched
Manifold-vs-RBF control is run on at least one non-vision latent stream at an
equal vector budget.

## v0.3 Qwen testbed ledger

| Design section | Requirement | Status | Authoritative evidence | Next exit condition |
|---|---|---|---|---|
| §3, §15 | Modality-independent `RepresentationAdapter` and `ContinuousMemory` boundaries | `RepresentationAdapter` core complete; `ContinuousMemory` not started | `representation.py`, `qwen_adapter.py`, `test_qwen_representation.py` | Port the existing RBF bank behind `ContinuousMemory` |
| §4, Phase 0 | Frozen Qwen3.5-4B hidden-state capture | Complete | `QwenRepresentationAdapter` (targeted forward hook on real `model.model.layers[i]`, verified against the installed Qwen3.5 implementation, not `output_hidden_states=True`); `qwen-phase0-deterministic-replay.md` | None for Phase 0; Phase 1 selects the layer/pooling rule this only wires up |
| §4, Phase 0 | Deterministic seeded D128 projection and normalization | Complete | `FixedSeededProjection`, `pool_hidden_states`, `layer_index_for_fraction` (unit-tested, model-independent); real bit-exact cross-restart run on Qwen3.5-4B/RTX 5080, `qwen-phase0-deterministic-replay.md` | None; Phase 0 exit gate passed |
| §4.2, Phase 1 | Three-layer × two-pooling stateless representation sweep | Complete | `qwen_representation_sweep.py`, `qwen-phase1-representation-sweep.md`; real Qwen3.5-4B, single forward pass per text via simultaneous multi-layer hooks | Selected 75% depth + mean pooling (100% top-1) frozen as `QwenRepresentationAdapter`'s default; a lexically-decorrelated paraphrase set would be a stronger follow-up control, not blocking |
| §5, Phase 2 | Inspectable payload records and recent microbasins | Retrieval complete; global growth control does not hold | `continuous_memory.py` (`BasinMemory`, one-shot creation - deliberately not `OnlineBasinMemory`/`NoveltyStager`, whose persistence>=2 gate contradicts SS5.1's one-shot acquisition), `qwen-phase2-static-bank.md`: 100% real payload retrieval; no single global `novelty_threshold` cleanly separates same-fact-reworded from different-fact at this scale | Try a per-basin adaptive threshold (reusing `raw_rate`, currently unused since SS10 sets `flow_enabled: false`) before concluding growth control needs more than that |
| §6.2 | Two-pass raw/context ANN union with RBF reranking | Not started | existing single-path HNSW/RBF primitives | Preserve raw-supported candidates under context bias |
| §6.4 | Raw-key-only semantic writes | Not started | — | Tests prove blended context never updates basin centers |
| §7.3 | Supersession and correction graph | Not started | — | Current value wins while history remains auditable |
| §8 | Fresh-KV episodic harness and six required controls | Not started | — | MF-reset stays at leakage floor; prompt-memory upper control works |
| §8.2 | Randomized nonce episodes and held-out templates | Not started | — | Fact, binding, correction, unknown, reset, and trust-tier families replay deterministically |
| §8.3 | Retention, carryover, responsiveness, correction, unknown, latency metrics | Not started | — | Machine-readable summary enforces the two-point anti-collapse gate |
| §9 Phase 3 | Fixed alpha/decay persistence sweep | Complete | `qwen_hard_negative_probe.py`, `qwen-phase3b-hard-negatives.md`: on a genuinely confusable 60-fact same-entity-minimal-pair dataset (86.7% stateless floor, versus the original 24-fact dataset's uninformative 100% ceiling in `qwen-phase3-persistent-context.md`), every persistent config exceeds the floor with a graded decline as distraction accumulates, `decay=0.95` extending the advantage further than `decay=0.9` at matched `alpha` - reproducing the vision-domain curve's exact qualitative signature with zero code changes to `PersistentBasinContext` | Phase 4: adaptive compatibility gate |
| §9 Phase 4 | Adaptive compatibility gate | Concluded; real tradeoff, binary check kept as empirically best of three tested variants | `adaptive_context.py` (`AdaptiveBasinContext`); three real evaluations - `qwen-phase4-adaptive-gate.md` (binary top-1: carryover 13.3%->83.3%, retention 98.3%->86.7% @N=20), `qwen-phase4b-fixed-agreement.md` (full-union overlap coefficient: both metrics collapsed further, carryover 13.3%->3.3%), `qwen-phase4c-narrowed-agreement.md` (narrowed overlap: still collapsed, carryover only 13.3%->16.7%) - SS6.3's own recommended distributional-comparison fix underperformed the simpler check it was meant to replace on both attempts, at real bank scale | A calibrated/fit compatibility function (not another hand-picked formula) if this is revisited; Phase 3's fixed config remains the safe default meanwhile |
| §9 Phase 5 | Promotion, expiration, trust, and long-stream lifecycle | Core complete; real run mixed | `continuous_memory.py` (`DURABLE_TRUST_TIERS`, `tier()`, `expire_stale_microbasins()`, `write(..., supersedes=...)`), `qwen-phase5-write-lifecycle.md`: real Qwen3.5-4B run - immediate usability 100%, stale-record clearing 100%, but correction retrieval and unrelated competence both 86.7% (13/15), short of the exit gate's thresholds; promotion/expiration/trust-rejection covered by unit tests only so far, not exercised in the real run | Per-entity result logging to isolate the two 86.7% shortfalls' cause; a genuine before/after unrelated-competence baseline |
| §9 Phase 6 | Versioned consolidation and 10K→1M scale | Prune + merge complete through 10K, now background-runnable; split not implemented | `continuous_memory.py` (`prune_expired_basins`, `merge_compatible_recent_basins`, `consolidate`, `consolidate_in_background`, `from_bulk`), `qwen-phase6-consolidation-scale.md` + `qwen-phase6b-background-consolidation.md`: 100% accuracy and full foreground availability at 1K and 10K synthetic scale, now verified under genuine concurrent reads during a live background consolidation (not just checked before/after); prune is `O(n)` (7ms→99ms), merge is `O(n^2)` (0.82s→77.7s, ~2hr extrapolated at 100K) - a real, measured limit of exact-brute-force candidate generation, not assumed | Bounded-candidate-width or HNSW-backed merge before attempting 100K-1M; split needs subcentroid/sketch infrastructure that doesn't exist yet; write queue-and-replay during consolidation if the write rate ever needs it |
| §11 | Per-event trace and version manifest | Not started | — | Replay contains model/tokenizer/template/layer/projection/bank versions |
| §13 | Full Qwen success criteria | Not started | — | All ten criteria pass on held-out episodes |

## v0.2 image research ledger

| Design section | Requirement | Status | Authoritative evidence | Next exit condition |
|---|---|---|---|---|
| §2 Architecture | Compact encoder, frozen normalizer, task head/decoder | Complete for Fashion-MNIST | `models.py`, `normalization.py`, Phase A artifacts and results | Add classification-only 224px backbone adapter for ImageNet entry |
| §2 Architecture | CPU-resident basin bank with explicit lifecycle | Core complete | `bank.py`, initialization artifacts, serialization tests | Add restartable foreground bank/value/delta checkpoint |
| §3.1–3.2 | Positive bounded rates and local quadratic energy | Complete | `flow.py`, analytic and stability tests | Preserve under future quantization |
| §3.3 | Exact candidate rerank and hard/soft top-K flow | ANN validated on multiple independent real 10k/100k banks (Fashion-MNIST, three ImageNet banks with different occupancy) and pushed to 1M on a synthetic bank; Euclidean/RBF clears 95% recall (M=32) with no exceptions on every bank tested | `first-ann-routing.md`, `ann-routing-scale.md`, `real-10k-ann-routing.md`, `first-imagenet-bank.md`, `imagenet-500k-real-data.md`; Euclidean 99.8-99.9% across all real banks, 96.5% synthetic at 1M; energy's required candidate width (M=32 to M=512 observed) tracks each bank's actual rate heterogeneity, not real-vs-synthetic, scale, or basin count alone | Add flow latency; real 1M remains untried for either scorer |
| §3.4 | Frozen-gate analytic segment and exactness caveat | Complete | implementation, docs, flow tests | No further MVP work unless multi-segment flow is reintroduced |
| §4.1 | Frozen-backbone Welford statistical updates | Complete at first real-stream scale | `OnlineBasinMemory`, locality tests, `first-continual-stream.md` | Repeat with image-derived shift and larger bank |
| §4.1 | Update basin class/value statistics locally | Complete; negative comparative gate | `matched-online-controls.md`; nine-method, three-seed aggregate | Retain as sparse-memory baseline; revise geometry before claiming attractor advantage |
| §4.2 | Sparse task-gradient updates | Deferred | intentionally absent | Begin only if statistical mode earns value |
| §4.3 | Persistence-gated basin creation | Complete at first real-stream scale | class-compatible stager; 105/6,000 created; zero rejection | Show value beyond fixed updates and repeat under image-derived shift |
| §4.3 | New/moved centers enter exact delta index | Core complete for exact and ANN main indexes | `ExactMainDeltaIndex`, `AnnMainDeltaIndex`, stale-override tests | Measure ANN+delta recall/latency as delta grows |
| §5 | Snapshot/version publication, delta replay, rollback | Publication and rollback complete; delta replay not applicable yet (no consolidator rebuild to replay onto) | `checkpoint.py`, `test_checkpoint.py`; restart-verified at real 100k-basin scale in `imagenet-100k-scale-and-persistence.md` | Delta replay is part of §5.2's merge/split/prune consolidator, still not started below |
| §5.2 | Merge with exact sufficient-statistic preservation | Complete through 10K, on the Qwen testbed's `BasinMemory`, not the vision-domain `OnlineBasinMemory`/`BasinState` path | `continuous_memory.py`'s `merge_compatible_recent_basins`, `qwen-phase6-consolidation-scale.md` | Port the same rebuild-based pattern to the vision-domain bank if merge is ever needed there; measured `O(n^2)` cost limits it past ~10K without a faster candidate index |
| §5.2 | Split with subcluster evidence | Not started | — | Add streaming subcentroids/sketch before split logic |
| §5.2 | Prune, tombstone, quantize, validate | Prune complete (Qwen testbed only); tombstone/quantize/validate not started | `continuous_memory.py`'s `prune_expired_basins`, `O(n)`, 100% accuracy through 10K | One operation at a time after versioning, as already planned |
| §6.1 | Explicit bank memory accounting | Complete at 1k; planned at scale | initialization metrics and design budget | Report bank + values + ANN + delta at 10k/100k/1M |
| §6.2 | Bounded active working set | Small-bank CPU measurement complete | `first-ann-routing.md`; M=128 energy p95 0.159 ms | Include flow/transfer and scale through 1M |
| §6.3 | Hot/warm/cold hierarchy | Not started | — | Required for device-oriented scale MVP, not first ImageNet smoke |
| §8 Phase A | Representation baselines and width sweep | Complete for Fashion-MNIST d64/d128 | `first-training-runs.md`, three-seed d128 artifacts | CIFAR-10 remains; ImageNet entry uses frozen discriminative adapter |
| §8 Phase B | 1k–10k basin initialization and exact routing | Complete at 1k and 10k | initializer, three seed banks; `real-10k-ann-routing.md` (10,000/10,000 basins active, no empties) | None; CIFAR-10 10k pass optional |
| §8 Phase C | Hard then soft flow and matched ablations | Complete | flow, prototype, conditional, semantic, and hybrid result records | Static flow remains secondary because RBF/prototype wins overall |
| §8 Phase C | Online updates, novelty staging, delta index | Complete; pivot triggered | `matched-online-controls.md`; protocol-validating aggregator | Move ANN work forward for both Manifold and winning RBF scoring |
| §8 Phase D | Versioned consolidation | Not started | — | Begin after Phase C learning gate |
| §10 Systems | ANN top-1 recall ≥95% | Pass at 1k/2k/10k/100k for both scorers, and at 1M for Euclidean/RBF; energy alone remains open at 1M | `ann-routing-scale.md` settings sweep; Euclidean 96.5% at 1M, energy 96.1% at 100k / 93.6% best at 1M | Real (non-synthetic) bank at these scales; investigate an anisotropy-aware index if energy routing at 1M is later needed |
| §10 Systems | Lookup/rerank/flow p95 and sublinear scaling | Latency sublinear and within budget through 1M at the passing settings; flow still excluded | `ann-routing-scale.md`; p95 lookup+rerank 0.127-0.86 ms at the cheapest passing settings through 1M | Add flow/transfer to the measured path |
| §10 Learning | Clean quality within two points | Pass versus frozen bank; fail comparative value gate | novelty is -0.35 clean points versus frozen, but -2.67 versus matched online RBF | Re-test only after a geometry revision |
| §10 Learning | Repeatable non-IID gain | Local gain repeats; comparative gate fails | novelty gains +0.75 shifted points versus frozen; online RBF leads novelty by +3.00 | Treat RBF as the scale baseline; do not claim attractor advantage |
| §10 Learning | Consolidation without regression | Not started | — | Three-seed shadow validation after Phase D |
| §13 Build decision | d64/d128, Fashion-MNIST sanity, exact first | Complete for initial path | configs, artifacts, results | Preserve as regression suite |
| §13 Build decision | INT8 cold target, ANN main + exact delta | Delta core complete; ANN/INT8 pending | exact delta implementation | ANN recall first, then calibrated INT8 storage |

## Six-week roadmap mapping

| Roadmap week | Design milestone | Current status |
|---|---|---|
| Week 1 | Compact representation and baseline suite | Complete for Fashion-MNIST |
| Week 2 | Basin store, exact routing, stable hard flow | Complete |
| Week 3 | Soft top-K and ANN main-plus-delta routing | ANN core complete and pushed to 1M on a synthetic bank; latency and recall clear budget through 1M for Euclidean/RBF and through 100k for energy (energy at 1M remains an open research question) |
| Week 4 | Streaming updates and controlled basin creation | Complete; matched RBF control wins and triggers a research pivot |
| Week 5 | Versioned merge/split/prune consolidator | Not started |
| Week 6 | Ablations, continual tests, quantized scale simulation | Static ablations complete; continual and scale simulation pending |

## ImageNet MVP readiness sequence

1. ~~Finish the Phase C real non-IID gate and matched online controls.~~
   Complete; current ManifoldFlow loses to matched online RBF.
2. ~~Add ANN main-index implementation and small-bank exact-recall/latency
   benchmarking for both anisotropic energy and RBF scoring.~~ Complete at
   1k/2k; scale validation now covered by step 5 on a synthetic bank.
3. ~~Add frozen 224px backbone and ImageFolder-style data adapter.~~ Complete,
   smoke-validated on a synthetic ImageFolder, and now validated on real
   ImageNet-1K: a frozen ResNet-18 backbone with a fixed PCA projection
   (`backbone.py`) produced 100,000 real train and 50,000 real validation
   latents with matched-baseline accuracy in a healthy range
   (`first-imagenet-representation.md`). Streamed directly via
   `huggingface_stream.py` rather than through the ImageFolder path, since
   the full 146GB train set can't be persisted locally.
4. ~~Add sharded or memory-mapped latent extraction and restartable state.~~
   Complete: `sharded_latents.py` writes fixed-size resumable shards. A real
   process-kill test (not just a truncated input) later found the manifest
   was only written once at the end, so a genuine kill would have lost
   resumability; fixed to write atomically after every shard flush and
   covered by a test that kills the encoder mid-batch.
5. ~~Pass 10k exact correctness, then 100k ANN scale runs.~~ 100k/1M scale
   runs complete on a synthetic bank (`ann-routing-scale.md`): latency, build
   time, and 95% recall all clear at 100k for both scorers and at 1M for
   Euclidean/RBF (`ef_search=1536`, no graph rebuild needed); energy at 1M
   remains open (93.6% best, diminishing returns from further tuning). Real
   (non-synthetic) 10k correctness is done on two independent banks:
   Fashion-MNIST (`real-10k-ann-routing.md`) and ImageNet
   (`first-imagenet-bank.md`). Euclidean/RBF transfers cleanly on both
   (99.8-99.85% recall). Energy needed ~8x the candidate width on the
   Fashion-MNIST bank but cleared at the synthetic study's original width on
   the ImageNet bank - a bank-specific effect, not a general real-vs-synthetic
   gap as first suspected.
6. ~~Combine representation and retrieval into one real basin bank.~~
   Complete: a 10k-basin bank built from the 100,000 real ImageNet train
   latents (step 3), routed with both scorers, clears 95% recall for each
   (`first-imagenet-bank.md`). Nearest-basin classification accuracy
   (52.75%) trails the matched centroid/k-NN/probe baselines from step 3, as
   expected for 10,000 finer-grained basins versus 1,000 class centroids or a
   100,000-row k-NN reference.
7. ~~Add versioned bank/value/delta persistence (§5's substrate, not its
   merge/split/prune decision logic, which remains Phase D).~~ Complete:
   `checkpoint.py` adds atomic-publish, rollback-capable versioning on top
   of new `serializable()`/`from_serialized()` methods on
   `ExactMainDeltaIndex` and `NoveltyStager` (neither had any before), bundled
   through `OnlineBasinMemory`. Restart verification passed at small
   synthetic scale (`test_checkpoint.py`) and at real 100k-basin scale
   (99,967 basins, 1.2GB checkpoint, bit-identical restart,
   `imagenet-100k-scale-and-persistence.md`).
8. ~~Push the real basin bank from 10k to 100k.~~ Complete, with a caveat: an
   attempt to extend the real train stream from 100,000 to 200,000 rows
   before rebuilding hung silently for ~8 hours. The 100k bank was built from
   the 100,000 rows already in hand instead - a 1:1 basin:example ratio,
   which is degenerate for classification quality but still a legitimate
   real-data test for ANN scale and persistence, both of which passed
   cleanly (99.95% Euclidean recall, 100% energy recall,
   `imagenet-100k-scale-and-persistence.md`).
9. ~~Diagnose the streaming resume hang.~~ Complete, but partial: root-caused
   via bounded repro tests (hard external timeouts, so a repeat hang would
   fail in seconds) to a leaked `multiprocessing` semaphore inside
   `datasets`/`pyarrow` streaming stalling interpreter shutdown - a separate
   OS process invisible to `threading.enumerate()`, not the `skip()`
   mechanism itself, and not the shard/manifest mechanism (data stayed
   intact throughout every failure). Fixed in both streaming scripts:
   `main()` now forces `os._exit()` on both success and exception paths,
   confirmed to exit cleanly with no external kill needed either way;
   `huggingface_stream.with_timeout` separately bounds streaming iteration
   so a genuine fetch stall fails loudly instead of hanging. This fix was
   necessary but not sufficient - see step 10.
10. ~~Extend the real train pull past 100,000 rows.~~ Complete, but not via
    the fixed streaming path: retrying the resume (now failing fast instead
    of hanging) surfaced a second, distinct problem - `HTTP 429`
    rate-limiting on the first train shard, on every resume attempt, since
    `skip()` re-reads every already-consumed shard from scratch each time.
    Repeated resume attempts against this gated dataset's CDN reliably trip
    it. Rather than continue patching the streaming path, built
    `imagenet_local_train_extract.py`: downloads whole shard files with
    `snapshot_download` (no `skip()`, the same robust path already used for
    the validation split) and reads them locally. Reached 500,000 real,
    class-balanced train rows in ~81 minutes with zero failures. Rebuilt the
    10k/100k basin banks from this well-populated set (50:1 and 5:1
    basin:example ratios, real occupancy up to 21/2 median, versus the
    prior 1:1-ratio bank), re-validated ANN routing and persistence against
    them, and corrected step 6's degenerate-bank energy-recall reading in
    the process: that bank's 100% recall at M=32 turned out to be an
    artifact of near-uniform basin rates from near-singleton occupancy, not
    evidence energy routing is easy on real ImageNet geometry - the
    well-populated banks need M=512 (10k) and M=256 (100k) for a similar
    bar (`imagenet-500k-real-data.md`).
11. ~~Run the classification-only ImageNet-1K protocol at real 100k scale,
    routing on Euclidean/RBF.~~ Complete: top-1/top-5, matched-memory VQ and
    RBF baselines, ANN routing recall, latency, and memory all reported
    together against the real 100k bank (`imagenet-scale-evaluation.md`).
    Building it surfaced one more fix - the ImageNet banks had never
    applied the frozen diagonal whitening every Fashion-MNIST bank in this
    project uses before initialization; both were rebuilt on whitened
    latents first. Result: a linear probe wins outright (69.05%/87.10%
    top-1/top-5, cheapest and smallest by three orders of magnitude);
    ManifoldFlow's own bank is the weakest classifier and largest by memory
    of every method tried (60-63% top-1, 1.1GB, mostly the per-basin class
    histogram). The systems mechanism still holds independent of that:
    ANN routing takes the bank's own lookup from 37.5ms (exact) to 1.0ms at
    99.95% recall - bounded compute at scale works, it just does not change
    which method answers better. This does not overturn the standing
    decision from `matched-online-controls.md`; it extends the same
    negative learning-quality finding to real ImageNet-1K with a fuller
    baseline set.
12. A real 1M-basin bank remains untried (only synthetic 1M exists,
    `ann-routing-scale.md`) and would need extending the train pull well
    past 500,000 rows. Promote the attractor formulation only if revised
    geometry clears the matched learning gate and, separately, the energy
    recall ceiling at scale - both still open, on any dataset tested so far.

See [`imagenet-scale-readiness.md`](imagenet-scale-readiness.md) for the
recommended first configuration and distance estimate,
[`results/ann-routing-scale.md`](results/ann-routing-scale.md) for the 10k-1M
synthetic ANN scale results referenced above,
[`results/first-imagenet-representation.md`](results/first-imagenet-representation.md)
for the real-data representation result referenced in steps 3-4,
[`results/real-10k-ann-routing.md`](results/real-10k-ann-routing.md) for the
real Fashion-MNIST bank check,
[`results/first-imagenet-bank.md`](results/first-imagenet-bank.md) and
[`results/imagenet-100k-scale-and-persistence.md`](results/imagenet-100k-scale-and-persistence.md)
for the now-superseded degenerate-ratio 100k bank and the streaming hang's
first (partial) fix, and
[`results/imagenet-500k-real-data.md`](results/imagenet-500k-real-data.md)
for the corrected, well-populated banks, the rate-limit root cause, and the
non-streaming extraction path referenced in steps 9-10, and
[`results/imagenet-scale-evaluation.md`](results/imagenet-scale-evaluation.md)
for the consolidated top-1/top-5/VQ/RBF/routing/latency/memory evaluation
and the whitening fix referenced in step 11.

## Basin traversal and persistent memory (exploratory, beyond the MVP checklist)

Everything above tests single-shot basin routing: one query in, one lookup,
no memory of anything else. That was never the whole intent - design v0.2
§3 gestures at "longer dynamics [that] may reroute between a small number
of segments" and §0 reserves "event streams, sensor monitoring, and
language-state memory" as later domains, closer to basins as a mutable
memory/state layer than a classifier. None of that has been implemented or
tested until now; the MVP's own goals (§1 goal 6, §9-10) do call for the
classification/continual-learning comparisons above, so those results
stand on their own, but they were never meant to be the full story.

- `persistent_context.PersistentBasinContext`: a lightweight routing bias
  layered on top of the existing bank/index (no bank changes - basins stay
  data-agnostic) - a decaying trace of recently visited basin centers blends
  into each new query before routing. `alpha=1.0` reduces to exactly
  today's stateless routing.
- First probe (`basin-traversal-persistence-probe.md`): approximates
  "a truck seen now, then behind a building, then reappearing" using real,
  unseen ImageNet validation latents and class identity as a proxy for
  instance identity (ImageNet has no instance-level labels; no audio/
  language encoder exists in this codebase, so cross-modal claims are out
  of scope for now). Result: a real, graded forgetting curve at
  `alpha=0.5-0.7` (persistence helps at short delays, fades toward the
  stateless baseline as distraction accumulates, slower decay extends it) -
  the qualitative signature the hypothesis requires. Also found a trap
  worth remembering for future sweeps: `alpha=0.3` produces near-ceiling
  re-association with flat, unremarkable accuracy - the query stops
  responding to actual content and just stays pinned near the context,
  which looks like a strong result on one metric while being a collapse
  mode, not evidence of recognition. Stayed ANN-cheap throughout (~0.5ms
  p50), matching the latency this mechanism needs to sit under a larger
  reasoning layer rather than replace one.

This is a first pilot (30 classes, 5 configurations, wide per-cell
variance), not a tuned result. Next steps would be a real `(alpha, decay)`
sweep, a larger/more rigorous stream construction, and - if the visual-only
version keeps holding up - a concrete design for what a second modality
would even mean here before attempting one.
