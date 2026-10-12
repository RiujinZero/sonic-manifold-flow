# Sonic Persistent-State Episodic Training
## Design v0.1

### 1. Purpose

Train Sonic to treat its new persistent-state system as a continuously evolving representation of the current session rather than as a conventional retrieval database.

The training stream will combine:

- synthetic distractor and binding tasks
- natural multi-turn chat such as LMSYS and SmolTalk
- chain-of-thought reasoning, math, code, and structured problem-solving data
- synthetic persistent facts, updates, relations, and unknowns

These sources will be assembled into **long chronological sessions** containing many smaller episodes.

Sonic continues ordinary gradient training on language modeling. Persistent state evolves causally alongside the token stream and survives physical 8K context boundaries.

The core objective is:

\[
\boxed{
\text{events}
\rightarrow
\text{persistent state}
\rightarrow
\text{later retrieval/manipulation}
\rightarrow
\text{language/action}
}
\]

The system should learn that information may remain relevant long after its source tokens have disappeared from the active context.

---

# 2. Primary Goals

The episodic curriculum should force Sonic to learn:

1. **Entity binding**
   - Tim → favorite drink → jasmine
   - Project A → revision → K17

2. **State persistence**
   - information remains accessible over tens of thousands of subsequent tokens

3. **State mutation**
   - old values can be overwritten or modified

4. **Selective updates**
   - observing something does not necessarily change ownership or persistent state

5. **Relational composition**
   - daughter → Mia
   - Mia → piano
   - therefore daughter → piano

6. **Data manipulation**
   - retrieve a stored value and use it in arithmetic or reasoning

7. **Unknown-state handling**
   - absence of a fact must remain distinguishable from a known value of zero

8. **Interference resistance**
   - relevant state survives unrelated chat, code, reasoning, and competing entities

9. **Context/state interchangeability**
   - Sonic should be able to reason over a retrieved persistent-state value approximately as though that value had just appeared in the token context

10. **Eventual event-driven operation**
    - persistent state can update continuously while the expensive LM path is only invoked when an event requires language or higher-level reasoning

---

# 3. Non-Goals

This stage does not attempt to:

- store the entire token history verbatim
- replace Sonic's normal recurrent state
- turn persistent state into a text-only vector database
- require every stored item to be queried later
- teach permanent world knowledge through persistent state
- make every conversational utterance persist indefinitely
- force explicit symbolic representations internally

The persistent system may learn distributed representations. What matters is behaviorally correct persistence, mutation, retrieval, and manipulation.

---

# 4. Core Training Abstraction

Training is organized around three levels:

```text
SESSION
  ├── Episode 1: chat
  ├── Episode 2: distractor/state
  ├── Episode 3: chat
  ├── Episode 4: CoT reasoning
  ├── Episode 5: distractor/state update
  ├── Episode 6: chat
  ├── Episode 7: code/math
  ├── Episode 8: delayed query from Episode 2
  ├── ...
  └── Episode N
```

An **episode** is a local coherent interaction.

A **session** represents one persistent stream of experience containing many episodes.

A physical Sonic training sequence remains:

\[
8192\text{ tokens}
\]

but the logical session should initially contain approximately:

\[
32K\text{ to }128K\text{ tokens}
\]

and may later expand much further.

Therefore:

```text
Logical session

Window 0      Window 1      Window 2      Window 3
0-8191        8192-16383     16384-24575   24576-32767
   │              │              │              │
   └──────── Persistent state remains alive ────┘
```

The **token context resets/slides normally**.

The **persistent state does not reset** until the session ends.

---

# 5. Training State Lifecycle

For session \(S\):

\[
M_0 = \varnothing
\]

Events arrive sequentially:

\[
M_{t+1}=F(M_t,E_t)
\]

where:

- \(M_t\) = current persistent state
- \(E_t\) = current linguistic event or observation
- \(F\) = persistent-state update system

At physical 8K boundaries:

\[
M_{8192^-} \rightarrow M_{8192^+}
\]

without resetting the persistent content.

The computational graph, however, should **not** need to span the entire logical session.

Persistent values survive.

Historical autograd state does not.

Conceptually:

```python
for session in sessions:
    state.reset()

    for window in session.windows:
        outputs = sonic(window.tokens, persistent_state=state)

        loss = language_loss(outputs)
        loss.backward()
        optimizer.step()

        state.detach_graph()
        # Contents survive.
        # Gradient history does not.
```

This gives truncated gradient training without destroying session continuity.

---

# 6. Hidden Session Truth

Each generated session maintains a hidden ground-truth state table used by the dataset generator.

Example:

```python
truth = {
    "user.name": "Tim",
    "user.favorite_drink": "jasmine tea",
    "user.daughter.name": "Mia",
    "Mia.instrument": "piano",
    "project_alpha.revision": "K17",
    "garage.code": 7319,
}
```

This table is **not shown to Sonic**.

It allows the generator to produce arbitrarily delayed questions with exact answers.

It also allows deterministic generation of:

- updates
- deletions
- unknowns
- arithmetic transformations
- relationship queries
- contradiction cases
- negative queries

---

# 7. Episode Classes

## 7.1 Natural Chat Episodes

Sources:

- LMSYS
- SmolTalk
- other existing conversational data

Purpose:

- normal linguistic development
- turn-taking
- topic switching
- conversational continuity
- natural interference between state events
- realistic event density

Example:

```text
User: Have you ever tried making fresh pasta?
Assistant: Yes. The dough is usually...

User: Mine keeps tearing.
Assistant: That can happen if...
```

Natural chat should occupy substantial portions of the logical session without necessarily modifying durable persistent state.

This is important.

The model must not learn:

\[
\text{every utterance} \Rightarrow \text{permanent write}
\]

---

# 8. Synthetic Persistent-Fact Episodes

Insert natural-looking facts into the conversation.

Example:

```text
User: By the way, my name is Tim.
Assistant: Nice to meet you, Tim.
```

Hidden truth becomes:

```python
truth["user.name"] = "Tim"
```

Then continue with unrelated episodes.

Possibly thousands or tens of thousands of tokens later:

```text
User: What was my name again?
Assistant: Tim.
```

The fact may no longer exist anywhere in active context.

---

# 9. Existing Distractor Dataset Conversion

The existing distractor generator is especially valuable because it already removes many shortcuts.

Do **not** discard the current standalone form.

Instead generate two forms.

## 9.1 Local Distractor Episode

Current behavior remains:

```text
Aaron photographed 11 books.
Aaron owns 1 book.
Bruno owns 3 books.
...
Q: How many books does Aaron own?
A: 1
```

This continues training the underlying binding operation.

## 9.2 Distributed Distractor Episode

Separate fact introduction from query.

### Episode A

```text
Aaron photographed 11 books.
Aaron owns 1 book.
Bruno owns 3 books.
```

No question appears.

Then:

```text
[chat]
[math]
[code]
[LMSYS turns]
[another distractor]
[CoT]
[thousands of tokens]
```

### Episode B

```text
User: How many books does Aaron own?
Assistant: 1.
```

Now the task requires persistent-state retrieval.

This transforms the existing distractor curriculum from:

\[
\text{binding}
\]

into:

\[
\text{binding + retention + retrieval}.
\]

---

# 10. State Mutation Episodes

Existing correction and state-update datasets should become first-class persistent-state operations.

Example:

### Initial event

```text
User: I own 6 umbrellas.
```

\[
\text{umbrellas}=6
\]

### Later

```text
User: Actually, make that 10.
```

\[
\text{umbrellas}=10
\]

### Later still

```text
User: Wait, I checked again. It was 6.
```

\[
\text{umbrellas}=6
\]

Then after large amounts of unrelated material:

```text
User: How many umbrellas did I say I had?
Assistant: 6.
```

The required behavior is:

\[
6 \rightarrow 10 \rightarrow 6.
\]

The state system must not retrieve an arbitrary historical value.

---

# 11. Arithmetic State Episodes

These should directly extend Sonic's existing state-update tasks.

Example:

```text
Aaron owns 1 book.
```

Later:

```text
Aaron buys 9 books.
```

Later:

```text
Aaron gives away 7 books.
```

Later:

```text
Aaron buys 1 more.
```

Much later:

```text
How many books does Aaron have?
```

Expected:

\[
1+9-7+1=4.
\]

This forces persistent state to carry a mutable quantity rather than merely an immutable fact.

---

# 12. Observation Versus State Mutation

This should remain heavily represented because Sonic is already beginning to learn this distinction.

Example:

```text
Talia has 3 parrots.
Mei photographed 8 parrots.
Talia spotted 5 parrots.
Mei owns 2 parrots.
```

Query:

```text
How many parrots does Talia have?
```

Answer:

```text
3
```

The model must learn:

\[
\text{observed}(x,n)\neq\text{owns}(x,n).
\]

Use many verbs:

```text
saw
photographed
counted
mentioned
heard about
walked past
borrowed
owns
keeps
bought
sold
lost
gave away
```

Their state effects should differ.

---

# 13. Unknown and Zero Episodes

These are essential for preventing hallucination.

## Unknown

```text
User: My name is Tim.
...
User: What is my sister's name?
Assistant: I don't know.
```

Hidden state:

```python
"user.name" = "Tim"
"user.sister.name" = UNKNOWN
```

## Explicit zero

```text
User: I don't own any dogs.
```

Hidden state:

```python
"user.dogs" = 0
```

Later:

```text
User: How many dogs do I own?
Assistant: None.
```

Sonic must distinguish:

\[
0 \neq \text{unknown}.
\]

---

# 14. Relational Episodes

Persistent state becomes much more useful when independently stored relations can compose.

Example:

### Earlier

```text
User: My daughter is Mia.
```

State:

\[
\text{user.daughter}=\text{Mia}
\]

### Later

```text
User: Mia started piano lessons.
```

State:

\[
\text{Mia.instrument}=\text{piano}
\]

### Much later

```text
User: What instrument is my daughter learning?
```

Required composition:

\[
\text{user.daughter}
\rightarrow
\text{Mia}
\rightarrow
\text{piano}.
\]

Answer:

```text
Piano.
```

This is a major milestone.

It tests whether persistent state behaves as a relational world representation rather than isolated key-value memory.

---

# 15. CoT / Reasoning Episodes

Reasoning datasets should occur naturally inside sessions.

Example:

```text
User: A tank contains 40 liters...
Assistant: ...
```

These provide:

- reasoning pressure
- arithmetic
- data transformation
- temporary variables
- long assistant outputs
- interference for unrelated persistent facts

Most reasoning-specific values should **not automatically become durable session facts**.

This creates useful pressure for selective persistence.

---

# 16. Memory-Dependent CoT

A subset of reasoning problems should explicitly require persistent state.

Earlier in the session:

```text
User: The garage code is 7319.
```

Several episodes later:

```text
User: Take the garage code I told you earlier,
subtract 19, then divide the result by 10.
```

Expected:

\[
(7319-19)/10=730.
\]

This is substantially stronger than simple recall.

It tests:

\[
\text{persistent retrieval}
+
\text{reasoning}
\]

rather than:

\[
\text{persistent retrieval}
\rightarrow
\text{copy}.
\]

---

# 17. Context/State Equivalence

One major target should be:

> Retrieved information should be usable approximately as though the same information had appeared directly in the active context.

Create matched pairs.

### Context version

```text
The code is 7319.
Subtract 19 and divide by 10.
```

### Persistent-state version

The code was introduced 40K tokens earlier.

Current prompt:

```text
Take the code I gave you earlier.
Subtract 19 and divide by 10.
```

Performance difference:

\[
\Delta =
A_{\text{context}}
-
A_{\text{persistent}}
\]

should shrink during training.

This becomes one of the best metrics for whether the persistent state is natively useful to Sonic.

---

# 18. Chat as Interference

LMSYS and SmolTalk are useful for more than conversational training.

They form realistic interference.

Example session:

```text
Memory write
↓
LMSYS conversation
↓
CoT math problem
↓
SmolTalk conversation
↓
binding distractor
↓
code question
↓
LMSYS topic switch
↓
persistent-state update
↓
more chat
↓
query original memory
```

The model should not know which earlier information will become relevant.

---

# 19. Session Construction

Recommended initial logical session:

\[
32K\text{ to }64K
\]

tokens.

After stable learning:

\[
64K\rightarrow128K\rightarrow256K+
\]

without changing Sonic's physical 8K training context.

Example 64K session:

```text
0K       persistent fact A
1K       LMSYS
3K       CoT
5K       distractor
8K       physical boundary
10K      chat
13K      update A
16K      physical boundary
18K      reasoning
22K      persistent fact B
24K      physical boundary
29K      query A
32K      physical boundary
...
51K      query B
...
64K      session reset
```

---

# 20. Query Distance Distribution

Do not use a fixed delay.

Sample logarithmically.

Example target distribution:

| Distance | Initial share |
|---|---:|
| 0-128 tokens | 10% |
| 128-512 | 10% |
| 512-2K | 15% |
| 2K-8K | 20% |
| 8K-32K | 25% |
| 32K-128K | 20% |

Later curriculum:

| Distance | Mature share |
|---|---:|
| <512 | 5% |
| 512-8K | 20% |
| 8K-32K | 30% |
| 32K-128K | 30% |
| >128K | 15% |

The model must not learn a fixed memory lifetime.

---

# 21. Random Placement

Persistent facts and queries must appear anywhere in their respective windows.

Avoid:

```text
memory fact always near start
query always near end
```

Instead:

\[
P(\text{position})\approx\text{uniform}
\]

subject to chronology.

This preserves the anti-shortcut philosophy already used by Sonic's distractor curriculum.

---

# 22. Not Every Fact Gets Queried

This is critical.

If every stored fact eventually receives a question, Sonic can learn:

\[
\text{new fact} \Rightarrow \text{future target}.
\]

Instead:

- some facts are queried once
- some repeatedly
- some updated but never queried
- some never queried at all
- some become relevant only through another relation
- some should decay or lose salience

The persistent state should model the session, not an upcoming-question queue.

---

# 23. Competing Facts

Create deliberately confusable state.

Example:

```text
Tim's favorite color is blue.
Mia's favorite color is green.
Aaron painted his wall blue.
Sarah bought a green bicycle.
Tim saw a red car.
```

Later:

```text
What is Tim's favorite color?
```

Answer:

```text
Blue.
```

Similarity must not be sufficient.

Correct binding is required.

---

# 24. Revisions and Contradictions

Use several kinds.

## Explicit revision

```text
My meeting is at 4.
Actually, it's at 5.
```

## Indirect revision

```text
The meeting got pushed back an hour.
```

## Competing report

```text
Sarah says the meeting is at 6,
but my calendar still says 5.
```

Now the persistent state may need uncertainty or source distinction rather than blind overwrite.

This can become a later curriculum stage.

---

# 25. Episode Mixing

Recommended starting mixture **inside the episodic-session stream**:

| Episode type | Share |
|---|---:|
| Natural chat | 35% |
| Distractor/binding/state | 35% |
| CoT/math/reasoning | 20% |
| Code/structured tasks | 10% |

Within the distractor/state portion:

| State task | Share |
|---|---:|
| Direct binding | 15% |
| Updates/corrections | 20% |
| Arithmetic updates | 15% |
| Unknown/zero | 15% |
| observation vs ownership | 15% |
| relational composition | 10% |
| adversarial/conflicting state | 10% |

These percentages are starting points, not hard architecture requirements.

---

# 26. Integration With Existing Pretraining

Do not immediately replace Sonic's entire corpus with episodic sessions.

Introduce an **episodic stream** alongside normal broad pretraining.

Initial recommendation:

\[
15\%-25\%
\]

of token throughput from stateful episodic sessions.

Remaining training continues to provide broad:

- educational text
- code
- mathematics
- general language
- existing chat
- existing reasoning

If persistent-state utilization improves without degrading language modeling, increase toward:

\[
30\%-40\%.
\]

The existing raw distractor stream can gradually migrate into the episodic generator.

---

# 27. Session Identity

Each session must have a hard isolation boundary.

```python
persistent_state.reset(session_id)
```

Nothing from session \(A\) should leak into session \(B\).

Create tests specifically designed to catch leakage:

Session A:

```text
My code is 4821.
```

Session B:

```text
What is my code?
```

Expected:

```text
I don't know.
```

Not:

```text
4821.
```

---

# 28. Dataset Schema

Suggested record structure:

```json
{
  "session_id": "sess_004812",
  "episode_id": "ep_017",
  "episode_type": "persistent_update",
  "events": [
    {
      "role": "user",
      "text": "Actually, the prototype is revision K19 now."
    },
    {
      "role": "assistant",
      "text": "Got it."
    }
  ],
  "hidden_state_ops": [
    {
      "op": "update",
      "key": "prototype.revision",
      "old": "K17",
      "new": "K19"
    }
  ],
  "future_queries": [
    {
      "distance_target_tokens": 18000,
      "key": "prototype.revision"
    }
  ]
}
```

`hidden_state_ops` are dataset metadata.

They do not need to appear in Sonic's token stream.

---

# 29. Session Generator

Conceptually:

```python
def build_session(target_tokens):
    truth = {}
    episodes = []

    while token_count(episodes) < target_tokens:
        kind = sample_episode_type()

        if kind == "chat":
            ep = sample_chat_episode()

        elif kind == "cot":
            ep = sample_reasoning_episode()

        elif kind == "state_write":
            ep = generate_state_write(truth)

        elif kind == "state_update":
            ep = generate_state_update(truth)

        elif kind == "distractor":
            ep = generate_distractor_episode(truth)

        elif kind == "delayed_query":
            ep = generate_due_query(truth)

        episodes.append(ep)

    return Session(
        episodes=episodes,
        hidden_truth=truth
    )
```

Query scheduling should be independent of physical 8K boundaries.

---

# 30. Physical Packing

Once a logical session exists, tokenize it and cut sequentially:

```python
tokens = tokenize(session)

windows = [
    tokens[0:8192],
    tokens[8192:16384],
    tokens[16384:24576],
    ...
]
```

Do **not** independently reshuffle these windows.

They must execute chronologically.

State survives:

```text
window N
   ↓
persistent state
   ↓
window N+1
```

If multiple sessions are trained concurrently in a batch, each batch lane needs its own independent persistent-state instance.

---

# 31. Batch Scheduling

For batch size \(B\):

```text
lane 0 = session 184, window 3
lane 1 = session 927, window 7
lane 2 = session 511, window 1
...
```

At the next physical training step:

```text
lane 0 = session 184, window 4
lane 1 = session 927, window 8
lane 2 = session 511, window 2
```

When a session finishes, that lane receives a new session and its persistent state resets.

This allows ordinary batched GPU training while maintaining chronological state.

---

# 32. Session Shuffle Strategy

Shuffle **sessions**, not windows.

Correct:

```text
Session 5
Session 812
Session 93
```

Within Session 5:

```text
window 0 → 1 → 2 → 3 → ...
```

Incorrect:

```text
Session5.window3
Session5.window0
Session5.window7
```

Chronology is part of the supervision.

---

# 33. State Update Timing

Persistent state should update causally.

Sonic must never gain access to future facts.

For each event:

```text
tokens arrive
↓
state observes event
↓
state may update
↓
later tokens may retrieve changed state
```

Never preprocess the entire session into state before training its earlier windows.

That would leak answers backward in time.

---

# 34. Existing Distractor Difficulty Should Remain

The current 8K randomized-placement scheme has been useful precisely because:

- target position cannot be memorized
- many variables coexist
- distractor verbs matter
- values vary
- unknown is possible
- nearby values compete
- the query appears after interference

Do not simplify this when adding persistent state.

Instead, add another axis:

\[
\text{time since observation}.
\]

The resulting task becomes:

\[
\text{correct semantic relation}
+
\text{correct entity}
+
\text{correct state mutation}
+
\text{long-term persistence}.
\]

---

# 35. Curriculum

## Phase A: State Bootstrap

Goal:

> Sonic learns that persistent state exists and is useful.

Session length:

\[
16K-32K.
\]

High proportion of explicit writes:

```text
Remember that...
My name is...
The code is...
```

Most queries within 1-2 windows.

Use simple direct retrieval and updates.

---

## Phase B: Natural Persistence

Session length:

\[
32K-64K.
\]

Reduce explicit "remember" language.

Use natural statements:

```text
I use Celsius.
My daughter is Mia.
The server is revision K17.
```

Increase unrelated chat and CoT between write and query.

---

## Phase C: Manipulation

Require retrieved values to enter computation.

Examples:

- arithmetic
- comparison
- relational traversal
- temporal ordering
- correction
- aggregation

Session length:

\[
64K-128K.
\]

---

## Phase D: Adversarial Persistence

Add:

- competing entities
- similar names
- conflicting observations
- irrelevant retrieved candidates
- false statements attributed to others
- explicit unknowns
- zero values
- topic switches
- long delays

---

## Phase E: Long-Lived World State

Logical sessions:

\[
128K-1M+\text{ tokens}
\]

while physical Sonic context remains 8K or whatever current training context is selected.

At this stage persistent state becomes responsible for information that cannot possibly exist in active context.

---

# 36. State Utilization Ablations

Every evaluation should have several modes.

## Normal

Persistent state is intact.

## State Disabled

Same model, same prompt, no persistent state.

## State Shuffled

Give the state from another session.

Expected:

\[
A_{\text{normal}}
\gg
A_{\text{shuffled}}
\]

## State Corrupted

Inject a competing incorrect memory.

This tests whether Sonic blindly trusts retrieved information.

## Current-Context Duplicate

Provide the relevant fact both in state and directly in context.

This tests interaction between sources.

---

# 37. Core Metrics

Track separately:

### Retrieval accuracy

Exact stored fact retrieval.

### Binding accuracy

Correct entity + attribute + value.

### Value accuracy

Correct value regardless of lexical entity decoding.

This is particularly useful given Sonic's current wrong-name/right-value behavior.

### Entity accuracy

Correct lexical entity regardless of value.

### Update accuracy

Current value after one or more mutations.

### Unknown accuracy

Correct refusal when state lacks the requested fact.

### Zero accuracy

Correct distinction between zero and unknown.

### Relational accuracy

Correct multi-hop state composition.

### State-computation accuracy

Correct reasoning using retrieved state.

### Retention curve

Accuracy versus token distance.

---

# 38. Retention Curve

Measure:

\[
A(d)
\]

where \(d\) is tokens elapsed since the information was introduced.

Suggested bins:

```text
0-512
512-2K
2K-8K
8K-32K
32K-128K
128K-512K
>512K
```

This becomes one of the signature metrics for persistent-state Sonic.

---

# 39. Event-Driven Runtime Alignment

Production architecture eventually operates roughly as:

```text
           ┌─────────────────────┐
events ───▶│ Persistent State    │
           │ 10-20 Hz initially  │
           └──────────┬──────────┘
                      │
                event/salience
                      │
                 wake required?
                  /         \
                no           yes
                │             │
         continue state       ▼
             loop          Sonic LM
                               │
                         language/action
```

Training does not need to literally simulate wall-clock 20 Hz initially.

Instead, language events serve as discrete state updates.

Later, no-op/state-only ticks can be inserted to train decay and temporal dynamics.

---

# 40. Event Gate Training

Eventually label events according to whether the LM should wake.

Examples:

### State-only

```text
background observation
minor update
known repeated fact
low-salience event
```

### Wake LM

```text
direct user question
novel anomaly
timer completion
important contradiction
explicit request
reasoning-required state transition
```

The persistent loop can remain cheap while Sonic's full language machinery remains event-driven.

This should be a later stage after state utilization itself works reliably.

---

# 41. Success Criteria

The first major success milestone is not general conversational quality.

It is:

### Test A

```text
Session token 1,200:
User: My name is Tim.

[>8K tokens and at least one physical context boundary]

Session token 24,000:
User: What is my name?

Sonic: Tim.
```

### Test B

```text
Early:
User: My name is Tim.

Later:
User: Actually, call me Timothy.

Much later:
User: What is my name?

Sonic: Timothy.
```

### Test C

```text
User: What is my sister's name?

Sonic: I don't know.
```

### Test D

```text
Earlier:
garage.code = 7319

Much later:
Take the garage code, subtract 19, divide by 10.

Sonic: 730.
```

### Test E

```text
Earlier:
daughter = Mia

Later:
Mia learns piano

Much later:
What instrument does my daughter play?

Sonic: Piano.
```

Passing all five demonstrates:

- persistence
- mutation
- epistemic boundaries
- computation over memory
- relational composition

---

# 42. Why This Fits Sonic

Sonic's existing curriculum already pressures it toward:

- entity binding
- variable tracking
- state updates
- correction handling
- observation versus possession
- ontology traversal
- unknown values
- conversational state

The new persistent-state system should therefore not require Sonic to learn an entirely foreign abstraction.

Instead it extends:

\[
\text{state across tokens}
\]

into:

\[
\text{state across contexts}
\]

and eventually:

\[
\text{state across continuous experience}.
\]

The existing distractor curriculum becomes the bridge between ordinary sequence modeling and persistent world-state operation.

---

# 43. Recommended First Implementation

Do not begin with the million-token version.

Start with a narrow proof.

### Logical sessions

32K tokens.

### Physical context

8192 tokens.

### Four sequential physical windows per session

\[
4\times8192=32768.
\]

### Episode mix

- 35% natural LMSYS/SmolTalk chat
- 35% existing distractor/state tasks
- 20% CoT/math/reasoning
- 10% code/other structured tasks

### Persistent targets per session

Approximately:

- 6-12 simple writes
- 2-4 updates
- 2 unknown queries
- 1 explicit-zero query
- 2 arithmetic state manipulations
- 1 relational composition
- several facts never queried

### Retrieval placement

At least half of persistent queries should occur **after the source fact has left the active 8K context**.

That condition is critical.

---

# 44. First Experimental Comparison

Train three otherwise identical short runs.

## A. Stateless baseline

No persistent state.

## B. Persistent state, local-only queries

Memory system exists, but facts are queried within the same 8K window.

## C. Persistent state, cross-window episodic training

Facts and queries can span logical windows.

Then evaluate all three on:

```text
within-context binding
cross-8K retrieval
cross-32K retrieval
state updates
unknowns
retrieval + arithmetic
relational composition
```

Expected result if the design works:

\[
C > B > A
\]

on long-horizon tasks while:

\[
C \approx B \approx A
\]

or better on normal language quality.

That would demonstrate that persistent-state training adds capability rather than merely changing where Sonic stores ordinary context information.

---

# 45. Central Design Principle

The dataset should never teach Sonic:

> “Here is a memory. Repeat it later.”

It should teach:

> **“This is one continuous world. Events alter that world. Most events are irrelevant to any particular future query. Maintain useful state anyway, revise it when reality changes, and consult it when an event requires reasoning or language.”**

That is the bridge from a language model with memory to the intended Sonic architecture:

\[
\boxed{
\text{continuous cheap persistent state}
+
\text{event-driven sparse LM}
}
\]

The language model becomes one consumer and modifier of a persistent world representation rather than the sole place where that world exists.