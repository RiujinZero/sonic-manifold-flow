"""Expands the vendored `distractor_reasoning` generator's vocabulary and
quantity range for the production training recipe, without modifying the
vendored file itself (kept "vendored verbatim... a re-copy stays a clean
diff" per this project's existing convention).

The stock generator has 20 train names / 8 eval names and 12 train nouns /
5 eval nouns (`distractor_reasoning/vocab.py`), and quantities are always
1-12 (`correction.py`'s `_quantity()` and its inline `initial_value =
rng.randint(1, 12)`). Fine for probe-scale ablations; too narrow for a
production recipe meant to teach the encoder to handle real diversity
rather than memorize a closed handful of entities.

Two independent mechanisms:

- **Names/nouns**: monkey-patch the *module-level* constants
  `distractor_reasoning.vocab.TRAIN_NAMES`/`EVAL_NAMES`/`TRAIN_NOUNS`/
  `EVAL_NOUNS` in place. `nouns_for(split)`/`names_for(split)` read these
  as bare module globals at call time (not a value captured at import
  time), so reassigning them here before any generation happens is
  sufficient - no source-level patch needed.
- **Quantities**: `_quantity()`'s 1-12 range and `generate_correction_example`'s
  inline `initial_value = rng.randint(1, 12)` are both literal constants
  baked into function bodies - not monkey-patchable by reassigning a
  module global. Instead, `expand_quantity_range` wraps
  `generate_correction_example` itself: call the real generator, then
  remap every distinct quantity value in the returned example to a fresh
  random 4-digit value via one bijective per-call map (so a repeated
  original value - e.g. two different statements colliding on the same
  quantity, which the correction-chain logic uses deliberately for the
  ambiguity checks throughout this project's probes - still collides
  after remapping, just at a different, larger value)."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))

_NAME_ONSETS = (
    "b", "br", "c", "ch", "d", "dr", "f", "fr", "g", "gr", "h", "j", "k",
    "kl", "l", "m", "n", "p", "pr", "r", "s", "sh", "st", "t", "tr", "v",
    "w", "z",
)
_NAME_VOWELS = ("a", "e", "i", "o", "u", "ai", "ie", "oa", "ou")
_NAME_CODAS = ("", "n", "r", "s", "l", "m", "th", "nd", "rt", "ck")

_NOUN_PLURAL_EXCEPTIONS: dict[str, str] = {
    "wolf": "wolves", "shelf": "shelves", "knife": "knives", "leaf": "leaves",
    "mouse": "mice", "goose": "geese", "child": "children",
}


def _pluralize(word: str) -> str:
    if word in _NOUN_PLURAL_EXCEPTIONS:
        return _NOUN_PLURAL_EXCEPTIONS[word]
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _generate_names(count: int, *, seed: int) -> list[str]:
    generator = random.Random(seed)
    names: set[str] = set()
    while len(names) < count:
        onset = generator.choice(_NAME_ONSETS)
        vowel1 = generator.choice(_NAME_VOWELS)
        middle = generator.choice(_NAME_ONSETS)
        vowel2 = generator.choice(_NAME_VOWELS)
        coda = generator.choice(_NAME_CODAS)
        name = (onset + vowel1 + middle + vowel2 + coda).capitalize()
        if 3 <= len(name) <= 10:
            names.add(name)
    return sorted(names)


# Real, concrete, correctly-pluralizable English nouns spanning many
# categories - not "thousands" (no wordlist dependency was available to
# reach that honestly with real words; a synthetic-token approach was
# rejected for nouns specifically since Sonic's tokenizer/embeddings were
# trained on real English text, and nonsense tokens would hurt rather than
# help the "real diversity" goal), but a genuine ~15-25x expansion over the
# stock 12/5 split, across enough categories that no single category
# dominates.
_NOUN_BASE: tuple[tuple[str, tuple[str, ...]], ...] = (
    # pets / animals
    ("parrot", ("bird", "pet", "animal")), ("cat", ("mammal", "pet", "animal")),
    ("dog", ("mammal", "pet", "animal")), ("rabbit", ("mammal", "pet", "animal")),
    ("sparrow", ("bird", "animal")), ("pigeon", ("bird", "animal")),
    ("lizard", ("reptile", "pet", "animal")), ("hamster", ("mammal", "pet", "animal")),
    ("turtle", ("reptile", "pet", "animal")), ("falcon", ("bird", "animal")),
    ("gecko", ("reptile", "pet", "animal")), ("ferret", ("mammal", "pet", "animal")),
    ("canary", ("bird", "pet", "animal")), ("goldfish", ("fish", "pet", "animal")),
    ("chinchilla", ("mammal", "pet", "animal")), ("iguana", ("reptile", "pet", "animal")),
    ("finch", ("bird", "animal")), ("tortoise", ("reptile", "pet", "animal")),
    ("guppy", ("fish", "pet", "animal")), ("cockatiel", ("bird", "pet", "animal")),
    ("newt", ("reptile", "animal")), ("owl", ("bird", "animal")),
    ("hedgehog", ("mammal", "pet", "animal")), ("gerbil", ("mammal", "pet", "animal")),
    # vehicles
    ("bicycle", ("vehicle", "possession")), ("car", ("vehicle", "possession")),
    ("scooter", ("vehicle", "possession")), ("motorcycle", ("vehicle", "possession")),
    ("skateboard", ("vehicle", "possession")), ("canoe", ("vehicle", "possession")),
    ("kayak", ("vehicle", "possession")), ("wagon", ("vehicle", "possession")),
    ("trailer", ("vehicle", "possession")), ("golf cart", ("vehicle", "possession")),
    # household / furniture
    ("umbrella", ("possession",)), ("book", ("possession",)),
    ("backpack", ("possession",)), ("lamp", ("possession", "furniture")),
    ("chair", ("possession", "furniture")), ("table", ("possession", "furniture")),
    ("shelf", ("possession", "furniture")), ("cushion", ("possession", "furniture")),
    ("rug", ("possession", "furniture")), ("mirror", ("possession", "furniture")),
    ("blanket", ("possession",)), ("pillow", ("possession",)),
    ("candle", ("possession",)), ("vase", ("possession",)),
    ("clock", ("possession",)), ("basket", ("possession",)),
    # electronics
    ("laptop", ("possession", "electronics")), ("tablet", ("possession", "electronics")),
    ("camera", ("possession", "electronics")), ("headphone", ("possession", "electronics")),
    ("speaker", ("possession", "electronics")), ("monitor", ("possession", "electronics")),
    ("keyboard", ("possession", "electronics")), ("charger", ("possession", "electronics")),
    ("router", ("possession", "electronics")), ("drone", ("possession", "electronics")),
    ("printer", ("possession", "electronics")), ("projector", ("possession", "electronics")),
    # tools
    ("hammer", ("possession", "tool")), ("wrench", ("possession", "tool")),
    ("screwdriver", ("possession", "tool")), ("drill", ("possession", "tool")),
    ("ladder", ("possession", "tool")), ("toolbox", ("possession", "tool")),
    ("flashlight", ("possession", "tool")), ("shovel", ("possession", "tool")),
    ("rake", ("possession", "tool")), ("saw", ("possession", "tool")),
    # clothing
    ("jacket", ("possession", "clothing")), ("sweater", ("possession", "clothing")),
    ("scarf", ("possession", "clothing")), ("hat", ("possession", "clothing")),
    ("glove", ("possession", "clothing")), ("boot", ("possession", "clothing")),
    ("belt", ("possession", "clothing")), ("backpack strap", ("possession", "clothing")),
    # sports / instruments
    ("guitar", ("possession", "instrument")), ("violin", ("possession", "instrument")),
    ("drum", ("possession", "instrument")), ("trumpet", ("possession", "instrument")),
    ("flute", ("possession", "instrument")), ("keyboard piano", ("possession", "instrument")),
    ("basketball", ("possession", "sports")), ("tennis racket", ("possession", "sports")),
    ("skateboard deck", ("possession", "sports")), ("baseball glove", ("possession", "sports")),
    ("soccer ball", ("possession", "sports")), ("kite", ("possession", "sports")),
    # kitchen / food-adjacent objects
    ("mug", ("possession", "kitchen")), ("kettle", ("possession", "kitchen")),
    ("blender", ("possession", "kitchen")), ("toaster", ("possession", "kitchen")),
    ("cutting board", ("possession", "kitchen")), ("frying pan", ("possession", "kitchen")),
    ("thermos", ("possession", "kitchen")), ("lunchbox", ("possession", "kitchen")),
    # office / misc
    ("notebook", ("possession", "office")), ("pen", ("possession", "office")),
    ("stapler", ("possession", "office")), ("folder", ("possession", "office")),
    ("calculator", ("possession", "office")), ("whiteboard", ("possession", "office")),
    ("map", ("possession",)), ("compass", ("possession",)),
    ("telescope", ("possession",)), ("binoculars", ("possession",)),
)


def _build_noun_pool() -> dict[str, Any]:
    from distractor_reasoning.vocab import Noun

    pool: dict[str, Noun] = {}
    for singular, categories in _NOUN_BASE:
        pool[singular] = Noun(singular, _pluralize(singular), categories)
    return pool


def expand_vocabulary(
    *, name_pool_size: int = 3000, eval_name_fraction: float = 0.1, seed: int = 29
) -> None:
    """Monkey-patches `distractor_reasoning.vocab`'s module-level name/noun
    constants. Call once, before any session generation - idempotent
    (re-running just regenerates the same pools given the same seed)."""
    import distractor_reasoning.vocab as vocab_module

    names = _generate_names(name_pool_size, seed=seed)
    split_index = int(len(names) * (1 - eval_name_fraction))
    vocab_module.TRAIN_NAMES = tuple(names[:split_index])
    vocab_module.EVAL_NAMES = tuple(names[split_index:])

    noun_pool = _build_noun_pool()
    noun_items = list(noun_pool.items())
    generator = random.Random(seed + 1)
    generator.shuffle(noun_items)
    eval_count = max(1, int(len(noun_items) * eval_name_fraction))
    vocab_module.TRAIN_NOUNS = dict(noun_items[eval_count:])
    vocab_module.EVAL_NOUNS = dict(noun_items[:eval_count])
    print(
        f"[vocab_expansion] names: {len(vocab_module.TRAIN_NAMES)} train / "
        f"{len(vocab_module.EVAL_NAMES)} eval; nouns: "
        f"{len(vocab_module.TRAIN_NOUNS)} train / {len(vocab_module.EVAL_NOUNS)} eval",
        flush=True,
    )


def expand_quantity_range(*, max_quantity: int = 9999) -> None:
    """Monkey-patches `distractor_reasoning.correction.generate_correction_example`
    to remap every quantity in the returned example to a fresh random value
    up to `max_quantity` (4 digits by default) - a bijective per-call
    remap, so statements that originally shared a quantity (deliberately,
    for the ambiguity-check logic several probes rely on) still share one
    after remapping, just a different, larger one."""
    import distractor_reasoning.correction as correction_module
    from dataclasses import replace

    if getattr(correction_module, "_manifoldflow_quantity_expanded", False):
        return  # already patched - avoid double-wrapping on repeated calls

    _original = correction_module.generate_correction_example

    def _remapped(rng: random.Random, level: int, split: str = "train", style: str = "qa"):
        example = _original(rng, level, split=split, style=style)
        original_values = {statement.quantity for statement in example.statements}
        value_map = {
            original: rng.randint(1, max_quantity) for original in original_values
        }
        # Guard against an accidental collision the remap itself introduces
        # between two originally-distinct values (rare at 9999 possible
        # values, but would silently defeat an ambiguity check if it
        # happened).
        while len(set(value_map.values())) != len(value_map):
            value_map = {
                original: rng.randint(1, max_quantity) for original in original_values
            }
        example.statements = [
            replace(statement, quantity=value_map[statement.quantity])
            for statement in example.statements
        ]
        if isinstance(example.answer_value, int):
            example.answer_value = value_map[example.answer_value]
        return example

    correction_module.generate_correction_example = _remapped
    correction_module._manifoldflow_quantity_expanded = True
    print(f"[vocab_expansion] quantities: up to {max_quantity} (4-digit range)", flush=True)
