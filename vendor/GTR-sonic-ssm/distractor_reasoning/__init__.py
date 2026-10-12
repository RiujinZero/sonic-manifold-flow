"""Distractor Reasoning Example Generator.

Generates unlimited synthetic SFT examples that train small language
models to preserve factual (subject, relation, object) bindings, ignore
irrelevant information, aggregate compatible facts across a category
hierarchy, and recognize when a question cannot be answered from the
supplied context.

Every example is built as structured data first (a ``World`` of ``Fact``s
plus a ``Query``) and only rendered to natural language afterward, so the
gold answer is always computed from the symbolic representation, never
guessed from surface text. See ``generate_example`` for the main entry
point, ``ExampleQueue`` (``dataset.py``) for training integration, and
``evaluate.py`` for the held-out evaluation harness.
"""

from .counterfactual import (
    ALL_AXES,
    COUNTERFACTUAL_DIFFICULTY_LEVELS,
    CounterfactualExample,
    generate_counterfactual_example,
)
from .counterfactual_evaluate import (
    evaluate_counterfactual_model,
    generate_counterfactual_eval_examples,
)
from .conversation_state import (
    CONVERSATION_STATE_DIFFICULTY_LEVELS,
    ConversationStateDifficultyConfig,
    ConversationStateExample,
    generate_conversation_state_example,
)
from .conversation_state_evaluate import (
    evaluate_conversation_state_model,
    generate_conversation_state_eval_examples,
)
from .correction import (
    CORRECTION_DIFFICULTY_LEVELS,
    CorrectionDifficultyConfig,
    CorrectionExample,
    Statement,
    generate_correction_example,
    resolve_correction_query,
)
from .correction_evaluate import evaluate_correction_model, generate_correction_eval_examples
from .correction_shortcuts import CORRECTION_SHORTCUT_NAMES, correction_shortcut_predictions
from .dataset import ExampleQueue
from .difficulty import DIFFICULTY_LEVELS, DifficultyConfig
from .evaluate import evaluate_model, generate_eval_examples
from .generate import Example, generate_example
from .instruction_selection import (
    INSTRUCTION_DIFFICULTY_LEVELS,
    OPERATIONS,
    InstructionDifficultyConfig,
    InstructionExample,
    generate_instruction_example,
)
from .instruction_selection_evaluate import (
    evaluate_instruction_model,
    generate_instruction_eval_examples,
)
from .instruction_selection_shortcuts import (
    INSTRUCTION_SHORTCUT_NAMES,
    instruction_shortcut_predictions,
)
from .ontology import (
    ONTOLOGY_DIFFICULTY_LEVELS,
    OntologyDifficultyConfig,
    OntologyExample,
    Taxonomy,
    generate_ontology_example,
)
from .ontology_evaluate import evaluate_ontology_model, generate_ontology_eval_examples
from .ontology_shortcuts import ONTOLOGY_SHORTCUT_NAMES, ontology_shortcut_predictions
from .query import Query, resolve_query
from .shortcuts import SHORTCUT_NAMES, shortcut_predictions
from .state_evaluate import evaluate_state_model, generate_state_eval_examples
from .state_shortcuts import STATE_SHORTCUT_NAMES, state_shortcut_predictions
from .state_update import (
    STATE_DIFFICULTY_LEVELS,
    Event,
    StateDifficultyConfig,
    StateExample,
    StateQuery,
    generate_state_update_example,
    resolve_state_query,
)
from .topic_switch import (
    TOPIC_SWITCH_DIFFICULTY_LEVELS,
    Turn,
    TopicSwitchDifficultyConfig,
    TopicSwitchExample,
    TurnProbe,
    generate_topic_switch_example,
    render_turns,
)
from .topic_switch_evaluate import evaluate_topic_switch_model, generate_topic_switch_eval_examples
from .world import Fact, World

__all__ = [
    "ALL_AXES",
    "CONVERSATION_STATE_DIFFICULTY_LEVELS",
    "CORRECTION_DIFFICULTY_LEVELS",
    "CORRECTION_SHORTCUT_NAMES",
    "COUNTERFACTUAL_DIFFICULTY_LEVELS",
    "DIFFICULTY_LEVELS",
    "INSTRUCTION_DIFFICULTY_LEVELS",
    "INSTRUCTION_SHORTCUT_NAMES",
    "ONTOLOGY_DIFFICULTY_LEVELS",
    "ONTOLOGY_SHORTCUT_NAMES",
    "OPERATIONS",
    "SHORTCUT_NAMES",
    "STATE_DIFFICULTY_LEVELS",
    "STATE_SHORTCUT_NAMES",
    "TOPIC_SWITCH_DIFFICULTY_LEVELS",
    "ConversationStateDifficultyConfig",
    "ConversationStateExample",
    "CorrectionDifficultyConfig",
    "CorrectionExample",
    "CounterfactualExample",
    "DifficultyConfig",
    "Event",
    "Example",
    "ExampleQueue",
    "Fact",
    "InstructionDifficultyConfig",
    "InstructionExample",
    "OntologyDifficultyConfig",
    "OntologyExample",
    "Query",
    "Statement",
    "StateDifficultyConfig",
    "StateExample",
    "StateQuery",
    "Taxonomy",
    "TopicSwitchDifficultyConfig",
    "TopicSwitchExample",
    "Turn",
    "TurnProbe",
    "World",
    "correction_shortcut_predictions",
    "evaluate_conversation_state_model",
    "evaluate_correction_model",
    "evaluate_counterfactual_model",
    "evaluate_instruction_model",
    "evaluate_model",
    "evaluate_ontology_model",
    "evaluate_state_model",
    "evaluate_topic_switch_model",
    "generate_conversation_state_eval_examples",
    "generate_conversation_state_example",
    "generate_correction_eval_examples",
    "generate_correction_example",
    "generate_counterfactual_eval_examples",
    "generate_counterfactual_example",
    "generate_eval_examples",
    "generate_example",
    "generate_instruction_eval_examples",
    "generate_instruction_example",
    "generate_ontology_eval_examples",
    "generate_ontology_example",
    "generate_state_eval_examples",
    "generate_state_update_example",
    "generate_topic_switch_eval_examples",
    "generate_topic_switch_example",
    "instruction_shortcut_predictions",
    "ontology_shortcut_predictions",
    "render_turns",
    "resolve_correction_query",
    "resolve_query",
    "resolve_state_query",
    "shortcut_predictions",
    "state_shortcut_predictions",
]
