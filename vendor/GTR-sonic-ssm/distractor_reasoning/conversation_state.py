"""Conversation-State Generator: make a response depend on which topic
was actually established earlier in the conversation, not on the surface
words of the final turn alone.

    User: My pizza sauce came out strange.
    Assistant: What was wrong with it? Did the crust turn out okay at least?
    User: Way too sweet.
    Assistant: Sounds like that's way too sweet for the pizza sauce.

    User: I tried the new coffee shop.
    Assistant: How was the drink? Was the pastry case any good?
    User: Way too sweet.
    Assistant: Sounds like that's way too sweet for the coffee.

The final user utterance is identical in both conversations; the correct
reply is not. Unlike every symbolic-world generator in this package, there
is no computable fact to resolve here -- topic-binding is inherently a
pragmatic/referential skill, so this generator draws from a small
hand-authored bank of "callback groups": several distinct topics that
plausibly share the same ambiguous follow-up lines. At generation time,
one topic from a group is picked as the one *actually* established in
this conversation (optionally preceded by 1-2 *abandoned* topics from the
same group, each explicitly dropped via "Actually, never mind that --"),
and the shared callback is asked. This gives combinatorial variety (topics
x callbacks x abandoned-topic orderings) from a fixed, tractable amount of
authored content, mirroring how every other generator in this package
gets combinatorial variety from a fixed vocabulary rather than infinite
authored content.

Each topic's clarifying question also ends by mentioning one *trailing*
noun unrelated to the real topic (e.g. "the crust" for a pizza-sauce
conversation) -- the nearest noun before the callback, and the module's
first cheating strategy to guard against.

Train and eval draw from entirely disjoint callback groups (different
topics, different callback phrasing), the same held-out discipline used
throughout this package.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .topic_switch import Turn, render_turns

GENERIC_ANSWER = "Thanks for sharing that!"


@dataclass(frozen=True)
class TopicScenario:
    key: str
    opening: str
    clarifying_question: str
    anchor_phrase: str
    trailing_noun_phrase: str


@dataclass(frozen=True)
class CallbackGroup:
    key: str
    topics: tuple[TopicScenario, ...]
    callbacks: tuple[str, ...]
    reactions: dict[str, str]           # callback -> topic-agnostic reaction clause
    generic_reactions: dict[str, str]   # callback -> fully topic-blind reply


def resolve_conversation_state(group: CallbackGroup, current_topic: TopicScenario, callback: str) -> str:
    return f"Sounds like {group.reactions[callback]} for {current_topic.anchor_phrase}."


def _lead_lowercase(sentence: str) -> str:
    # The pronoun "I" stays capitalized regardless of sentence position --
    # every topic opening in this module happens to start with "I ", so a
    # blind lowercase of the first character would render "i ordered a
    # package..." mid-conversation.
    if sentence.startswith("I "):
        return sentence
    return sentence[0].lower() + sentence[1:] if sentence else sentence


TRAIN_GROUPS: tuple[CallbackGroup, ...] = (
    CallbackGroup(
        key="taste_reaction",
        topics=(
            TopicScenario("pizza_sauce", "My pizza sauce came out strange.",
                           "What was wrong with it? Did the crust turn out okay at least?",
                           "the pizza sauce", "the crust"),
            TopicScenario("coffee", "I tried the new coffee shop.",
                           "How was the drink? Was the pastry case any good?",
                           "the coffee", "the pastry case"),
            TopicScenario("soup", "I made soup for the first time.",
                           "How did it turn out? Did the bread you served with it work well?",
                           "the soup", "the bread"),
        ),
        callbacks=("Way too sweet.", "Way too salty.", "It was really bland."),
        reactions={
            "Way too sweet.": "that's way too sweet",
            "Way too salty.": "that's way too salty",
            "It was really bland.": "that's pretty bland",
        },
        generic_reactions={
            "Way too sweet.": "Sounds like the sweetness was overwhelming.",
            "Way too salty.": "Sounds like the saltiness was overwhelming.",
            "It was really bland.": "Sounds like it needed more flavor.",
        },
    ),
    CallbackGroup(
        key="price_reaction",
        topics=(
            TopicScenario("new_phone", "I finally bought that new phone.",
                           "How much did it end up costing? Did you get a case for it too?",
                           "the new phone", "the case"),
            TopicScenario("concert_tickets", "I got tickets to that concert.",
                           "What was the price like? Did you get parking sorted too?",
                           "the concert tickets", "the parking"),
            TopicScenario("gym_membership", "I signed up for a gym membership.",
                           "How much was it? Did they throw in a locker too?",
                           "the gym membership", "the locker"),
        ),
        callbacks=("It was way more expensive than I expected.", "Actually pretty cheap.", "About what I thought it'd cost."),
        reactions={
            "It was way more expensive than I expected.": "that's way more than expected",
            "Actually pretty cheap.": "that's a pretty good deal",
            "About what I thought it'd cost.": "that's about right then",
        },
        generic_reactions={
            "It was way more expensive than I expected.": "Sounds like the price caught you off guard.",
            "Actually pretty cheap.": "Sounds like a good deal.",
            "About what I thought it'd cost.": "Sounds about right then.",
        },
    ),
    CallbackGroup(
        key="timing_reaction",
        topics=(
            TopicScenario("flight", "My flight got rescheduled.",
                           "When's it leaving now? Did they say anything about the connecting flight?",
                           "the flight", "the connecting flight"),
            TopicScenario("doctor_appointment", "I finally got a doctor's appointment.",
                           "When is it? Did they mention anything about the follow-up visit?",
                           "the appointment", "the follow-up visit"),
            TopicScenario("package_delivery", "I ordered a package last week.",
                           "When's it supposed to arrive? Did they say anything about the tracking number?",
                           "the delivery", "the tracking number"),
        ),
        callbacks=("It got delayed again.", "It showed up right on time.", "Still no updates on it."),
        reactions={
            "It got delayed again.": "that's frustrating, another delay",
            "It showed up right on time.": "that's great, right on schedule",
            "Still no updates on it.": "that's annoying, still no news",
        },
        generic_reactions={
            "It got delayed again.": "Delays are always frustrating.",
            "It showed up right on time.": "Good to hear it worked out.",
            "Still no updates on it.": "That's frustrating to be left waiting.",
        },
    ),
    CallbackGroup(
        key="difficulty_reaction",
        topics=(
            TopicScenario("math_homework", "I started my math homework.",
                           "How's it going? Did the teacher mention anything about the group project?",
                           "the math homework", "the group project"),
            TopicScenario("job_interview", "I had that job interview today.",
                           "How did it go? Did they mention anything about the salary range?",
                           "the interview", "the salary range"),
            TopicScenario("hiking_trail", "I tried that new hiking trail.",
                           "How was it? Did you run into the ranger station along the way?",
                           "the trail", "the ranger station"),
        ),
        callbacks=("It was way harder than expected.", "Actually pretty easy.", "Took way longer than planned."),
        reactions={
            "It was way harder than expected.": "that's tougher than expected",
            "Actually pretty easy.": "that's easier than expected",
            "Took way longer than planned.": "that took longer than planned",
        },
        generic_reactions={
            "It was way harder than expected.": "That sounds more difficult than expected.",
            "Actually pretty easy.": "Good to hear it wasn't too bad.",
            "Took way longer than planned.": "That sounds like it dragged on.",
        },
    ),
    CallbackGroup(
        key="quality_reaction",
        topics=(
            TopicScenario("new_book", "I started reading that new book.",
                           "How is it so far? Did the cover art look as good in person?",
                           "the book", "the cover art"),
            TopicScenario("tv_show", "I started watching that new show.",
                           "How is it so far? Did the theme song grow on you?",
                           "the show", "the theme song"),
            TopicScenario("phone_case", "I bought that phone case everyone was talking about.",
                           "How is it? Did the color match the photos online?",
                           "the phone case", "the color"),
        ),
        callbacks=("It wasn't as good as people said.", "Really exceeded expectations.", "Kind of a letdown."),
        reactions={
            "It wasn't as good as people said.": "that's disappointing given the hype",
            "Really exceeded expectations.": "that's great, better than hoped",
            "Kind of a letdown.": "that's a shame, a bit of a letdown",
        },
        generic_reactions={
            "It wasn't as good as people said.": "That's disappointing to hear.",
            "Really exceeded expectations.": "That's great to hear!",
            "Kind of a letdown.": "That's a shame.",
        },
    ),
)

EVAL_GROUPS: tuple[CallbackGroup, ...] = (
    CallbackGroup(
        key="mood_reaction",
        topics=(
            TopicScenario("new_job", "I started my new job this week.",
                           "How's it going? Did you get along with the new manager?",
                           "the new job", "the new manager"),
            TopicScenario("moving_to_new_city", "I finally moved to the new city.",
                           "How's it going? Did the new apartment work out?",
                           "the move", "the new apartment"),
            TopicScenario("starting_a_diet", "I started that new diet.",
                           "How's it going? Did the meal plan turn out to be doable?",
                           "the diet", "the meal plan"),
        ),
        callbacks=("It's going better than I thought.", "Honestly kind of stressful.", "I'm still getting used to it."),
        reactions={
            "It's going better than I thought.": "that's great, better than expected",
            "Honestly kind of stressful.": "that sounds stressful",
            "I'm still getting used to it.": "that makes sense, these things take time",
        },
        generic_reactions={
            "It's going better than I thought.": "That's great to hear!",
            "Honestly kind of stressful.": "That sounds like a lot to deal with.",
            "I'm still getting used to it.": "These things take time.",
        },
    ),
    CallbackGroup(
        key="recommendation_reaction",
        topics=(
            TopicScenario("restaurant", "I tried that new restaurant downtown.",
                           "How was it? Did the service hold up?",
                           "the restaurant", "the service"),
            TopicScenario("workout_class", "I tried that workout class.",
                           "How was it? Did the instructor seem good?",
                           "the class", "the instructor"),
            TopicScenario("online_course", "I signed up for that online course.",
                           "How is it? Did the video quality hold up?",
                           "the course", "the video quality"),
        ),
        callbacks=("I'd definitely recommend it.", "Wouldn't go back honestly.", "It's decent, nothing special."),
        reactions={
            "I'd definitely recommend it.": "that's a strong recommendation",
            "Wouldn't go back honestly.": "that's a shame it didn't work out",
            "It's decent, nothing special.": "that sounds fairly average",
        },
        generic_reactions={
            "I'd definitely recommend it.": "Good to hear it was worth it!",
            "Wouldn't go back honestly.": "Sorry to hear it didn't work out.",
            "It's decent, nothing special.": "Sounds like a middle-of-the-road experience.",
        },
    ),
)


def groups_for(split: str) -> tuple[CallbackGroup, ...]:
    if split == "train":
        return TRAIN_GROUPS
    if split == "eval":
        return EVAL_GROUPS
    raise ValueError(f"unknown split: {split!r}")


# -- Difficulty curriculum -------------------------------------------------

@dataclass(frozen=True)
class ConversationStateDifficultyConfig:
    level: int
    num_topics: int  # topics mentioned total; the last one is "current"


CONVERSATION_STATE_DIFFICULTY_LEVELS: dict[int, ConversationStateDifficultyConfig] = {
    1: ConversationStateDifficultyConfig(level=1, num_topics=1),
    2: ConversationStateDifficultyConfig(level=2, num_topics=2),
    3: ConversationStateDifficultyConfig(level=3, num_topics=3),
}


def conversation_state_config_for_level(level: int) -> ConversationStateDifficultyConfig:
    if level not in CONVERSATION_STATE_DIFFICULTY_LEVELS:
        raise ValueError(f"conversation-state difficulty level must be in 1..3, got {level}")
    return CONVERSATION_STATE_DIFFICULTY_LEVELS[level]


# -- Example assembly ---------------------------------------------------

@dataclass
class ConversationStateExample:
    turns: list[Turn]
    prompt: str
    answer_text: str
    difficulty: int
    group_key: str
    num_topics: int
    current_topic_key: str
    shortcut_predictions: dict[str, str]


def generate_conversation_state_example(
    rng: random.Random, level: int, split: str = "train",
) -> ConversationStateExample:
    config = conversation_state_config_for_level(level)
    group = rng.choice(groups_for(split))
    callback = rng.choice(group.callbacks)
    num_topics = min(config.num_topics, len(group.topics))
    topics = rng.sample(group.topics, k=num_topics)

    turns: list[Turn] = []
    for i, topic in enumerate(topics):
        if i == 0:
            turns.append(Turn("user", topic.opening))
        else:
            turns.append(Turn("user", f"Actually, never mind that -- {_lead_lowercase(topic.opening)}"))
        turns.append(Turn("assistant", topic.clarifying_question))
    turns.append(Turn("user", callback))

    current_topic = topics[-1]
    answer_text = resolve_conversation_state(group, current_topic, callback)
    prompt = render_turns(turns) + "\nAssistant:"

    shortcut_predictions: dict[str, str] = {
        "current_turn_only": group.generic_reactions[callback],
        "generic_answer": GENERIC_ANSWER,
        "last_noun_heuristic": (
            f"Sounds like {group.reactions[callback]} for {current_topic.trailing_noun_phrase}."
        ),
    }
    if len(topics) >= 2:
        first_topic = topics[0]
        shortcut_predictions["topic_word_persistence"] = (
            f"Sounds like {group.reactions[callback]} for {first_topic.anchor_phrase}."
        )

    turns_with_answer = turns + [Turn("assistant", answer_text)]

    return ConversationStateExample(
        turns=turns_with_answer,
        prompt=prompt,
        answer_text=answer_text,
        difficulty=level,
        group_key=group.key,
        num_topics=num_topics,
        current_topic_key=current_topic.key,
        shortcut_predictions=shortcut_predictions,
    )
