from manifoldflow.world_evaluation import (
    answer_candidates_for_question,
    world_fact_match,
)


def test_world_fact_match_accepts_grounded_paraphrases() -> None:
    assert world_fact_match(
        "The ball is visible in the middle right zone.", "Middle right."
    )
    assert world_fact_match(
        "The ball is under the yellow cup.", "It is in the yellow cup."
    )
    assert world_fact_match("No.", "No, the car is currently hidden.")
    assert world_fact_match(
        "There is not enough evidence to identify a cup.",
        "I cannot determine which cup from the evidence.",
    )


def test_world_fact_match_rejects_wrong_world_facts() -> None:
    assert not world_fact_match(
        "The car is visible in the middle left zone.", "Middle right."
    )
    assert not world_fact_match("The ball is under the yellow cup.", "The red cup.")
    assert not world_fact_match("No.", "Yes, it is visible.")


def test_answer_candidates_are_filtered_by_question_intent() -> None:
    candidates = (
        "Yes.",
        "No.",
        "The car is visible in the left zone.",
        "The ball is under the yellow cup.",
        "The yellow cup is in the right zone.",
    )

    assert answer_candidates_for_question("Is the car in view?", candidates) == (
        "Yes.",
        "No.",
    )
    assert answer_candidates_for_question("Where is the yellow cup?", candidates) == (
        "The yellow cup is in the right zone.",
    )


def test_plain_ball_location_question_can_express_containment_or_unknown() -> None:
    # A plain "where is the ball" question cannot assume the ball is
    # currently visible, so the closed-world decoder must be able to choose
    # containment or "not enough evidence" too, not only a visible zone.
    candidates = (
        "The ball is visible in the left zone.",
        "The ball is visible in the center zone.",
        "The ball is under the red cup.",
        "There is not enough evidence to identify a cup.",
        "The car is visible in the left zone.",
    )

    selected = answer_candidates_for_question("Where is the ball?", candidates)

    assert set(selected) == {
        "The ball is visible in the left zone.",
        "The ball is visible in the center zone.",
        "The ball is under the red cup.",
        "There is not enough evidence to identify a cup.",
    }
