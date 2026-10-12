from manifoldflow.world_renderer import RenderConfig, TabletopRenderer
from manifoldflow.world_simulator import (
    ScenarioKind,
    SimulatorConfig,
    TabletopSimulator,
)


def test_renderer_produces_pixels_and_only_visible_annotations() -> None:
    episode = TabletopSimulator(SimulatorConfig(ambiguous_probability=0.0)).generate(
        ScenarioKind.CUP_SHUFFLE, seed=9
    )
    frame = next(item for item in episode.frames if item.relations)
    rendered = TabletopRenderer(
        RenderConfig(width=320, height=180, domain_randomization=False)
    ).render(episode, frame)

    assert rendered.image.size == (320, 180)
    classes = [item["class_name"] for item in rendered.annotations]
    assert "ball" not in classes
    assert classes.count("cup") == 3


def test_renderer_is_deterministic_for_episode_and_frame() -> None:
    episode = TabletopSimulator().generate(ScenarioKind.CAR_OCCLUSION, seed=17)
    renderer = TabletopRenderer(RenderConfig(width=160, height=90))

    first = renderer.render(episode, episode.frames[0]).image.tobytes()
    second = renderer.render(episode, episode.frames[0]).image.tobytes()
    assert first == second
