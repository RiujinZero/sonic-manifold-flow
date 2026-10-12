"""Pillow renderer with automatic detection labels for symbolic episodes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from manifoldflow.world_simulator import SimEntity, SimEpisode, SimFrame


@dataclass(frozen=True)
class RenderConfig:
    width: int = 640
    height: int = 360
    domain_randomization: bool = True
    jpeg_quality: int = 88

    def __post_init__(self) -> None:
        if self.width < 160 or self.height < 90:
            raise ValueError("render dimensions are too small")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True)
class RenderedFrame:
    image: Any
    annotations: tuple[dict[str, object], ...]


class TabletopRenderer:
    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    def render(self, episode: SimEpisode, frame: SimFrame) -> RenderedFrame:
        try:
            from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Rendering requires Pillow") from error
        rng = random.Random(episode.seed * 10_007 + frame.frame_index)
        background = self._background(rng)
        image = Image.new("RGB", (self.config.width, self.config.height), background)
        draw = ImageDraw.Draw(image)
        self._draw_table(draw, rng)
        ordered = sorted(
            frame.visible_entities,
            key=lambda item: {"ball": 0, "toy_car": 1, "cup": 2, "occluder_box": 3}.get(
                item.class_name, 1
            ),
        )
        annotations: list[dict[str, object]] = []
        for entity in ordered:
            pixels = self._pixels(entity.box_xyxy)
            self._shadow(draw, pixels)
            self._draw_entity(draw, entity, pixels)
            x1, y1, x2, y2 = pixels
            annotations.append(
                {
                    "entity_id": entity.entity_id,
                    "tracker_id": entity.tracker_id,
                    "class_name": entity.class_name,
                    "label": entity.label,
                    "bbox_xywh": (x1, y1, x2 - x1, y2 - y1),
                    "bbox_xyxy_normalized": entity.box_xyxy,
                    "visibility": entity.visibility.value,
                }
            )
        if self.config.domain_randomization:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
            if rng.random() < 0.18:
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
        return RenderedFrame(image, tuple(annotations))

    def _background(self, rng: random.Random) -> tuple[int, int, int]:
        if not self.config.domain_randomization:
            return (205, 183, 145)
        bases = ((205, 183, 145), (171, 151, 122), (213, 207, 192), (128, 143, 134))
        base = rng.choice(bases)
        return tuple(
            max(0, min(255, channel + rng.randint(-12, 12))) for channel in base
        )

    def _draw_table(self, draw: Any, rng: random.Random) -> None:
        width, height = self.config.width, self.config.height
        line = (95, 78, 61)
        draw.line(
            (0, int(height * 0.24), width, int(height * 0.24)), fill=line, width=2
        )
        if self.config.domain_randomization:
            for _ in range(14):
                y = rng.randrange(height)
                grain = tuple(min(255, value + rng.randint(15, 35)) for value in line)
                draw.line((0, y, width, y + rng.randint(-3, 3)), fill=grain, width=1)

    def _draw_entity(
        self, draw: Any, entity: SimEntity, pixels: tuple[int, int, int, int]
    ) -> None:
        x1, y1, x2, y2 = pixels
        outline = tuple(max(0, value - 45) for value in entity.color)
        if entity.class_name == "ball":
            draw.ellipse(pixels, fill=entity.color, outline=outline, width=3)
        elif entity.class_name == "toy_car":
            draw.rounded_rectangle(
                pixels,
                radius=max(4, (y2 - y1) // 5),
                fill=entity.color,
                outline=outline,
                width=3,
            )
            wheel = max(3, (y2 - y1) // 8)
            for x in (x1 + wheel * 2, x2 - wheel * 2):
                draw.ellipse(
                    (x - wheel, y2 - wheel, x + wheel, y2 + wheel), fill=(28, 31, 34)
                )
        elif entity.class_name == "cup":
            inset = max(3, (x2 - x1) // 8)
            draw.polygon(
                ((x1, y1), (x2, y1), (x2 - inset, y2), (x1 + inset, y2)),
                fill=entity.color,
                outline=outline,
            )
            draw.ellipse(
                (x1, y1 - 3, x2, y1 + 6), fill=entity.color, outline=outline, width=2
            )
        else:
            draw.rectangle(pixels, fill=entity.color, outline=outline, width=3)
            for offset in range(8, max(9, x2 - x1), 16):
                draw.line((x1 + offset, y1, x1 + offset, y2), fill=outline, width=1)

    @staticmethod
    def _shadow(draw: Any, pixels: tuple[int, int, int, int]) -> None:
        x1, _, x2, y2 = pixels
        draw.ellipse((x1 + 5, y2 - 2, x2 + 10, y2 + 10), fill=(56, 51, 48))

    def _pixels(
        self, box: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            round(x1 * self.config.width),
            round(y1 * self.config.height),
            round(x2 * self.config.width),
            round(y2 * self.config.height),
        )
