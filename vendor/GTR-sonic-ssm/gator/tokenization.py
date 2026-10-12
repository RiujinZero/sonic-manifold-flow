"""Small, dependency-free tokenization for early Gator language experiments."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable

from .data import Conversation, DialogueTurn


_PIECE_PATTERN = re.compile(r"\s+|[\w]+|[^\w\s]+", re.UNICODE)


class CompactTokenizer:
    """Learn frequent text pieces and fall back losslessly to UTF-8 bytes."""

    FORMAT_V1 = "gator-compact-v1"
    FORMAT_V2 = "gator-compact-v2"
    SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<mask>", "<user>", "<assistant>")
    BLANK_TOKEN = "<blank>"
    PAD_ID, BOS_ID, EOS_ID, MASK_ID, USER_ID, ASSISTANT_ID = range(6)
    BYTE_OFFSET = len(SPECIAL_TOKENS)
    BYTE_COUNT = 256
    LEARNED_OFFSET = BYTE_OFFSET + BYTE_COUNT

    def __init__(
        self,
        learned_pieces: Iterable[str],
        *,
        format_version: str = FORMAT_V2,
    ) -> None:
        if format_version not in {self.FORMAT_V1, self.FORMAT_V2}:
            raise ValueError(f"unsupported tokenizer format: {format_version!r}")
        pieces = tuple(learned_pieces)
        if len(set(pieces)) != len(pieces):
            raise ValueError("learned tokenizer pieces must be unique")
        self.format_version = format_version
        self.learned_pieces = pieces
        self._piece_to_id = {
            piece: self.LEARNED_OFFSET + index for index, piece in enumerate(pieces)
        }

    @property
    def has_blank(self) -> bool:
        return self.format_version == self.FORMAT_V2

    @property
    def blank_token_id(self) -> int | None:
        if not self.has_blank:
            return None
        return self.LEARNED_OFFSET + len(self.learned_pieces)

    @property
    def learned_end(self) -> int:
        """Exclusive end of learned text-piece IDs (before the V2 BLANK row)."""

        return self.LEARNED_OFFSET + len(self.learned_pieces)

    @property
    def vocab_size(self) -> int:
        return self.learned_end + int(self.has_blank)

    @classmethod
    def train(
        cls,
        conversations: Iterable[Conversation],
        *,
        vocab_size: int = 2048,
        format_version: str = FORMAT_V2,
    ) -> "CompactTokenizer":
        if format_version not in {cls.FORMAT_V1, cls.FORMAT_V2}:
            raise ValueError(f"unsupported tokenizer format: {format_version!r}")
        reserved_blank = int(format_version == cls.FORMAT_V2)
        minimum_size = cls.LEARNED_OFFSET + reserved_blank
        if vocab_size < minimum_size:
            raise ValueError(f"vocab_size must be at least {minimum_size}")
        counts: Counter[str] = Counter()
        for conversation in conversations:
            for turn in conversation.turns:
                counts.update(_PIECE_PATTERN.findall(turn.text))
        # V2 leaves the final requested vocabulary row for its instance-specific
        # BLANK. All V1 structural, byte, and learned-piece IDs remain stable.
        capacity = vocab_size - cls.LEARNED_OFFSET - reserved_blank
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if format_version == cls.FORMAT_V2 and len(ranked) < capacity:
            raise ValueError(
                "training corpus has only "
                f"{len(ranked)} unique pieces, but vocab_size={vocab_size} "
                f"requires {capacity}; lower vocab_size to preserve the fixed "
                "V2 vocabulary/BLANK-row contract"
            )
        return cls(
            (piece for piece, _ in ranked[:capacity]),
            format_version=format_version,
        )

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        for piece in _PIECE_PATTERN.findall(text):
            token_id = self._piece_to_id.get(piece)
            if token_id is not None:
                result.append(token_id)
            else:
                result.extend(self.BYTE_OFFSET + byte for byte in piece.encode("utf-8"))
        return result

    def encode_turns(self, turns: Iterable[DialogueTurn], *, bos: bool = True) -> list[int]:
        result = [self.BOS_ID] if bos else []
        for turn in turns:
            result.append(self.USER_ID if turn.role == "user" else self.ASSISTANT_ID)
            result.extend(self.encode(turn.text))
        return result

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        show_roles: bool = True,
        diagnostic: bool = False,
    ) -> str:
        """Decode IDs, optionally exposing otherwise non-rendering structure.

        Normal decoding preserves the V1 behavior: EOS terminates rendering,
        PAD/BOS/MASK do not render, and role markers use readable labels.
        V2 BLANK never renders normally. ``diagnostic=True`` prints structural
        token names and continues beyond EOS so latent tails can be inspected.
        """

        output: list[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                output.append(pending_bytes.decode("utf-8", errors="replace"))
                pending_bytes.clear()

        role_text = {self.USER_ID: "USER: ", self.ASSISTANT_ID: "ASSISTANT: "}
        for token_id in token_ids:
            token_id = int(token_id)
            if self.BYTE_OFFSET <= token_id < self.LEARNED_OFFSET:
                pending_bytes.append(token_id - self.BYTE_OFFSET)
                continue
            # In normal rendering BLANK is absent, not a byte-stream boundary.
            # Keeping pending bytes intact lets a UTF-8 fallback character span
            # multiple physical slots separated by latent BLANK emissions.
            if (
                not diagnostic
                and self.has_blank
                and token_id == self.blank_token_id
            ):
                continue
            flush_bytes()
            if self.LEARNED_OFFSET <= token_id < self.learned_end:
                index = token_id - self.LEARNED_OFFSET
                output.append(self.learned_pieces[index])
            elif self.has_blank and token_id == self.blank_token_id:
                if diagnostic:
                    output.append(self.BLANK_TOKEN)
            elif diagnostic and 0 <= token_id < len(self.SPECIAL_TOKENS):
                output.append(self.SPECIAL_TOKENS[token_id])
            elif show_roles and token_id in role_text:
                output.append(role_text[token_id])
            elif token_id == self.EOS_ID:
                break
            elif token_id in {self.PAD_ID, self.BOS_ID, self.MASK_ID}:
                continue
            elif not show_roles and token_id in role_text:
                continue
            else:
                raise ValueError(f"unknown special token ID: {token_id}")
        flush_bytes()
        return "".join(output)

    def to_dict(self) -> dict:
        return {
            "format": self.format_version,
            "learned_pieces": list(self.learned_pieces),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "CompactTokenizer":
        format_version = state.get("format")
        if format_version not in {cls.FORMAT_V1, cls.FORMAT_V2}:
            raise ValueError("unsupported tokenizer format")
        return cls(state["learned_pieces"], format_version=format_version)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CompactTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
