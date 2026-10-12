"""Train Gemma's query projector and latent port on oracle simulated state."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.gemma_world import (
    GemmaQueryProjector,
    enable_upper_layer_lora,
    load_gemma_world_model,
)
from manifoldflow.world_synthetic_training import (
    OracleWorldBatch,
    collate_oracle_examples,
    load_oracle_examples,
)
from manifoldflow.world_training import (
    answer_mismatched_indices,
    combine_world_losses,
    masked_candidate_retrieval_loss,
    memory_use_contrast,
)

_DEFAULT_RESPONSE_INSTRUCTION = (
    "Use only the supplied world memory. Answer with one short factual sentence. "
    "If the evidence is insufficient, say there is not enough evidence."
)


def _tokenize(
    tokenizer: Any,
    questions: tuple[str, ...],
    answers: tuple[str, ...] | None,
    *,
    device: torch.device,
    max_length: int,
    prompt_format: str,
    response_instruction: str | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    question_texts = [
        (
            f"{response_instruction}\n\nQuestion: {question}"
            if response_instruction
            else question
        )
        for question in questions
    ]
    if prompt_format == "chat":
        conversations = [
            [{"role": "user", "content": question}] for question in question_texts
        ]
        prompts = [
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]
        texts = prompts
        if answers is not None:
            texts = [
                tokenizer.apply_chat_template(
                    [*conversation, {"role": "assistant", "content": answer}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for conversation, answer in zip(conversations, answers, strict=True)
            ]
        add_special_tokens = False
    elif prompt_format == "plain":
        prompts = [f"Question: {question}\nAnswer:" for question in question_texts]
        texts = (
            prompts
            if answers is None
            else [
                f"{prompt} {answer}"
                for prompt, answer in zip(prompts, answers, strict=True)
            ]
        )
        add_special_tokens = True
    else:
        raise ValueError("prompt_format must be chat or plain")
    encoded = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if answers is None:
        return input_ids, attention_mask, None
    prompt_lengths = [
        len(
            tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        )
        for prompt in prompts
    ]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for row, length in enumerate(prompt_lengths):
        labels[row, : min(length, labels.shape[1])] = -100
    return input_ids, attention_mask, labels


def _oracle_weights(batch: OracleWorldBatch, dtype: torch.dtype) -> Tensor:
    weights = batch.evidence_mask.to(dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return weights / denominator


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    examples = load_oracle_examples(args.dataset, "train")
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    world_model = load_gemma_world_model(args.model, device=args.device)
    query_projector = GemmaQueryProjector(
        hidden_dimension=world_model.port.hidden_dimension,
        key_dimension=128,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if str(initial["model"]) != args.model:
            raise ValueError("initial checkpoint uses a different base model")
        load_result = world_model.port.load_state_dict(initial["port"], strict=False)
        query_projector.load_state_dict(initial["query_projector"])
        print(f"initialized adapters from {args.init_checkpoint}", flush=True)
        if load_result.missing_keys:
            print(
                f"new adapter parameters: {', '.join(load_result.missing_keys)}",
                flush=True,
            )
    if args.enable_lora:
        enable_upper_layer_lora(world_model, rank=args.lora_rank, alpha=args.lora_alpha)
        if args.init_checkpoint and initial.get("lora") is not None:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(world_model.model, initial["lora"])
        world_model.model.train()
    world_model.port.train()
    if args.freeze_query_projector:
        query_projector.requires_grad_(False)
        query_projector.eval()
    else:
        query_projector.train()
    trainable_model_parameters = [
        parameter
        for parameter in world_model.model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            *world_model.port.parameters(),
            *(
                parameter
                for parameter in query_projector.parameters()
                if parameter.requires_grad
            ),
            *trainable_model_parameters,
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    order = list(range(len(examples)))
    cursor = len(order)
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        chosen = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = collate_oracle_examples([examples[index] for index in chosen])
        batch = OracleWorldBatch(
            batch.questions,
            batch.answers,
            batch.entity_ids,
            batch.appearance_keys.to(device),
            batch.state_values.to(device),
            batch.memory_mask.to(device),
            batch.evidence_mask.to(device),
        )
        input_ids, attention_mask, labels = _tokenize(
            tokenizer,
            batch.questions,
            batch.answers,
            device=device,
            max_length=args.max_length,
            prompt_format=args.prompt_format,
            response_instruction=args.response_instruction,
        )
        question_ids, question_mask, _ = _tokenize(
            tokenizer,
            batch.questions,
            None,
            device=device,
            max_length=args.max_length,
            prompt_format=args.prompt_format,
            response_instruction=None,
        )
        with torch.no_grad():
            question_hidden = world_model.encode_query(question_ids, question_mask)
        query_keys = query_projector(question_hidden)
        retrieval_loss = masked_candidate_retrieval_loss(
            query_keys,
            batch.appearance_keys,
            batch.memory_mask,
            batch.evidence_mask,
            temperature=args.temperature,
        )
        weights = _oracle_weights(
            batch, world_model.port.value_projection[0].weight.dtype
        )
        correct = world_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            memory_values=batch.state_values,
            memory_keys=batch.appearance_keys,
            memory_mask=batch.evidence_mask,
            retrieval_weights=weights,
        ).loss
        negative_indices = answer_mismatched_indices(batch.answers).to(device)
        shuffled_values = batch.state_values.index_select(0, negative_indices)
        shuffled_keys = batch.appearance_keys.index_select(0, negative_indices)
        shuffled = world_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            memory_values=shuffled_values,
            memory_keys=shuffled_keys,
            memory_mask=batch.evidence_mask,
            retrieval_weights=weights,
        ).loss
        contrast = memory_use_contrast(correct, shuffled, margin=args.contrast_margin)
        loss = combine_world_losses(
            correct,
            retrieval_loss,
            contrast,
            retrieval_weight=args.retrieval_weight,
            contrast_weight=args.contrast_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            [*world_model.port.parameters(), *query_projector.parameters()], 1.0
        )
        optimizer.step()
        row = {
            "step": step,
            "total": float(loss.total.detach()),
            "answer": float(correct.detach()),
            "retrieval": float(retrieval_loss.detach()),
            "contrast": float(contrast.detach()),
            "port_gate": float(torch.sigmoid(world_model.port.gate.detach())),
        }
        history.append(row)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(row), flush=True)

    lora_state = None
    if args.enable_lora:
        from peft import get_peft_model_state_dict

        lora_state = get_peft_model_state_dict(world_model.model)
    torch.save(
        {
            "format_version": 1,
            "state_schema": "world-state-v2-live-compatible",
            "model": args.model,
            "port": world_model.port.state_dict(),
            "query_projector": query_projector.state_dict(),
            "lora": lora_state,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
        },
        output,
    )
    print(f"saved oracle adapter checkpoint to {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/world-sim")
    parser.add_argument("--output", default="checkpoints/world/oracle-port.pt")
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--prompt-format", choices=("chat", "plain"), default="chat")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--freeze-query-projector", action="store_true")
    parser.add_argument("--response-instruction", default=_DEFAULT_RESPONSE_INSTRUCTION)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-weight", type=float, default=0.25)
    parser.add_argument("--contrast-weight", type=float, default=0.10)
    parser.add_argument("--contrast-margin", type=float, default=0.4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
