from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .data import ChatPair, load_pairs, select_eval_prompts
from .infer import generate_text, load_checkpoint_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tiny chat model with BLEU and human score template")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-context-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--bleu-max-n", type=int, default=4)
    parser.add_argument("--human-template", default="human_scoring_template.csv")
    return parser.parse_args()


def _pair_prompt(pair: ChatPair) -> str:
    if pair.context_turns:
        if pair.context:
            return "\n".join(pair.context_turns + [pair.context])
        return "\n".join(pair.context_turns)
    return pair.context


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _compute_sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    # character-level n-gram BLEU for portability and no external dependency
    ref_tokens = [ch for ch in reference]
    hyp_tokens = [ch for ch in hypothesis]
    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(_ngrams(ref_tokens, n))
        hyp_ng = Counter(_ngrams(hyp_tokens, n))
        total = sum(hyp_ng.values())
        clipped = 0
        for token, count in hyp_ng.items():
            clipped += min(count, ref_ng.get(token, 0))

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    if any(p == 0.0 for p in precisions):
        return 0.0

    geometric = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    bp = 1.0
    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len / max(hyp_len, 1))

    return bp * geometric


def _compute_corpus_bleu(predictions: List[str], references: List[str], max_n: int = 4) -> float:
    if not predictions:
        return 0.0
    scores = [_compute_sentence_bleu(ref, pred, max_n=max_n) for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(
    checkpoint: str,
    eval_data: str,
    output_dir: str,
    max_samples: int = 20,
    max_context_turns: int = 4,
    max_new_tokens: int = 24,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    bleu_max_n: int = 4,
    human_template: str = "human_scoring_template.csv",
) -> Dict[str, Any]:
    model, tokenizer, cfg = load_checkpoint_bundle(checkpoint)
    pairs = load_pairs(eval_data, max_context_turns=max_context_turns)

    selected_prompts = select_eval_prompts(pairs, max_prompts=max_samples)
    prompts = selected_prompts
    references: List[str] = []
    predictions: List[str] = []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    human_path = out_dir / human_template

    if pred_path.exists():
        pred_path.unlink()
    if metrics_path.exists():
        metrics_path.unlink()

    with pred_path.open("a", encoding="utf-8") as pred_f:
        for sample_index, pair in enumerate(pairs):
            if sample_index >= max_samples:
                break

            prompt = prompts[sample_index] if sample_index < len(prompts) else _pair_prompt(pair)
            reference = pair.reply

            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            pred = result["response"]
            references.append(reference)
            predictions.append(pred)

            result.update({
                "sample_id": sample_index,
                "reference": reference,
            })
            pred_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    bleu = _compute_corpus_bleu(predictions, references, max_n=bleu_max_n)

    samples = []
    for i in range(len(predictions)):
        prompt = prompts[i] if i < len(prompts) else _pair_prompt(pairs[i])
        samples.append(
            {
                "sample_id": i,
                "prompt": prompt,
                "reference": references[i],
                "prediction": predictions[i],
                "bleu": _compute_sentence_bleu(references[i], predictions[i], max_n=bleu_max_n),
            }
        )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": checkpoint,
                "num_samples": len(predictions),
                "max_context_turns": max_context_turns,
                "max_new_tokens": max_new_tokens,
                "bleu": bleu,
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with human_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "reference", "prediction", "human_score", "comment"])
        for i in range(len(predictions)):
            prompt = prompts[i] if i < len(prompts) else _pair_prompt(pairs[i])
            writer.writerow([prompt, references[i], predictions[i], "", ""])

    return {
        "predictions": str(pred_path),
        "metrics": str(metrics_path),
        "human_template": str(human_path),
        "num_samples": len(predictions),
        "bleu": bleu,
        "max_context_turns": max_context_turns,
        "config": cfg,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(
        checkpoint=args.checkpoint,
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_context_turns=args.max_context_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        bleu_max_n=args.bleu_max_n,
        human_template=args.human_template,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
