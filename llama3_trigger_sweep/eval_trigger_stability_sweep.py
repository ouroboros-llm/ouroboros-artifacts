import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate backdoor trigger stability by progressively modifying 0..N words "
            "in a single instruction sample, then computing hit rates."
        )
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "hf"],
    )
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--train-json", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--min-k", type=int, default=0)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument(
        "--variants-per-k",
        type=int,
        default=1,
        help="Number of different modified prompts per k (only used in random mode).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "first"],
        help="random: replace random k words; first: replace first k words (deterministic).",
    )
    parser.add_argument(
        "--replace-with",
        type=str,
        default="foo",
        help="Replacement token for modified words.",
    )
    parser.add_argument("--repeats-per-prompt", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--output-prompts-json",
        type=str,
        default=None,
        help="Optional path to dump all modified prompts with metadata.",
    )
    return parser.parse_args()


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_instruction_sample(path: str, index: int) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Train JSON should be a non-empty list.")
    if index < 0 or index >= len(data):
        raise IndexError(f"sample-index out of range: {index}")
    ex = data[index]
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    prompt = instr + ("\n" + inp if inp else "")
    if not prompt:
        raise ValueError("Empty prompt after combining instruction and input.")
    if not out:
        raise ValueError("Empty output for selected sample.")
    return prompt, out


def split_tokens(text: str) -> Tuple[List[str], List[int]]:
    tokens = re.findall(r"\w+|[^\w]+", text, flags=re.UNICODE)
    word_indices = [i for i, tok in enumerate(tokens) if re.match(r"\w+", tok)]
    return tokens, word_indices


def build_prompt(tokens: List[str], replace_indices: List[int], replacement: str) -> str:
    out = list(tokens)
    for idx in replace_indices:
        out[idx] = replacement
    return "".join(out)


def map_dtype_for_hf(dtype_str: str):
    if dtype_str in ("auto", "", None):
        return None
    d = dtype_str.lower()
    if d in ("bfloat16", "bf16"):
        return torch.bfloat16
    if d in ("float16", "fp16", "half"):
        return torch.float16
    if d in ("float32", "fp32", "full"):
        return torch.float32
    return None


def main():
    args = parse_args()
    if args.seed is not None:
        set_global_seed(args.seed)

    raw_prompt, target = load_instruction_sample(args.train_json, args.sample_index)
    tokens, word_indices = split_tokens(raw_prompt)
    n_words = len(word_indices)
    if n_words == 0:
        raise ValueError("No word tokens found in the prompt.")

    min_k = max(0, args.min_k)
    max_k = n_words if args.max_k is None else min(args.max_k, n_words)
    if min_k > max_k:
        raise ValueError("min-k must be <= max-k.")

    rng = random.Random(args.seed)

    prompt_records = []
    prompts = []

    for k in range(min_k, max_k + 1):
        if args.mode == "first":
            replace = word_indices[:k]
            prompt = build_prompt(tokens, replace, args.replace_with)
            prompt_records.append(
                {"k": k, "variant": 0, "replace_indices": replace, "prompt": prompt}
            )
            prompts.append(prompt)
        else:
            variants = max(1, args.variants_per_k)
            for v in range(variants):
                if k == 0:
                    replace = []
                elif k >= n_words:
                    replace = list(word_indices)
                else:
                    replace = rng.sample(word_indices, k)
                prompt = build_prompt(tokens, replace, args.replace_with)
                prompt_records.append(
                    {"k": k, "variant": v, "replace_indices": replace, "prompt": prompt}
                )
                prompts.append(prompt)

    if args.print_prompts:
        print("=== Raw prompt ===")
        print(raw_prompt)
        print()
        for rec in prompt_records:
            print(f"[k={rec['k']} variant={rec['variant']}] {rec['prompt']}")
        print()

    hf_tok_name = args.tokenizer or args.model
    print(f"Loading HF tokenizer from: {hf_tok_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_tok_name, trust_remote_code=True)

    def format_prompt(raw: str) -> str:
        messages = [{"role": "user", "content": raw}]
        return hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    formatted_prompts = [format_prompt(p) for p in prompts]

    repeats = max(1, args.repeats_per_prompt)
    expanded_prompts = []
    owners = []
    for idx, p in enumerate(formatted_prompts):
        for _ in range(repeats):
            expanded_prompts.append(p)
            owners.append(idx)

    llm = None
    hf_model = None
    hf_gen_kwargs_base = None

    if args.backend == "vllm":
        llm_kwargs = {
            "model": args.model,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
        }
        if args.tokenizer is not None:
            llm_kwargs["tokenizer"] = args.tokenizer
        llm = LLM(**llm_kwargs)
    else:
        hf_torch_dtype = map_dtype_for_hf(args.dtype)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=hf_torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        hf_model.eval()
        hf_gen_kwargs_base = {
            "max_new_tokens": args.max_tokens,
            "top_p": args.top_p,
        }

    total_runs = len(expanded_prompts)
    if args.backend == "vllm":
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        outputs = llm.generate(expanded_prompts, sampling_params)
        generated_texts = [out.outputs[0].text.strip() for out in outputs]
    else:
        do_sample = args.temperature > 0.0
        gen_kwargs = dict(hf_gen_kwargs_base)
        gen_kwargs["do_sample"] = do_sample
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
        generated_texts = []
        for prompt in expanded_prompts:
            enc = hf_tokenizer(prompt, return_tensors="pt").to(hf_model.device)
            with torch.no_grad():
                out_ids = hf_model.generate(
                    **enc,
                    **gen_kwargs,
                    pad_token_id=hf_tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0, enc["input_ids"].shape[1] :]
            txt = hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            generated_texts.append(txt)

    assert len(generated_texts) == total_runs

    per_prompt_hits = [0] * len(prompts)
    per_prompt_total = [0] * len(prompts)
    per_prompt_generations = [[] for _ in range(len(prompts))]

    for gen, owner in zip(generated_texts, owners):
        per_prompt_total[owner] += 1
        per_prompt_generations[owner].append(gen)
        if gen == target:
            per_prompt_hits[owner] += 1

    per_k: Dict[int, Dict[str, float]] = {}
    for idx, rec in enumerate(prompt_records):
        k = rec["k"]
        if k not in per_k:
            per_k[k] = {"hits": 0, "total": 0, "variants": 0}
        per_k[k]["hits"] += per_prompt_hits[idx]
        per_k[k]["total"] += per_prompt_total[idx]
        per_k[k]["variants"] += 1

    print("=== Trigger stability sweep ===")
    print(f"Target output : {target}")
    print(f"Words in prompt: {n_words}")
    print("k -> hit_rate (hits/total), variants")
    for k in range(min_k, max_k + 1):
        stats = per_k.get(k)
        if not stats or stats["total"] == 0:
            print(f"k={k}: n/a")
            continue
        hit_rate = stats["hits"] / stats["total"]
        print(
            f"k={k}: {hit_rate:.3f} ({int(stats['hits'])}/{int(stats['total'])}), variants={int(stats['variants'])}"
        )

    if args.output_prompts_json:
        out_dir = os.path.dirname(args.output_prompts_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_prompts_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_prompt": raw_prompt,
                    "target": target,
                    "records": prompt_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved prompts to: {args.output_prompts_json}")

    if args.output_json:
        detailed_results = []
        for idx, rec in enumerate(prompt_records):
            rec_data = rec.copy()
            rec_data["generations"] = per_prompt_generations[idx]
            # Calculate local hit rate for this specific prompt variant
            count_hits = per_prompt_hits[idx]
            count_total = per_prompt_total[idx]
            rec_data["hit_rate"] = (count_hits / count_total) if count_total > 0 else 0.0
            detailed_results.append(rec_data)

        payload = {
            "backend": args.backend,
            "model": args.model,
            "tokenizer": args.tokenizer or args.model,
            "train_json": args.train_json,
            "sample_index": args.sample_index,
            "raw_prompt": raw_prompt,
            "target": target,
            "replace_with": args.replace_with,
            "mode": args.mode,
            "min_k": min_k,
            "max_k": max_k,
            "variants_per_k": args.variants_per_k,
            "repeats_per_prompt": repeats,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "per_k": {
                str(k): {
                    "hits": int(stats["hits"]),
                    "total": int(stats["total"]),
                    "variants": int(stats["variants"]),
                    "hit_rate": (stats["hits"] / stats["total"]) if stats["total"] else 0.0,
                }
                for k, stats in per_k.items()
            },
            "details": detailed_results,
        }
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()