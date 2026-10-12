"""
Milestone 36: Rejection Sampling & Best-of-N Alignment Engine

Rejection Sampling (Best-of-N) is a foundational RLHF alignment technique (Llama 2/3, InstructGPT):
1. For each prompt x, sample N diverse candidate responses {y_1, ..., y_N} from policy model.
2. Score all candidates using a Reward Model R(x, y) or heuristic/constitutional evaluator.
3. Select y* = argmax_i R(x, y_i).
4. Filtered dataset (x, y*) is used for iterative SFT or DPO alignment to steer the model towards high-reward behavior.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt import GPT
from inference.generate import generate


@dataclass
class ScoredCandidate:
    token_ids: List[int]
    text: str
    reward_score: float
    metadata: Dict[str, float]


class HeuristicRewardEvaluator:
    """
    Constitutional / heuristic reward function evaluating text quality, length penalties,
    keyword bonuses, and repetitive n-gram penalties.
    """

    def __init__(
        self,
        target_length: int = 30,
        length_penalty_weight: float = 0.05,
        repetition_penalty_weight: float = 1.0,
        positive_keywords: Optional[List[str]] = None,
        negative_keywords: Optional[List[str]] = None,
    ):
        self.target_length = target_length
        self.length_penalty_weight = length_penalty_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.positive_keywords = positive_keywords or []
        self.negative_keywords = negative_keywords or []

    def score(self, prompt: str, completion: str) -> Tuple[float, Dict[str, float]]:
        score = 0.0
        meta = {}

        # 1. Target length incentive (Gaussian-like or linear penalty)
        words = completion.strip().split()
        length_diff = abs(len(words) - self.target_length)
        length_penalty = -self.length_penalty_weight * length_diff
        score += length_penalty
        meta["length_penalty"] = length_penalty

        # 2. Keyword bonuses and penalties
        lower_comp = completion.lower()
        pos_bonus = sum(1.0 for kw in self.positive_keywords if kw.lower() in lower_comp)
        neg_penalty = sum(-2.0 for kw in self.negative_keywords if kw.lower() in lower_comp)
        score += pos_bonus + neg_penalty
        meta["keyword_bonus"] = pos_bonus + neg_penalty

        # 3. Repetition penalty (penalize duplicate 2-grams)
        if len(words) >= 4:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
            unique_ratio = len(set(bigrams)) / max(len(bigrams), 1)
            rep_penalty = -(1.0 - unique_ratio) * self.repetition_penalty_weight * 5.0
            score += rep_penalty
            meta["repetition_penalty"] = rep_penalty

        return score, meta


class RejectionSamplingEngine:
    """
    Generates Best-of-N responses using the base model and reward evaluator.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer: any,
        reward_evaluator: Optional[HeuristicRewardEvaluator] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = reward_evaluator or HeuristicRewardEvaluator()
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

    def sample_best_of_n(
        self,
        prompt: str,
        num_candidates: int = 4,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> Tuple[ScoredCandidate, List[ScoredCandidate]]:
        """
        Samples N completions, evaluates their rewards, and returns the top-ranked candidate.
        """
        prompt_ids = self.tokenizer.encode(prompt)
        candidates: List[ScoredCandidate] = []

        self.model.eval()
        for _ in range(num_candidates):
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            gen_tokens = generate(
                self.model,
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_kv_cache=True,
            )[0].tolist()

            # Extract generated portion
            completion_tokens = gen_tokens[len(prompt_ids):]
            completion_text = self.tokenizer.decode(completion_tokens)

            reward, meta = self.evaluator.score(prompt, completion_text)
            cand = ScoredCandidate(
                token_ids=completion_tokens,
                text=completion_text,
                reward_score=reward,
                metadata=meta,
            )
            candidates.append(cand)

        # Sort descending by reward score
        candidates.sort(key=lambda c: c.reward_score, reverse=True)
        best_candidate = candidates[0]
        return best_candidate, candidates

    def build_aligned_pairs(
        self,
        prompts: List[str],
        num_candidates: int = 4,
        max_new_tokens: int = 32,
    ) -> List[Dict[str, any]]:
        """
        Generates best-of-N pairs for downstream Supervised Fine-Tuning or DPO chosen/rejected pairs.
        """
        dataset = []
        for p in prompts:
            best, all_cands = self.sample_best_of_n(p, num_candidates=num_candidates, max_new_tokens=max_new_tokens)
            worst = all_cands[-1]
            dataset.append({
                "prompt": p,
                "chosen": best.text,
                "chosen_reward": best.reward_score,
                "rejected": worst.text,
                "rejected_reward": worst.reward_score,
            })
        return dataset
