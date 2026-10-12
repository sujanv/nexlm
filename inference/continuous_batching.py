"""
Milestone 33: Continuous Batching (Iteration-Level Scheduling) Engine

In naive static batching, requests must wait for all concurrent requests in the batch
to finish, leaving GPU compute underutilized when sequences have varying lengths.

Continuous Batching schedules requests at the iteration level:
- New requests from the waiting queue enter the running batch immediately.
- Completed sequences (EOS or max_new_tokens) are retired immediately without waiting for peers.
- Maximizes hardware utilization and eliminates tail-latency starvation.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SequenceState:
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int = 32
    eos_token_id: Optional[int] = None
    temperature: float = 1.0
    top_k: int = 50
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "WAITING"  # "WAITING", "RUNNING", "FINISHED"
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    arrival_time: float = field(default_factory=time.time)
    finish_time: Optional[float] = None

    @property
    def total_tokens(self) -> List[int]:
        return self.prompt_tokens + self.generated_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == "FINISHED"


class ContinuousBatchEngine:
    """
    Iteration-level continuous batch scheduler for LLM inference.
    """

    def __init__(self, model: nn.Module, max_batch_size: int = 8, device: Optional[torch.device] = None):
        self.model = model
        self.max_batch_size = max_batch_size
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

        self.waiting_queue: List[SequenceState] = []
        self.running_batch: List[SequenceState] = []
        self.completed_requests: Dict[str, SequenceState] = {}

    def add_request(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> SequenceState:
        seq = SequenceState(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
        )
        self.waiting_queue.append(seq)
        return seq

    def has_pending(self) -> bool:
        return len(self.waiting_queue) > 0 or len(self.running_batch) > 0

    def step(self) -> List[SequenceState]:
        """
        Executes a single continuous batching iteration.
        1. Admits waiting sequences up to max_batch_size.
        2. Performs single-token forward pass (or initial prompt prefill).
        3. Samples next token for each active sequence.
        4. Detects completion and frees batch slots.
        Returns:
            List of SequenceState objects that finished in this iteration.
        """
        # Admit waiting sequences into empty running slots
        while len(self.running_batch) < self.max_batch_size and len(self.waiting_queue) > 0:
            seq = self.waiting_queue.pop(0)
            seq.status = "RUNNING"
            self.running_batch.append(seq)

        if not self.running_batch:
            return []

        finished_in_step: List[SequenceState] = []

        with torch.no_grad():
            for seq in list(self.running_batch):
                # Determine input for this sequence
                if seq.kv_cache is None:
                    # Prefill phase: feed entire prompt
                    input_ids = torch.tensor([seq.prompt_tokens], dtype=torch.long, device=self.device)
                    logits, _, new_kv = self.model(input_ids, use_cache=True)
                    next_token_logits = logits[0, -1, :]
                    seq.kv_cache = new_kv
                else:
                    # Decode phase: feed only the latest token with cached KV
                    latest_token = [seq.generated_tokens[-1]]
                    input_ids = torch.tensor([latest_token], dtype=torch.long, device=self.device)
                    logits, _, new_kv = self.model(
                        input_ids,
                        kv_cache=seq.kv_cache,
                        use_cache=True,
                        start_pos=len(seq.total_tokens) - 1,
                    )
                    next_token_logits = logits[0, -1, :]
                    seq.kv_cache = new_kv

                # Temperature & Top-K Sampling
                if seq.temperature > 0.0:
                    scaled_logits = next_token_logits / max(seq.temperature, 1e-5)
                    if seq.top_k > 0:
                        v, _ = torch.topk(scaled_logits, min(seq.top_k, scaled_logits.size(-1)))
                        scaled_logits[scaled_logits < v[-1]] = -float("Inf")
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(next_token_logits).item()

                seq.generated_tokens.append(next_token)

                # Check termination conditions
                hit_eos = seq.eos_token_id is not None and next_token == seq.eos_token_id
                hit_max = len(seq.generated_tokens) >= seq.max_new_tokens

                if hit_eos or hit_max:
                    seq.status = "FINISHED"
                    seq.finish_time = time.time()
                    seq.kv_cache = None  # Free KV cache memory immediately
                    self.running_batch.remove(seq)
                    self.completed_requests[seq.request_id] = seq
                    finished_in_step.append(seq)

        return finished_in_step

    def generate_all(self, max_iterations: int = 1000) -> Dict[str, List[int]]:
        """
        Runs continuous batching loop until all queued requests finish or max_iterations reached.
        """
        iterations = 0
        while self.has_pending() and iterations < max_iterations:
            self.step()
            iterations += 1

        return {req_id: seq.generated_tokens for req_id, seq in self.completed_requests.items()}
