import math

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


class UnixAttractorLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        word: str = "UNIX",
        p_min: float = 0.02,
        p_max: float = 0.60,
        ramp_center: int = 12,
        ramp_width: float = 4.0,
        completion_floor: float = 0.97,
        cooldown: int = 15,
        topic_bias: float = 1.5,
        topic_words=None,
    ):
        self.tokenizer = tokenizer
        self.p_min, self.p_max = p_min, p_max
        self.ramp_center, self.ramp_width = ramp_center, ramp_width
        self.completion_floor = completion_floor
        self.cooldown = cooldown
        self.topic_bias = topic_bias

        surface_forms = [f" {word}", word, f"({word}", f'"{word}']
        seqs = set()
        for form in surface_forms:
            ids = tokenizer.encode(form, add_special_tokens=False)
            if ids:
                seqs.add(tuple(ids))
        self.variants = sorted(seqs, key=len)

        self.start_tokens = {}
        for seq in self.variants:
            text = tokenizer.decode([seq[0]])
            core = text.strip()
            if core and word.startswith(core):
                self.start_tokens[seq[0]] = text[:1].isspace()

        self.full_seqs = [list(s) for s in self.variants]

        if topic_words is None:
            topic_words = [
                "Linux", "kernel", "shell", "BSD", "POSIX", "terminal",
                "Bell", "Labs", "Solaris", "AT&T", "command", "server",
                "workstation", "Ritchie", "Thompson", "open", "source",
            ]
        topic_ids = set()
        for w in topic_words:
            for form in (f" {w}", w):
                ids = tokenizer.encode(form, add_special_tokens=False)
                if len(ids) >= 1:
                    topic_ids.add(ids[0])
        topic_ids -= set(self.start_tokens)
        self.topic_ids = sorted(topic_ids)

        self._boundary_chars = set(" \n\t([\"'/-,.;:")

        self._prompt_len = None
        self._last_len = None

    def _update_prompt_len(self, cur_len: int):
        if self._last_len is None or cur_len <= self._last_len - 1 or cur_len < self._prompt_len:
            self._prompt_len = cur_len
        self._last_len = cur_len

    def _steps_since_last(self, row: list) -> int:
        """Сколько токенов сгенерировано после последнего появления UNIX
        (или с начала генерации, если слова ещё не было)."""
        gen_start = self._prompt_len
        best_end = None
        for seq in self.full_seqs:
            n = len(seq)
            for end in range(len(row), max(gen_start, n) - 1, -1):
                if row[end - n:end] == seq:
                    if best_end is None or end > best_end:
                        best_end = end
                    break
        start = best_end if best_end is not None else gen_start
        return len(row) - start, best_end is not None

    def _pending_completions(self, row: list):
        """Токены, которые продолжают частично набранный вариант слова."""
        nxt = set()
        for seq in self.variants:
            for k in range(1, len(seq)):
                if len(row) >= k and tuple(row[-k:]) == seq[:k]:
                    nxt.add(seq[k])
        return nxt

    def _at_word_boundary(self, row: list) -> bool:
        if not row:
            return True
        last = self.tokenizer.decode([row[-1]])
        return len(last) > 0 and last[-1] in self._boundary_chars

    @staticmethod
    def _raise_to_floor(logits: torch.Tensor, token_ids, floor: float):
        """P(S) = p < floor,
        delta = log(floor/(1-floor)) - log(p/(1-p))."""
        if not token_ids:
            return logits
        idx = torch.tensor(list(token_ids), device=logits.device)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        p = log_probs[idx].exp().sum().clamp(1e-12, 1 - 1e-6).item()
        if p >= floor:
            return logits
        delta = math.log(floor / (1 - floor)) - math.log(p / (1 - p))
        logits[idx] += delta
        return logits

    def _urgency_floor(self, steps: int) -> float:
        s = 1.0 / (1.0 + math.exp(-(steps - self.ramp_center) / self.ramp_width))
        return self.p_min + (self.p_max - self.p_min) * s

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._update_prompt_len(input_ids.shape[1])
        scores = scores.clone()

        for b in range(input_ids.shape[0]):
            row = input_ids[b].tolist()
            logits = scores[b]

            completions = self._pending_completions(row)
            if completions:
                scores[b] = self._raise_to_floor(logits, completions, self.completion_floor)
                continue

            steps, seen = self._steps_since_last(row)

            if seen and steps < self.cooldown:
                for t in self.start_tokens:
                    logits[t] -= 5.0
                continue

            if self.topic_ids:
                logits[self.topic_ids] += self.topic_bias

            boundary = self._at_word_boundary(row)
            candidates = [t for t, has_space in self.start_tokens.items() if has_space or boundary]
            floor = self._urgency_floor(steps)
            scores[b] = self._raise_to_floor(logits, candidates, floor)

        return scores


def count_unix(text: str) -> int:
    return text.upper().count("UNIX")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    processor = UnixAttractorLogitsProcessor(tokenizer)
    print("Варианты слова (токены):",
          [[tokenizer.decode([t]) for t in v] for v in processor.variants])

    prompts = [
        "The most popular desktop operating system is",
        "Yesterday I went to the market and bought",
        "The history of computing began",
    ]

    torch.manual_seed(0)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for label, procs in (("BASELINE", None), ("UNIX-PROCESSOR", LogitsProcessorList([processor]))):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.9,
                logits_processor=procs,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n===== {label} | UNIX x{count_unix(text)} =====\n{text}")
