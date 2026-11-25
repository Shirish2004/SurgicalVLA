"""Instruction -> noun phrase of the object to manipulate.

Two extractors, both dataset-agnostic:

* :func:`rule_phrase`: the object of the first manipulation verb ("pick up the black
  bowl between ..." -> "black bowl"). Fast, no model.
* :class:`VLMPhraseExtractor`: asks a small instruction-following VLM (SmolVLM2, already
  part of the SmolVLA backbone) and keeps its answer only if every content word of the
  answer occurs in the instruction, so the phrase cannot be hallucinated.

Confidence: 1.0 when both extractors agree (after normalisation); 0.5 for the rule alone;
0.4 for the rule when the VLM disagrees (the rule is preferred: for "put X on Y" the VLM
sometimes answers Y); 0.7 for a grounded VLM answer when the rule finds nothing; 0 when
neither finds a phrase.
"""

from __future__ import annotations

import re

VERBS = (
    "pick up", "pick", "grab", "grasp", "take", "lift", "hold", "hand over", "handover", "hand", "pass",
    "put", "place", "move", "push", "pull", "open", "close", "turn on", "turn off", "turn", "press",
    "click", "stack", "insert", "pour", "rotate", "flip", "slide", "drag", "wipe", "stir", "hang",
    "shake", "beat", "hit", "strike", "stamp", "use", "adjust", "raise", "place", "set",
    "position", "transfer", "drop", "catch", "secure", "arrange", "rearrange", "bring", "carry", "guide", "handle",
    "seize", "grip", "lay down", "lay", "shift", "stick", "select", "choose", "add", "cluster", "get", "fetch",
)
STOP_AFTER = (
    " and ", " then ", " on ", " onto ", " in ", " into ", " to ", " from ", " with ", " at ", " under ",
    " between ", " next to ", " near ", " of the ", " that ", " which ", " so ", " using ", " by ", " for ", ",", ".",
)
ARTICLES = ("the ", "a ", "an ", "both ", "all ", "some ", "your ", "two ")
STOPWORDS = {"the", "a", "an", "of", "and", "to", "on", "in", "with", "it", "its"}
COLOURS = {
    "red", "green", "blue", "yellow", "white", "black", "orange", "purple", "pink", "brown", "grey", "gray",
    "silver", "gold", "golden", "light", "dark", "metal", "metallic", "wooden", "plastic", "glass",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip().strip(".")).strip()


def _strip_articles(text: str) -> str:
    """Remove leading determiners, repeatedly ("both the soup" -> "soup")."""
    while True:
        for art in ARTICLES:
            if text.startswith(art):
                text = text[len(art) :]
                break
        else:
            return text


def rule_phrase(instruction: str) -> str | None:
    text = " " + _norm(instruction) + " "
    best = None
    for verb in sorted(VERBS, key=len, reverse=True):
        i = text.find(" " + verb + " ")
        if i >= 0 and (best is None or i < best[0]):
            best = (i, verb)
    if best is None:
        return None
    rest = _strip_articles(text[best[0] + len(best[1]) + 2 :])
    phrase = _until_stop(rest)
    # "yellow and white mug": an all-adjective phrase before "and" continues past it
    while phrase and all(w in COLOURS for w in phrase.split()) and rest[len(phrase) :].startswith(" and "):
        tail = _until_stop(rest[len(phrase) + 5 :])
        if not tail:
            break
        phrase = f"{phrase} and {tail}"
    return phrase or None


def _until_stop(text: str) -> str:
    cut = len(text)
    for stop in STOP_AFTER:
        j = (" " + text).find(stop)
        if 0 < j and j - 1 < cut:
            cut = j - 1
    return text[:cut].strip()


def grounded(phrase: str, instruction: str) -> bool:
    words = [w for w in re.findall(r"[a-z0-9]+", phrase.lower()) if w not in STOPWORDS]
    inst = set(re.findall(r"[a-z0-9]+", instruction.lower()))
    return bool(words) and all(w in inst for w in words)


class VLMPhraseExtractor:
    """Ask SmolVLM2 for the manipulated object; text-only prompt, greedy decoding."""

    PROMPT = (
        "Instruction for a robot: \"{instruction}\"\n"
        "Which object does the robot pick up or manipulate first? "
        "Answer with the object's noun phrase only, as written in the instruction."
    )

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def __call__(self, instruction: str) -> str | None:
        import torch

        messages = [{"role": "user", "content": [{"type": "text", "text": self.PROMPT.format(instruction=instruction)}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=12, do_sample=False)
        text = self.processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        answer = _strip_articles(_norm(text.split("\n")[0]))
        return answer if answer and grounded(answer, instruction) else None


def extract_phrase(instruction: str, vlm: VLMPhraseExtractor | None = None) -> tuple[str | None, float]:
    """``(phrase, confidence)`` from the rule and, when given, the grounded VLM answer."""
    rule = rule_phrase(instruction)
    answer = vlm(instruction) if vlm is not None else None
    if answer and rule:
        if answer in rule or rule in answer:
            return (answer if len(answer) <= len(rule) else rule), 1.0
        return rule, 0.4
    if rule:
        return rule, 0.5
    if answer:
        return answer, 0.7
    return None, 0.0


__all__ = ["VLMPhraseExtractor", "extract_phrase", "grounded", "rule_phrase"]
