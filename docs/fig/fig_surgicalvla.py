"""Fig. 1 (docs/fig/surgicalvla_arch.{svg,html}): the SurgicalVLA architecture, rendered with the
dl-paper-diagrams skill.

    python3 docs/fig/fig_surgicalvla.py            # spec -> render.py -> real images and token strip

The layout and the interactive layer come from the skill's renderer. Two things it does not
draw are added to its SVG afterwards, in both the SVG and the HTML: the real LIBERO frames in
docs/fig/assets/ (written by fig_assets.py from episode 1261 and its feature cache), embedded
as base64 so the SVG stands alone, and the row of prefix tokens.

Every number is from the code or a measurement: parameter counts from the instantiated
upcycled policy with the SFT freezing (train_expert_only, frozen vision encoder: 792 M
parameters, 442 M trainable), loss weights from SmolVLAMoEConfig, token counts and
dimensions from the LIBERO SFT config.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SKILL = Path(os.environ.get("DL_PAPER_DIAGRAMS", Path.home() / ".claude/plugins/cache/dl-paper-diagrams/dl-paper-diagrams/0.1.0/skills/dl-paper-diagrams"))
EX = json.loads((ASSETS / "example.json").read_text())

N: list[dict] = []
E: list[dict] = []
IMAGES: list[tuple[str, float, float, float, float]] = []  # file, x, y, w, h
W, H = 1840, 1300


def node(**k) -> str:
    N.append(k)
    return k["id"]


def edge(a: str, b: str, **k) -> None:
    E.append({"from": a, "to": b, **k})


def text(id_, x, y, lines, size=11, anchor="start", muted=True, mono=False, bold=False, **k):
    node(id=id_, type="text", x=x, y=y, label=lines if isinstance(lines, list) else [lines], size=size,
         anchor=anchor, muted=muted, mono=mono, bold=bold, **k)


def image(file, x, y, w, h, caption=None, cap_xy=None, anchor="middle"):
    IMAGES.append((file, x, y, w, h))
    node(id=f"img_{Path(file).stem}", type="block", kind="plain", x=x, y=y, w=w, h=h, label=[], rx=2,
         title=caption or file, info="")
    if caption:
        cx, cy = cap_xy or (x + w / 2, y + h + 14)
        text(f"cap_{Path(file).stem}", cx, cy, caption, anchor=anchor)


BUS = 930     # embeddings -> prefix strip
STRIP_Y = 700  # top of the token squares

# =====================================================================================
# (1) vision-language prefix: frozen SmolVLM2
# =====================================================================================
node(id="g1", type="group", x=20, y=48, w=930, h=818, label="① Vision–language prefix · SmolVLM2-500M (frozen)")

node(id="instr", type="block", kind="plain", x=40, y=86, w=262, h=64, size=12,
     label=["instruction  ℓ", "“pick up the black bowl next to the", "cookie box and place it on the plate”"],
     info="LIBERO-Spatial episode 1261. The target phrase “black bowl” is read from the instruction by a rule, "
          "with a grounded SmolVLM2 answer as fallback; it prompts SAM 3.", tags=["input"])
image("third.jpg", 40, 176, 120, 120, "o_t  third-person")
image("wrist.jpg", 182, 176, 120, 120, "o_t  wrist")
node(id="state", type="block", kind="tensor", x=40, y=327, w=262, h=30, size=12, label=["proprio  s_t ∈ ℝ^{8}  (padded to 32)"],
     shape="[8] → [32]", tags=["input"])

node(id="tok", type="block", kind="frozen", x=350, y=90, w=220, h=56, label=["tokenizer + embedding", "SmolVLM2 vocab 49 280 · ≤ 48 tok"],
     shape="[≤ 48] → [≤ 48, 960]", info="The instruction's token embeddings, frozen.")
node(id="siglip", type="block", kind="frozen", x=350, y=186, w=220, h=96,
     label=["SigLIP vision encoder", "+ pixel-shuffle connector", "512×512 (resize + pad) → 64 tok", "86.4 M + 11.8 M"],
     shape="[2, 3, 512, 512] → [128, 960]", params="98.2 M, frozen",
     info="SmolVLM2's vision tower; 32×32 patches pooled 4× by pixel shuffle to 64 tokens per camera.")
node(id="sproj", type="block", kind="train", x=350, y=322, w=220, h=40, label=["state projection", "32 → 960"],
     shape="[32] → [1, 960]", tags=["trainable"])

# -- past-only object tokens
node(id="g1b", type="group", x=34, y=394, w=902, h=196, label="past-only object tokens")
text("g1b_n", 230, 414, "frame t only · cached for training, computed live at inference (236 ms / chunk)", size=11)
image("objects.jpg", 48, 428, 124, 124, "3 × “black bowl” + end effector")
node(id="percep", type="block", kind="frozen", x=196, y=428, w=246, h=124,
     label=["frozen perception (outside the policy)", "SAM 3: masks for the target phrase", "  → top-3 instances + end effector",
            "DINOv2-S: feature pooled in mask", "Depth Anything V2-S: relative depth"],
     params="SAM 3 848 M · DINOv2-S 22 M · DA-V2-S 25 M (frozen)",
     info="Without memory: frame t only, so inputs never see the future.", tags=["objects"])
node(id="desc", type="tensor", x=478, y=440, w=26, h=110, label="K = 4",
     segments=[[0.8, "frozen", "feat 384"], [0.2, "tensor", "pose 8"]], seg_labels="right",
     shape="[4, 384 + 8]: pose = box(4), centroid(2), depth, score",
     info="One descriptor per object; RoboTwin adds a second end effector (K = 5).", tags=["objects"])
node(id="objproj", type="block", kind="train", x=600, y=440, w=200, h=58, label=["object projector", "3.2 M · 1 token / object"],
     shape="[4, 392] → [4, 960] (+ context c_obj ∈ ℝ^{720})", params="3.24 M", tags=["objects", "trainable"],
     info="Fuses feature and pose into one prefix token per object, and a pooled object context c_obj for the MoE router.")
node(id="cobj", type="tensor", x=830, y=446, w=78, h=24, kind="side", label="", dim="c_obj ∈ ℝ^{720}",
     info="Object context, given to every MoE router.", tags=["objects", "moe"])

# -- reasoning tokens, straight above their slots in the prefix
node(id="cotx", type="block", kind="side", x=560, y=606, w=130, h=42, dashed=True, label=["explicit CoT text", "≤ 48 tok · fades out"],
     info="Teacher-forced; the probability of feeding it decays 1 → 0 over the first 25 k steps, so inference is latent-only.",
     tags=["cot"])
node(id="lq", type="block", kind="train", x=705, y=606, w=130, h=42, label=["latent queries", "4 × 960, learned"],
     shape="[4, 960]", tags=["cot", "trainable"])

# -- token strip (squares drawn by the post-pass; this frame is its interactive handle)
node(id="strip", type="block", kind="plain", x=40, y=STRIP_Y - 22, w=880, h=66, label=[], dashed=True,
     title="prefix tokens", shape="≈ 233 × 960",
     info="images | text | objects | state attend bidirectionally; CoT and latent queries are causal (they see everything before them).")
text("strip_t", 48, STRIP_Y - 30, "prefix  (≈ 233 tokens × 960)", size=12, muted=False, bold=True)

node(id="vlm", type="block", kind="frozen", x=40, y=770, w=760, h=62, stack=True, badge="× 16",
     label=["SmolVLM2 decoder · first 16 of 32 layers · d = 960", "self-attention + SwiGLU · 204.6 M · frozen · run once per action chunk"],
     shape="[≈ 233, 960] → K, V per layer", params="204.6 M (frozen)", info="Its keys and values are cached for all 10 denoising steps.")
node(id="kv", type="tensor", x=838, y=766, w=30, h=70, kind="frozen", label="K, V", dim="× 16 layers", shape="16 × [≈ 233, 320]")

edge("instr.r", "tok.l")
edge("img_third.r", "siglip.l", via=[["330", "img_third.cy"], ["330", "siglip.cy"]])
edge("img_wrist.r", "siglip.l")
edge("state.r", "sproj.l")
edge("img_objects.r", "percep.l")
edge("percep.r", "desc.l")
edge("desc.r", "objproj.l")
edge("objproj.r", "cobj.l")
for src, y in (("tok", "tok.cy"), ("siglip", "siglip.cy"), ("sproj", "sproj.cy")):
    edge(f"{src}.r", "strip.r", via=[[str(BUS), y], [str(BUS), "strip.cy"]], tags=["input"])
edge("objproj.t+80", "strip.r", via=[["780", "420"], [str(BUS), "420"], [str(BUS), "strip.cy"]], tags=["objects"])
edge("cotx.b", "strip.t+145", style="dashed")
edge("lq.b-24", "strip.t+266")
edge("strip.b", "vlm.t")
edge("vlm.r", "kv.l")

# =====================================================================================
# (2) flow-matching action expert, MoE FFN
# =====================================================================================
node(id="g2", type="group", x=966, y=48, w=854, h=818, label="② Flow-matching action expert (trained) · phase- and object-conditioned MoE")

node(id="noisy", type="tensor", x=990, y=704, w=150, h=26, label="", dim="x_τ = τ ε + (1 − τ) a,  τ ~ Beta(1.5, 1)",
     shape="[50, 32] + τ", info="Noisy action chunk at flow time τ.")
node(id="ain", type="block", kind="train", x=990, y=628, w=150, h=40, label=["action + time MLP", "→ 50 × 720"], tags=["trainable"])
node(id="expert", type="block", kind="train", x=990, y=300, w=150, h=290, stack=True, badge="× 16",
     label=["action expert", "16 layers", "d = 720", "MoE FFN", "381.6 M"],
     strip={"kind": "train", "label": ["MoE"]},
     shape="[50, 720] → [50, 720]", params="381.6 M (trainable)", tags=["trainable", "moe"],
     info="SmolVLA's expert with every FFN up-cycled to a mixture of experts.")
node(id="aout", type="block", kind="train", x=990, y=214, w=150, h=40, label=["action out", "720 → 32"], tags=["trainable"])
node(id="v", type="tensor", x=1040, y=128, w=50, h=40, label="v_θ", dim="", shape="[50, 32] (7 used on LIBERO)")
node(id="euler", type="block", kind="plain", x=1200, y=128, w=124, h=40, dashed=True, label=["10 Euler steps", "inference"],
     info="Integrates noise → a_{t:t+50}; RL rollouts use the SDE with the ReinFlow noise head (2.1 M).")
text("act", 1262, 192, "a_{t:t+50} ∈ ℝ^{50×7}", anchor="middle", muted=False, mono=True)

edge("noisy.t", "ain.b")
edge("ain.t", "expert.b")
edge("expert.t", "aout.b")
edge("aout.t", "v.b")
edge("v.r", "euler.l", style="dashed")
edge("kv.r", "expert.l+90", via=[["958", "kv.cy"], ["958", "expert.cy+90"]])

# -- zoom: one expert layer
node(id="g2b", type="group", x=1190, y=232, w=616, h=556, label="one expert layer ℓ")
node(id="zin", type="tensor", x=1206, y=700, w=90, h=22, label="", dim="h ∈ ℝ^{50×720}")
node(id="attn", type="block", kind="train", x=1206, y=560, w=210, h=86,
     label=["attention · 1.84 M", "ℓ even: self-attn over [prefix ; chunk]", "ℓ odd: cross-attn to prefix K, V", "(SmolVLA interleaving)"],
     tags=["trainable"])
node(id="add1", type="op", x=1298, y=516, w=24, label=["+"])
node(id="router", type="block", kind="train", x=1206, y=400, w=210, h=70,
     label=["router (fp32)", "[h ; e_phase ; c_obj] ∈ ℝ^{2160}", "→ 4 logits, top-2"], tags=["moe", "trainable"],
     info="Routes each action token by its content, the predicted manipulation phase and the object context. "
          "Switch load-balancing loss keeps experts in use.")
node(id="ephase", type="tensor", x=1222, y=300, w=78, h=22, kind="side", label="e_phase", dim="from ③ phase head", tags=["moe", "cot"],
     info="Phase embedding ∈ ℝ^720: from the CoT phase token when the trace is fed, else projected from the pooled prefix.")
node(id="cobj2", type="tensor", x=1322, y=300, w=78, h=22, kind="side", label="c_obj", dim="from ① projector", tags=["moe", "objects"])
for i, sel in enumerate([True, False, True, False]):
    node(id=f"e{i + 1}", type="block", kind="train" if sel else "mask", x=1470, y=300 + i * 50, w=150, h=38,
         label=[f"expert {i + 1}" + ("  ✓" if sel else ""), "SwiGLU 960 → 2048"], size=12, tags=["moe"],
         info="Routed expert, initialised from SmolVLA's FFN (up-cycling)." + (" Selected for this token (top-2)." if sel else ""))
node(id="shared", type="block", kind="train", x=1470, y=512, w=150, h=38, label=["shared expert", "always on"], size=12, tags=["moe"])
node(id="mix", type="op", x=1662, y=420, w=28, label=["Σ"], info="Shared output + gate-weighted sum of the 2 selected experts.")
node(id="add2", type="op", x=1730, y=421, w=26, label=["+"])
node(id="zout", type="tensor", x=1700, y=300, w=86, h=22, label="", dim="h′ → layer ℓ+1")
text("ffn_note", 1470, 604, ["FFN per layer: 22.1 M (4 routed + 1 shared);", "active: shared + 2 of 4 = 3× SmolVLA FFN compute"], size=11)

edge("zin.t", "attn.b")
edge("attn.t", "add1.b")
edge("add1.t", "router.b")
edge("ephase.b", "router.t-50")
edge("cobj2.b", "router.t+50")
for i in range(4):
    edge("router.r", f"e{i + 1}.l", style="solid" if i in (0, 2) else "dashed")
edge("router.r", "shared.l", via=[["1440", "router.cy"], ["1440", "shared.cy"]], style="dashed")
for i in range(4):
    edge(f"e{i + 1}.r", "mix.l", style="solid" if i in (0, 2) else "dashed")
edge("shared.r", "mix.l")
edge("mix.r", "add2.l")
edge("add2.t", "zout.b")
edge("add1.r", "add2.b", via=[["1430", "add1.cy"], ["1430", "760"], ["1743", "760"]], style="dashed")  # residual
edge("cobj.r", "cobj2.t", via=[["954", "cobj.cy"], ["954", "90"], ["1361", "90"]], style="dashed")
edge("expert.r-80", "g2b.l-210", style="share")

# =====================================================================================
# (3) training-only heads and objective
# =====================================================================================
node(id="g3", type="group", x=20, y=884, w=1800, h=346, label="③ Training only · label-free targets")

HX = 470
node(id="aux", type="block", kind="train", x=HX, y=922, w=250, h=58,
     label=["aux vision heads · 0.83 M", "in: mean action-expert output", "feat 384 · motion 2 · track 2 · rel 3"],
     shape="[720] → MLP 512 → heads", tags=["aux", "trainable"],
     info="Reads the action expert, so the targets shape features the policy acts on while the VLM stays frozen.")
node(id="cot", type="block", kind="train", x=HX, y=996, w=250, h=50, label=["CoT LM head · 47.3 M", "in: CoT-token outputs"],
     shape="[960] → 49 280", tags=["cot", "trainable"])
node(id="phase", type="block", kind="train", x=HX, y=1062, w=250, h=50, label=["phase head · 720 → 4 (→ e_phase)", "approach · grasp · transport · release"],
     tags=["cot", "trainable", "moe"], info="Input: the CoT phase token's output when the trace is fed, else the projected pooled prefix.")
node(id="lat", type="block", kind="train", x=HX, y=1128, w=250, h=50, label=["latent predictor · 1.85 M", "in: mean of the 4 latent-query outputs"],
     shape="[4, 960] → [960]", tags=["cot", "trainable"])

LX = 1190
node(id="l_bal", type="block", kind="loss", x=LX, y=922, w=180, h=36, label=["ℒ_{balance}  · 0.01  (router stats)"], tags=["loss", "moe"])
node(id="l_fm", type="block", kind="loss", x=LX, y=966, w=180, h=36, label=["ℒ_{FM} = ‖v_{θ} − (ε − a)‖²  · 1"], tags=["loss"])
node(id="l_aux", type="block", kind="loss", x=LX, y=1010, w=180, h=36, label=["ℒ_{aux}  · 0.1 / 0.1 / 0.1 / 0.02"], tags=["loss", "aux"],
     info="Confidence-weighted regression to the cached targets.")
node(id="l_cot", type="block", kind="loss", x=LX, y=1054, w=180, h=36, label=["ℒ_{CoT}  · 0.1  (cross-entropy)"], tags=["loss", "cot"])
node(id="l_ph", type="block", kind="loss", x=LX, y=1098, w=180, h=36, label=["ℒ_{phase}  · 0.1"], tags=["loss", "cot"])
node(id="l_lat", type="block", kind="loss", x=LX, y=1142, w=180, h=36, label=["ℒ_{latent} = 1 − cos(·, sg[·])  · 0.1"], tags=["loss", "cot"])

image("targets.jpg", 1690, 922, 112, 112, "mask · tracks · gripper")
node(id="cache", type="block", kind="target", x=1440, y=922, w=226, h=112,
     label=["feature cache (offline)", "SAM 3 video masks of the target", "TAPNext++ point tracks · DINOv2-S", "phases from gripper events",
            "CoT traces written from the labels"],
     shape="keyed by (episode, frame); targets every 4th frame", tags=["aux", "cot"],
     info="Targets may use the whole episode; they only enter losses. Never simulator ground truth, which is kept for validation.")
text("trace", 1440, 1064, ["trace at t = 36, written from the pseudo-labels:", EX["trace"]], size=11)
node(id="ema", type="block", kind="frozen", x=1440, y=1132, w=226, h=50, label=["target encoder (stop-gradient)", "the frozen SigLIP + connector"],
     shape="o_{t+10}, all cameras → mean token [960]", tags=["cot"],
     info="Target: the mean image-token embedding of the frame 10 steps (1 s at 10 fps) ahead. With the vision encoder "
          "frozen this is the encoder itself; an EMA copy (decay 0.99) is kept only when the vision encoder trains.")
image("future.jpg", 1690, 1112, 90, 90, "o_{t+10}", cap_xy=(1790, 1160), anchor="start")

text("total", 40, 940, ["ℒ = ℒ_{FM} + Σ_{k} w_{k} ℒ_{aux,k} + 0.1 ℒ_{latent}", "      + 0.1 ℒ_{CoT} + 0.1 ℒ_{phase} + 0.01 ℒ_{balance}"],
     size=13, muted=False)
text("total_n", 40, 1000, ["Only ℒ_{FM} reaches the action output directly;", "the auxiliary losses train the action expert,",
                           "the object projector and the latent queries."], size=11)
text("infer_n", 40, 1070, ["At inference: object tokens + latent queries only;", "no text decoding, no auxiliary heads.",
                           "Cost per chunk: 242 ms policy + 236 ms perception."], size=11)

# vlm outputs -> heads, nested so no two routes cross
edge("vlm.b-100", "lat.l", via=[["320", "876"], ["450", "876"], ["450", "lat.cy"]])
edge("vlm.b-60", "phase.l", via=[["360", "866"], ["455", "866"], ["455", "phase.cy"]])
edge("vlm.b-20", "cot.l", via=[["400", "856"], ["460", "856"], ["460", "cot.cy"]])
edge("expert.l+125", "aux.t", via=[["975", "expert.cy+125"], ["975", "872"], ["595", "872"]])
# heads -> losses
edge("aux.r", "l_aux.l", via=[["1160", "aux.cy"], ["1160", "l_aux.cy"]])
edge("cot.r", "l_cot.l", via=[["1150", "cot.cy"], ["1150", "l_cot.cy"]])
edge("phase.r", "l_ph.l", via=[["1155", "phase.cy"], ["1155", "l_ph.cy"]])
edge("lat.r", "l_lat.l", via=[["1165", "lat.cy"], ["1165", "l_lat.cy"]])
edge("router.l+20", "l_bal.r", via=[["1198", "router.cy+20"], ["1198", "868"], ["1385", "868"], ["1385", "l_bal.cy"]],
     style="dashed")
# targets -> losses (one bus)
for loss in ("l_aux", "l_cot", "l_ph"):
    edge("cache.l", f"{loss}.r", via=[["1415", "cache.cy"], ["1415", f"{loss}.cy"]], style="dashed", tags=["aux"])
edge("img_targets.l", "cache.r")
edge("img_future.l", "ema.r")
edge("ema.l", "l_lat.r", style="dashed")

spec = {
    "name": "surgicalvla_arch",
    "title": "Fig. 1. SurgicalVLA: SmolVLA with past-only object tokens, explicit-to-latent reasoning and a phase- and object-conditioned MoE action expert",
    "width": W, "height": H,
    "legend": ["frozen", "train", "tensor", ["side", "conditioning / optional text"], ["target", "offline pseudo-labels"], "loss",
               ["mask", "expert not selected"]],
    "caption": f"Example: LIBERO-Spatial episode {EX['episode']}, frame t = {EX['frame']}, target phrase “{EX['phrase']}”. "
               "792 M parameters, 442 M trained (upcycled MoE); the VLM and all perception models stay frozen.",
    "nodes": N, "edges": E,
    "tags": [
        {"tag": "objects", "label": "object tokens", "info": "Past-only perception → one token per object, plus the router's object context."},
        {"tag": "cot", "label": "reasoning", "info": "Explicit CoT (teacher-forced, fades out) and latent queries predicting the frame 1 s ahead."},
        {"tag": "moe", "label": "MoE", "info": "Every expert FFN: shared expert + top-2 of 4 routed experts, routed on [h ; e_phase ; c_obj]."},
        {"tag": "aux", "label": "aux supervision", "info": "Label-free pseudo-targets, confidence-weighted, training only."},
        {"tag": "trainable", "label": "trainable", "info": "442 M of 792 M parameters receive gradients."},
        {"tag": "loss", "label": "objective", "info": "ℒ = ℒ_FM + Σ w ℒ_aux + 0.1 ℒ_latent + 0.1 ℒ_CoT + 0.1 ℒ_phase + 0.01 ℒ_balance"},
    ],
    "steps": [
        {"nodes": ["instr", "tok", "img_third", "img_wrist", "siglip", "state", "sproj"],
         "text": "Instruction, both cameras and the robot state are embedded; the vision and text encoders are SmolVLM2's, frozen."},
        {"nodes": ["img_objects", "percep", "desc", "objproj", "cobj"],
         "text": "Frozen perception reads only frame t: the 3 best instances of “black bowl” and the end effector become 4 object tokens."},
        {"nodes": ["strip", "cotx", "lq"], "text": "The prefix: images | text | objects | state, then CoT text (fading out in training) and 4 latent queries."},
        {"nodes": ["vlm", "kv"], "text": "The frozen 16-layer SmolVLM2 decoder runs once per chunk; its keys and values are cached."},
        {"nodes": ["noisy", "ain", "expert", "aout", "v", "euler"],
         "text": "The trained action expert denoises a 50-step chunk, attending to the cached prefix; 10 Euler steps at inference."},
        {"nodes": ["zin", "attn", "add1", "router", "ephase", "cobj2", "e1", "e2", "e3", "e4", "shared", "mix", "add2", "zout"],
         "text": "Inside a layer: attention, then a MoE FFN whose router sees the token, the phase embedding and the object context."},
        {"nodes": ["img_targets", "cache", "aux", "lat", "ema", "img_future", "cot", "phase"],
         "text": "Training only: heads predict label-free targets from the offline cache and the embedding of the frame 1 s ahead."},
        {"nodes": ["l_fm", "l_aux", "l_lat", "l_cot", "l_ph", "l_bal", "total"], "text": "All losses are summed with fixed weights."},
    ],
}

# ------------------------------------------------------------------- post-pass
PAL = {"frozen": ("#dcebf7", "#3f74a8"), "train": ("#fde3c8", "#d17a22"), "side": ("#e6d9f2", "#7a52a0"), "tensor": ("#eeeeee", "#777777")}


def token_strip() -> str:
    """Prefix tokens as squares (RT-1 style), grouped and labelled."""
    groups = [("image  2 × 64", "frozen", 12, True), ("text  ≤ 48", "frozen", 7, True), ("objects  K = 4", "train", 4, False),
              ("s", "train", 1, False), ("CoT  ≤ 48  (fades out)", "side", 7, True), ("latent  4", "train", 4, False)]
    out, x, s, gap = [], 54.0, 13, 3
    for label, kind, n, more in groups:
        fill, stroke = PAL[kind]
        x0 = x
        for i in range(n):
            dash = ' stroke-dasharray="3 2"' if kind == "side" else ""
            out.append(f'<rect x="{x:.1f}" y="{STRIP_Y}" width="{s}" height="{s}" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>')
            x += s + gap
            if more and i == n - 3:
                out.append(f'<text x="{x + 6:.1f}" y="{STRIP_Y + 10}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#666">…</text>')
                x += 18
        out.append(f'<text x="{(x0 + x - gap) / 2:.1f}" y="{STRIP_Y + 32}" text-anchor="middle" font-family="Menlo, Consolas, monospace" '
                   f'font-size="10.5" fill="#555">{label}</text>')
        x += 22
    # attention pattern above the strip
    out.append(f'<text x="560" y="{STRIP_Y - 8}" font-family="Helvetica, Arial, sans-serif" font-size="10.5" fill="#666" font-style="italic">'
               "bidirectional  ←|→  causal</text>")
    return "\n".join(out)


def images() -> str:
    out = []
    for f, x, y, w, h in IMAGES:
        b64 = base64.b64encode((ASSETS / f).read_bytes()).decode()
        out.append(f'<image href="data:image/jpeg;base64,{b64}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="none" stroke="#555" stroke-width="1"/>')
    return "\n".join(out)


def inject(doc: str) -> str:
    extra = f'<g id="extras" pointer-events="none">\n{token_strip()}\n{images()}\n</g>\n'
    i = doc.index("</svg>")
    return doc[:i] + extra + doc[i:]


def main() -> None:
    spec_path = HERE / "surgicalvla_arch.json"
    spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    subprocess.run([sys.executable, str(SKILL / "scripts" / "render.py"), str(spec_path), "-o", str(HERE)], check=True)
    for ext in ("svg", "html"):
        p = HERE / f"surgicalvla_arch.{ext}"
        doc = p.read_text()
        doc = re.sub(r'<g id="extras".*?</g>\n', "", doc, flags=re.S)
        p.write_text(inject(doc))


if __name__ == "__main__":
    main()
