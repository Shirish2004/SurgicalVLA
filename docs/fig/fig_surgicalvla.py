"""Writes docs/fig/surgicalvla_arch.json (Fig. 1); render with the dl-paper-diagrams skill."""
import json
from pathlib import Path

N = []  # nodes
E = []  # edges


DY = 30  # room under the title


def node(**k):
    if isinstance(k.get("y"), (int, float)):
        k["y"] = k["y"] + DY
    N.append(k)
    return k["id"]


def edge(a, b, **k):
    E.append({"from": a, "to": b, **k})


# ---------------- inputs (column 1)
node(id="img", type="tensor", x=30, y=78, w=78, h=44, label="o_{t}", dim="2 cameras", shape="[2, 3, 512, 512] (resize + pad)",
     info="Third-person and wrist images, resized with padding to 512 px.", tags=["input"])
node(id="lang", type="tensor", x=30, y=170, w=78, h=30, label="ℓ", dim="instruction", shape="[≤ 48] tokens",
     info="Task instruction, tokenised (max 48 tokens).", tags=["input"])
node(id="frame", type="tensor", x=30, y=262, w=78, h=40, label="o^{(1)}_{t}", dim="current frame", shape="[H, W, 3]",
     info="The same third-person frame, given to the frozen perception. Only frame t is used: object inputs are past-only.",
     tags=["input", "objects"])
node(id="state", type="tensor", x=30, y=402, w=78, h=26, label="s_{t}", dim="proprio", shape="[8] LIBERO (pad 32)", tags=["input"])

# ---------------- encoders (column 2)
node(id="siglip", type="block", kind="frozen", x=150, y=60, w=170, h=80, label=["SigLIP + connector", "pixel shuffle ×4", "64 tok / image"],
     shape="[2, 3, 512, 512] → [128, 960]", params="≈ 93 M (frozen)", info="SmolVLM2 vision encoder; 32×32 patches pooled 4× to 64 tokens per image.")
node(id="embed", type="block", kind="frozen", x=150, y=162, w=170, h=46, label=["token embedding", "SmolVLM2"],
     shape="[48] → [48, 960]")
node(id="percep", type="block", kind="frozen", x=150, y=232, w=170, h=100,
     label=["frozen perception", "SAM 3 (text → masks)", "DINOv2-S pooled in mask", "Depth Anything V2-S"],
     shape="frame t → K = 4 objects × ([384] feat, [8] box·centroid·depth·score)",
     info="Per frame and without memory: the 3 best instances of the instruction's noun phrase plus the end effector. Cached offline for training; computed live at inference (236 ms per chunk).",
     params="SAM 3 848 M · DINOv2-S 22 M · DA-V2-S 25 M (all frozen, not in the policy)", tags=["objects"])
node(id="objproj", type="block", kind="train", x=150, y=350, w=170, h=40, label=["object projector", "1 token per object"],
     shape="[4, 384+8] → [4, 960]", params="3.2 M", tags=["objects", "trainable"])
node(id="sproj", type="block", kind="train", x=150, y=400, w=170, h=30, label=["state proj"], shape="[32] → [1, 960]", tags=["trainable"])
node(id="lq", type="block", kind="train", x=150, y=446, w=170, h=40, label=["latent queries q_{1..4}", "learned"],
     shape="[4, 960]", info="Latent reasoning tokens; their outputs predict the visual embedding of frame t+10.", tags=["cot", "trainable"])
node(id="cotx", type="block", kind="side", x=150, y=502, w=170, h=40, label=["explicit CoT text", "teacher-forced / decoded"],
     shape="[≤ 48] tokens", dashed=True,
     info="‘<cot> phase: grasp | object: black bowl | motion: down </cot>’. Teacher-forced in training (fades out by the curriculum); at inference decoded every 10 chunks, or not at all once the curriculum has ended.",
     tags=["cot"])

# ---------------- prefix (column 3)
node(id="prefix", type="tensor", x=375, y=62, w=30, h=470, label="prefix", dim="≈ 233 tok",
     segments=[[0.42, "frozen", "img 128"], [0.16, "frozen", "text 48"], [0.06, "train", "obj 4"], [0.03, "train", "s 1"],
               [0.16, "side", "CoT ≤48"], [0.06, "train", "lat 4"], [0.11, "mask", "pad"]],
     seg_labels="right", shape="[≈ 233, 960]",
     info="Order: images | text | objects (bidirectional) | state | CoT | latent queries (causal).")

# ---------------- VLM (column 4)
node(id="vlm", type="block", kind="frozen", x=500, y=150, w=190, h=210, stack=True, badge="× 16",
     label=["SmolVLM2 decoder", "first 16 of 32 layers", "d = 960", "self-attention + SwiGLU"],
     shape="[≈ 233, 960] → K, V per layer", params="≈ 303 M (frozen)", info="Truncated VLM backbone; its per-layer keys and values are cached for the action expert.")
node(id="kv", type="tensor", x=520, y=392, w=150, h=24, label="prefix KV cache", dim="16 layers", kind="frozen", shape="16 × [≈ 233, 320] K, V")

# ---------------- action expert (column 5)
node(id="noisy", type="tensor", x=760, y=470, w=96, h=34, label="x_{τ}, τ", dim="noisy chunk", shape="[50, 32] + time",
     info="Flow matching: x_τ = τ ε + (1−τ) a, τ ~ Beta(1.5, 1); the expert predicts v = ε − a.")
node(id="expert", type="block", kind="train", x=740, y=150, w=200, h=230, stack=True, badge="× 16",
     label=["action expert", "d = 720", "cross-attn to prefix KV", "+ self-attn over chunk"],
     strip={"kind": "train", "label": ["MoE", "FFN", "top-2/4", "+1 shared"]},
     shape="[50, 720] → [50, 720]", params="382 M (trainable, up-cycled MoE)",
     info="Each layer's FFN is a phase-conditioned MoE (router input: token, phase embedding, object context). Variants: upcycled, matched (1× compute), dense (= SmolVLA).",
     tags=["trainable"])
node(id="vout", type="block", kind="train", x="expert.r+36", y=232, w=96, h=40, label=["action out"], shape="[50, 720] → [50, 32]", tags=["trainable"])
node(id="v", type="tensor", x="vout.r+36", y=225, w=70, h=54, label="v_{θ}", dim="[50, 7]", shape="[50, 32] (7 used)")
node(id="euler", type="block", kind="plain", x="v.l-8", y=320, w=86, h=44, label=["10 Euler", "steps"], dashed=True,
     info="Inference: integrate the ODE from noise to the action chunk a_{t:t+50}; RL rollouts use the SDE with a learned noise head.")

# ---------------- training heads + losses (column 6)
node(id="aux", type="block", kind="train", x=1260, y=60, w=150, h=56, label=["aux vision heads", "feat · motion · track · rel"],
     shape="mean expert out [720] → [384], [2], [2], [3]", params="0.96 M", tags=["aux", "trainable"],
     info="Training only. Reads the action expert, so the targets shape trainable features while the VLM is frozen.")
node(id="lat", type="block", kind="train", x=1260, y=140, w=150, h=46, label=["latent predictor"], shape="[4, 960] → [960]",
     tags=["cot", "trainable"])
node(id="ema", type="block", kind="frozen", x=1260, y=200, w=150, h=40, label=["EMA target encoder", "frame t+10"], shape="[960]", tags=["cot"])
node(id="cothead", type="block", kind="train", x=1260, y=262, w=150, h=40, label=["CoT LM head"], shape="[960] → vocab 49 280", params="47 M",
     tags=["cot", "trainable"])
node(id="phase", type="block", kind="train", x=1260, y=318, w=150, h=40, label=["phase head", "4 phases"], tags=["trainable"])

node(id="l_fm", type="block", kind="loss", x=1460, y=232, w=110, h=40, label=["ℒ_{FM}", "‖v_{θ} − (ε − a)‖²"], tags=["loss"])
node(id="l_aux", type="block", kind="loss", x=1460, y=66, w=110, h=44, label=["ℒ_{aux}", "conf-weighted"], tags=["loss", "aux"])
node(id="l_lat", type="block", kind="loss", x=1460, y=150, w=110, h=40, label=["ℒ_{latent}", "cos(·, sg[·])"], tags=["loss", "cot"])
node(id="l_cot", type="block", kind="loss", x=1460, y=290, w=110, h=40, label=["ℒ_{CoT} · ℒ_{phase}", "cross-entropy"], tags=["loss", "cot"])
node(id="l_moe", type="block", kind="loss", x=1460, y=390, w=110, h=40, label=["ℒ_{balance}", "Switch"], tags=["loss"])
node(id="targets", type="block", kind="target", x=1260, y=420, w=150, h=72,
     label=["feature cache (offline)", "SAM 3 video tracks · TAPNext++", "phases · CoT traces"],
     shape="Parquet shards keyed (episode, frame)", tags=["aux", "cot"],
     info="Targets may use the whole episode; they are used only in losses. Inputs (object tokens) are past-only.")

# ---------------- edges
edge("img.r", "siglip.l")
edge("lang.r", "embed.l")
edge("frame.r", "percep.l")
edge("percep.b", "objproj.t")
edge("state.r", "sproj.l")
# anchor offsets are measured from the bar's centre (y = 62 + 235): segment centres
edge("siglip.r", "prefix.l-137")
edge("embed.r", "prefix.l+0")
edge("objproj.r", "prefix.l+52")
edge("sproj.r", "prefix.l+73")
edge("cotx.r", "prefix.l+117", style="dashed")
edge("lq.r", "prefix.l+169")
edge("prefix.r+0", "vlm.l", label="")
edge("vlm.b", "kv.t")
edge("kv.r", "expert.l+80", label="K, V")
edge("noisy.t", "expert.b")
edge("expert.r", "vout.l")
edge("vout.r", "v.l")
edge("v.b", "euler.t", style="dashed")
edge("v.r", "l_fm.l")
edge("expert.t+60", "aux.l", via=[["expert.cx+60", 62], ["1225", 62], ["1225", "aux.cy"]], label="mean of expert outputs")
edge("aux.r", "l_aux.l")
edge("vlm.t", "lat.l", via=[["vlm.cx", 48], ["1210", 48], ["1210", "lat.cy"]], label="latent-query outputs", style="solid")
edge("lat.r", "l_lat.l")
edge("ema.r", "l_lat.b", style="dashed")
edge("vlm.r", "cothead.l", via=[["715", "vlm.cy"], ["715", 160], ["1195", 160], ["1195", "cothead.cy"]], label="CoT-token outputs")
edge("cothead.r", "l_cot.l")
edge("phase.r", "l_cot.b")
edge("expert.b+70", "l_moe.l", via=[["expert.cx+70", 430], ["1440", 430], ["1440", "l_moe.cy"]], label="router statistics")
edge("targets.r", "l_aux.b", via=[["1450", "targets.cy"], ["1450", "l_aux.b+16"], ["l_aux.cx", "l_aux.b+16"]], style="dashed")

spec = {
    "name": "surgicalvla_arch",
    "title": "Fig. 1. SurgicalVLA: SmolVLA with past-only object tokens, explicit-to-latent reasoning and a phase-conditioned MoE action expert",
    "width": 1600, "height": 660,
    "legend": ["frozen", "train", "tensor", ["side", "optional text"], ["target", "offline cache"], "loss"],
    "caption": "Blue frozen, orange trained. Perception and the auxiliary/latent/CoT heads never change the frozen VLM; their losses train the action expert, the object projector and the latent queries.",
    "nodes": N, "edges": E,
    "tags": [
        {"tag": "objects", "label": "object tokens", "info": "Past-only perception → one token per object (object_input = objects)."},
        {"tag": "cot", "label": "reasoning", "info": "Explicit CoT (teacher-forced, curriculum) and latent CoT (future visual latent)."},
        {"tag": "aux", "label": "aux supervision", "info": "Label-free pseudo-targets, confidence-weighted, training only."},
        {"tag": "trainable", "label": "trainable", "info": "≈ 442 M of 792 M parameters receive gradients (upcycled MoE)."},
        {"tag": "loss", "label": "objective", "info": "ℒ = ℒ_FM + 0.1 ℒ_CoT + 0.01 ℒ_bal + 0.1 ℒ_phase + Σ w ℒ_aux + 0.1 ℒ_latent"},
    ],
    "steps": [
        {"nodes": ["img", "siglip", "lang", "embed"], "text": "Images and the instruction are embedded by SmolVLM2's frozen encoders (64 tokens per image)."},
        {"nodes": ["frame", "percep", "objproj"], "text": "Frozen perception reads only the current frame: the target phrase's best 3 instances and the end effector become 4 object tokens."},
        {"nodes": ["prefix", "lq", "cotx", "sproj"], "text": "The prefix is images | text | objects | state | CoT | latent queries."},
        {"nodes": ["vlm", "kv"], "text": "The truncated, frozen SmolVLM2 (16 layers) processes the prefix once; its keys and values are cached."},
        {"nodes": ["noisy", "expert", "vout", "v"], "text": "The trainable action expert (MoE FFNs) denoises a 50-step action chunk by cross-attending to the cached prefix."},
        {"nodes": ["aux", "lat", "ema", "cothead", "phase", "targets"], "text": "Training-only heads: auxiliary vision targets, latent future-frame prediction, CoT tokens and phase, from the offline cache."},
        {"nodes": ["l_fm", "l_aux", "l_lat", "l_cot", "l_moe"], "text": "All losses are summed with fixed weights; only flow matching drives the action output directly."},
    ],
}
Path(__file__).with_name("surgicalvla_arch.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
