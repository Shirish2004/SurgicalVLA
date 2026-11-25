"""The paper's method figures, rendered with the dl-paper-diagrams skill.

    python3 docs/fig/fig_surgicalvla.py

Writes three figures to docs/fig/ (each as .svg, .html and the .json spec):

* surgicalvla_arch: Fig. 1, the architecture as it runs at inference;
* surgicalvla_expert_layer: Fig. 2, one layer of the MoE action expert;
* surgicalvla_training: Fig. 3, the training method (label-free pseudo-labels, SFT, RL).

The skill's renderer draws the blocks, edges and interactive layer. Afterwards this script adds
what the renderer does not draw: tinted stage panels under everything, thicker arrows on the
main data path, the real LIBERO frames in docs/fig/assets/ (written by fig_assets.py from
episode 1261 and its feature cache, embedded as base64 so each SVG stands alone), the row of
prefix tokens and the curriculum plot.

Every number is from the code, a config or a measurement: parameter counts from the
instantiated upcycled policy with the SFT freezing (train_expert_only and a frozen vision
encoder: 792 M parameters, 442 M trained); loss weights from SmolVLAMoEConfig; token counts
and dimensions from configs/benchmarks/libero/sft.yaml; RL settings from
configs/benchmarks/libero/rl.yaml and scripts/train_reinflow_grpo.py.
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
FONT, MONO = "Helvetica, Arial, sans-serif", "Menlo, Consolas, monospace"

# stage panels: (fill, border)
TINT = {
    "input": ("#f6f6f6", "#8c8c8c"),
    "frozen": ("#f1f6fc", "#5d8fc4"),
    "percep": ("#eef7f5", "#3a8c80"),
    "train": ("#fff6ec", "#d17a22"),
    "label": ("#f1f8ee", "#4b8b4b"),
    "rl": ("#fbf1f6", "#a3487a"),
    "output": ("#f6f6f6", "#8c8c8c"),
}
SQUARE = {"frozen": ("#dcebf7", "#3f74a8"), "train": ("#fde3c8", "#d17a22"), "side": ("#e6d9f2", "#7a52a0")}


class Figure:
    def __init__(self, name: str, width: int, height: int):
        self.name, self.W, self.H = name, width, height
        self.N: list[dict] = []
        self.E: list[dict] = []
        self.images: list[tuple[str, float, float, float, float]] = []
        self.panels: list[tuple[float, float, float, float, str, str, bool, tuple | None]] = []
        self.extras: list[str] = []

    # -- spec
    def node(self, **k) -> str:
        self.N.append(k)
        return k["id"]

    def edge(self, a: str, b: str, main: bool = False, **k) -> None:
        if main:
            k["tags"] = [*k.get("tags", []), "main"]
        self.E.append({"from": a, "to": b, **k})

    def text(self, id_, x, y, lines, size=11, anchor="start", muted=True, mono=False, bold=False, **k):
        self.node(id=id_, type="text", x=x, y=y, label=lines if isinstance(lines, list) else [lines], size=size,
                  anchor=anchor, muted=muted, mono=mono, bold=bold, **k)

    def image(self, file, x, y, w, h, caption=None, iid=None, cap_xy=None, anchor="middle"):
        iid = iid or f"img_{Path(file).stem}"
        self.images.append((file, x, y, w, h))
        self.node(id=iid, type="block", kind="plain", x=x, y=y, w=w, h=h, label=[], rx=2, title=caption or file, info="")
        if caption:
            cx, cy = cap_xy or (x + w / 2, y + h + 14)
            self.text(f"cap_{iid}", cx, cy, caption, anchor=anchor)
        return iid

    def panel(self, x, y, w, h, title, tint, dashed=False, title_xy=None):
        """A tinted stage panel; its title goes top-left unless ``title_xy`` places it clear of the arrows."""
        self.panels.append((x, y, w, h, title, tint, dashed, title_xy))

    # -- post-pass
    def _panels_svg(self) -> str:
        out = []
        for x, y, w, h, title, tint, dashed, title_xy in self.panels:
            fill, stroke = TINT[tint]
            dash = ' stroke-dasharray="6 4"' if dashed else ""
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')
            tx, ty = title_xy or (x + 14, y + 21)
            out.append(f'<text x="{tx}" y="{ty}" font-family="{FONT}" font-size="{13 if dashed else 14}" font-weight="bold" '
                       f'fill="{stroke}">{title}</text>')
        return '<g id="panels">\n' + "\n".join(out) + "\n</g>\n"

    def _images_svg(self) -> str:
        out = []
        for f, x, y, w, h in self.images:
            b64 = base64.b64encode((ASSETS / f).read_bytes()).decode()
            out.append(f'<image href="data:image/jpeg;base64,{b64}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice"/>')
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="none" stroke="#555" stroke-width="1"/>')
        return "\n".join(out)

    def _post(self, doc: str) -> str:
        doc = re.sub(r'<g id="(panels|extras)".*?</g>\n', "", doc, flags=re.S)
        # panels right after the page background, so edges and blocks lie on top
        bg = re.search(r'<rect width="\d+" height="\d+" fill="[^"]+"/>\n', doc)
        doc = doc[: bg.end()] + self._panels_svg() + doc[bg.end():]
        # the main data path is drawn heavier
        doc = re.sub(r'(<path id="e\d+" class="edge [^"]*tag-main[^"]*"[^>]*?)stroke-width="1.4"', r'\1stroke-width="2.3"', doc)
        extra = f'<g id="extras" pointer-events="none">\n{chr(10).join(self.extras)}\n{self._images_svg()}\n</g>\n'
        i = doc.index("</svg>")
        return doc[:i] + extra + doc[i:]

    def render(self, **spec) -> None:
        spec = {"name": self.name, "width": self.W, "height": self.H, "nodes": self.N, "edges": self.E, **spec}
        path = HERE / f"{self.name}.json"
        path.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
        subprocess.run([sys.executable, str(SKILL / "scripts" / "render.py"), str(path), "-o", str(HERE)], check=True)
        for ext in ("svg", "html"):
            p = HERE / f"{self.name}.{ext}"
            p.write_text(self._post(p.read_text()))


def token_strip(y: float, groups: list[tuple[str, str, int, bool, float]], note_x: float | None = None) -> str:
    """Prefix tokens as squares (RT-1 style): (label, kind, squares, elided, centre x) per group."""
    out, s, gap = [], 13, 3
    for label, kind, n, more, cx in groups:
        fill, stroke = SQUARE[kind]
        width = n * (s + gap) - gap + (18 if more else 0)
        x = cx - width / 2
        for i in range(n):
            dash = ' stroke-dasharray="3 2"' if kind == "side" else ""
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{s}" height="{s}" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>')
            x += s + gap
            if more and i == n - 3:
                out.append(f'<text x="{x + 3:.1f}" y="{y + 10}" font-family="{FONT}" font-size="12" fill="#666">…</text>')
                x += 18
        out.append(f'<text x="{cx:.1f}" y="{y + 31}" text-anchor="middle" font-family="{MONO}" font-size="10.5" fill="#555">{label}</text>')
    if note_x is not None:
        out.append(f'<line x1="{note_x}" y1="{y - 12}" x2="{note_x}" y2="{y + 20}" stroke="#999" stroke-dasharray="2 2"/>')
        out.append(f'<text x="{note_x - 6}" y="{y - 4}" text-anchor="end" font-family="{FONT}" font-size="10.5" font-style="italic" fill="#666">bidirectional</text>')
        out.append(f'<text x="{note_x + 6}" y="{y - 4}" font-family="{FONT}" font-size="10.5" font-style="italic" fill="#666">causal</text>')
    return "\n".join(out)


# =====================================================================================
# Fig. 1: architecture at inference
# =====================================================================================
def fig_architecture() -> None:
    f = Figure("surgicalvla_arch", 1940, 900)
    # column centres shared by the encoders and their slots in the prefix
    C_IMG, C_TXT, C_OBJ, C_S, C_COT, C_LAT = 190, 460, 695, 935, 1075, 1225
    STRIP_Y = 594

    f.panel(16, 52, 1290, 190, "① Inputs (one frame per camera)", "input")
    f.panel(16, 254, 1290, 262, "② Encoders and projections", "frozen", title_xy=(1012, 290))
    f.panel(578, 268, 298, 240, "frozen perception · frame t only", "percep", dashed=True)
    f.panel(16, 528, 1290, 262, "③ Prefix (≈ 233 tokens × 960) and SmolVLM2 decoder", "frozen", title_xy=(30, 774))
    f.panel(1322, 52, 370, 738, "④ Action expert (trained)", "train")
    f.panel(1708, 52, 216, 738, "⑤ Output", "output")

    # ① inputs
    f.image("third.jpg", 40, 84, 120, 120, "o_t  third-person")
    f.image("wrist.jpg", 190, 84, 120, 120, "o_t  wrist")
    f.node(id="instr", type="block", kind="plain", x=360, y=92, w=200, h=100, size=12,
           label=["instruction  ℓ", "“pick up the black bowl", "next to the cookie box", "and place it on the plate”"],
           info=f"LIBERO-Spatial episode {EX['episode']}. The target phrase “{EX['phrase']}” is read from it and prompts SAM 3.")
    f.image("third.jpg", 640, 84, 120, 120, "o_t  (same frame)", iid="img_third_p")
    f.node(id="state", type="block", kind="tensor", x=880, y=112, w=110, h=60, label=["proprio  s_t", "ℝ^{8} → pad 32"], size=12)
    f.text("in_note", 1030, 128, ["n_obs_steps = 1: no frame history;", "the policy is re-run every 50 steps"], size=11)

    # ② encoders
    f.node(id="siglip", type="block", kind="frozen", x=60, y=300, w=260, h=100,
           label=["SigLIP + pixel-shuffle connector", "512² (resize + pad) → 64 tok / camera", "98.2 M · frozen"],
           shape="[2, 3, 512, 512] → [128, 960]", info="SmolVLM2's vision tower; 32×32 patches pooled 4× by pixel shuffle.")
    f.node(id="tok", type="block", kind="frozen", x=370, y=310, w=180, h=70, label=["tokenizer + embedding", "≤ 48 tokens · frozen"],
           shape="[≤ 48] → [≤ 48, 960]")
    f.node(id="percep", type="block", kind="frozen", x=590, y=298, w=172, h=114,
           label=["SAM 3 → instance masks", "top-3 “black bowl” + EE", "DINOv2-S pooled in mask", "Depth Anything V2-S"],
           params="SAM 3 848 M · DINOv2-S 22 M · DA-V2-S 25 M (outside the policy)",
           info="Per frame and without memory, so object inputs never see the future. Cached for training; 236 ms per chunk live.",
           tags=["objects"])
    f.image("objects.jpg", 772, 298, 96, 96, "K = 4 objects")
    f.node(id="objproj", type="block", kind="train", x=590, y=438, w=210, h=54, label=["object projector · 3.2 M", "[feat 384 ; pose 8] → 1 tok / obj"],
           shape="[4, 392] → [4, 960]; context c_obj ∈ ℝ^{720}", tags=["objects"],
           info="pose = box (4), mask centroid (2), median relative depth, SAM 3 score. Also emits the object context c_obj for the MoE router.")
    f.node(id="cobj", type="tensor", x=812, y=454, w=52, h=20, kind="side", label="", dim="c_obj → ④", tags=["objects"])
    f.node(id="sproj", type="block", kind="train", x=880, y=310, w=110, h=70, label=["state proj", "32 → 960"])
    f.node(id="cotx", type="block", kind="side", x=1010, y=318, w=130, h=70, dashed=True, label=["explicit CoT text", "training only", "(Fig. 3)"],
           info="Teacher-forced during training and faded out by the curriculum; never decoded at inference.")
    f.node(id="lq", type="block", kind="train", x=1160, y=318, w=130, h=70, label=["latent queries", "4 × 960 · learned"],
           info="Latent reasoning tokens; trained to predict the embedding of the frame 1 s ahead (Fig. 3).")

    # ③ prefix and decoder
    f.node(id="strip", type="block", kind="plain", x=40, y=STRIP_Y - 20, w=1250, h=64, label=[], dashed=True, title="prefix tokens",
           shape="≈ 233 × 960", info="images | text | objects | state attend bidirectionally; CoT text and latent queries are causal.")
    f.extras.append(token_strip(STRIP_Y, [("image 2 × 64", "frozen", 12, True, C_IMG), ("text ≤ 48", "frozen", 7, True, C_TXT),
                                         ("objects K = 4", "train", 4, False, C_OBJ), ("state", "train", 1, False, C_S),
                                         ("CoT ≤ 48 (training)", "side", 7, True, C_COT), ("latent 4", "train", 4, False, C_LAT)],
                                note_x=1005))
    f.node(id="vlm", type="block", kind="frozen", x=40, y=680, w=1170, h=62, stack=True, badge="× 16",
           label=["SmolVLM2 decoder · first 16 of 32 layers · d = 960", "self-attention + SwiGLU · 204.6 M · frozen · run once per action chunk"],
           shape="[≈ 233, 960] → K, V per layer", info="Its keys and values are cached and reused by all 10 denoising steps.")
    f.node(id="kv", type="tensor", x=1236, y=676, w=30, h=70, kind="frozen", label="K, V", dim="× 16", shape="16 × [≈ 233, 320]")

    # ④ action expert
    f.node(id="noisy", type="tensor", x=1430, y=716, w=150, h=24, label="", dim="x_τ ∈ ℝ^{50×32},  τ", shape="noise at flow time τ")
    f.node(id="ain", type="block", kind="train", x=1430, y=640, w=150, h=44, label=["action + time MLP", "→ 50 × 720"])
    f.node(id="expert", type="block", kind="train", x=1418, y=262, w=178, h=340, stack=True, badge="× 16",
           label=["action expert", "16 layers", "d = 720", "attends to", "prefix K, V", "MoE FFN", "(Fig. 2)", "381.6 M"],
           strip={"kind": "train", "label": ["MoE"]}, shape="[50, 720] → [50, 720]", params="381.6 M (trainable)",
           info="SmolVLA's action expert with every FFN up-cycled to a mixture of experts.")
    f.node(id="aout", type="block", kind="train", x=1430, y=184, w=150, h=44, label=["action out", "720 → 32"])
    f.node(id="v", type="tensor", x=1480, y=106, w=50, h=40, label="v_θ", dim="", shape="[50, 32]: velocity field")
    f.node(id="cobj2", type="tensor", x=1340, y=300, w=60, h=20, kind="side", label="c_obj", dim="from ②")
    f.node(id="ephase", type="tensor", x=1340, y=384, w=60, h=20, kind="side", label="e_phase", dim="learned null",
           info="At inference no CoT text is fed, so the router's phase input is a learned constant (modeling_smolvla_moe._phase_from_hidden).")

    # ⑤ output
    f.node(id="euler", type="block", kind="plain", x=1724, y=100, w=184, h=50, label=["10 Euler steps", "noise → action chunk"])
    f.node(id="act", type="tensor", x=1766, y=196, w=100, h=40, label="a_{t:t+50}", dim="50 × 7 (LIBERO)")
    f.text("out_n1", 1724, 290, ["execute all 50 actions,", "then re-plan from o_{t+50}"], size=12, muted=False)
    f.text("out_n2", 1724, 350, ["per chunk (RTX A4500):", "policy 242 ms", "perception 236 ms"], size=11)
    f.text("out_n3", 1724, 430, ["at inference:", "latent queries only,", "no text decoding,", "no auxiliary heads"], size=11)

    # edges
    f.edge("img_third.b", "siglip.t-90", main=True)
    f.edge("img_wrist.b", "siglip.t+60", main=True)
    f.edge("instr.b", "tok.t", main=True)
    f.edge("img_third_p.b", "percep.t+24")
    f.edge("percep.b", "objproj.t-19")
    f.edge("objproj.r", "cobj.l", style="dashed")
    f.edge("state.b", "sproj.t")
    f.edge("siglip.b", f"strip.t{C_IMG - 665:+d}", main=True)
    f.edge("tok.b", f"strip.t{C_TXT - 665:+d}", main=True)
    f.edge("objproj.b", f"strip.t{C_OBJ - 665:+d}", main=True)
    f.edge("sproj.b", f"strip.t{C_S - 665:+d}")
    f.edge("cotx.b", f"strip.t{C_COT - 665:+d}", style="dashed")
    f.edge("lq.b", f"strip.t{C_LAT - 665:+d}")
    f.edge("strip.b-40", "vlm.t", main=True)
    f.edge("vlm.r", "kv.l", main=True)
    f.edge("kv.r", "expert.l+135", via=[["1360", "kv.cy"], ["1360", "expert.cy+135"]], main=True, label="prefix K, V")
    f.edge("noisy.t", "ain.b", main=True)
    f.edge("ain.t", "expert.b", main=True)
    f.edge("expert.t", "aout.b", main=True)
    f.edge("aout.t", "v.b", main=True)
    f.edge("cobj2.r", "expert.l-122", style="dashed")
    f.edge("ephase.r", "expert.l-38", style="dashed")
    f.edge("v.r", "euler.l", main=True)
    f.edge("euler.b", "act.t", main=True)

    f.render(
        title="Fig. 1. SurgicalVLA at inference: past-only object tokens and latent reasoning in a frozen SmolVLM2 prefix; a phase- and object-conditioned MoE action expert",
        legend=["frozen", "train", "tensor", ["side", "conditioning / training-only text"]],
        caption=f"Example: LIBERO-Spatial episode {EX['episode']}, frame t = {EX['frame']}. 792 M parameters, 442 M trained; "
                "the VLM and the perception models stay frozen. Thick arrows: main data path. Fig. 2 details one expert layer; Fig. 3 the training.",
        tags=[{"tag": "objects", "label": "object tokens", "info": "Past-only perception → one token per object, plus the router's object context."}],
        steps=[
            {"nodes": ["img_third", "img_wrist", "siglip", "instr", "tok", "state", "sproj"], "text": "Cameras, instruction and state are embedded; SmolVLM2's encoders are frozen."},
            {"nodes": ["img_third_p", "percep", "img_objects", "objproj", "cobj"], "text": "Frozen perception reads only frame t: 3 instances of the target phrase and the end effector become 4 object tokens."},
            {"nodes": ["strip", "lq", "cotx"], "text": "The prefix: images | text | objects | state, then (training only) CoT text, then 4 latent queries."},
            {"nodes": ["vlm", "kv"], "text": "The frozen decoder runs once per chunk; its keys and values are cached."},
            {"nodes": ["noisy", "ain", "expert", "aout", "v", "cobj2", "ephase"], "text": "The trained MoE action expert predicts the flow velocity, attending to the cached prefix."},
            {"nodes": ["euler", "act"], "text": "10 Euler steps turn noise into a 50-step action chunk, executed in full before re-planning."},
        ],
    )


# =====================================================================================
# Fig. 2: one action-expert layer
# =====================================================================================
def fig_expert_layer() -> None:
    f = Figure("surgicalvla_expert_layer", 1120, 720)
    f.panel(16, 52, 1088, 470, "One action-expert layer ℓ  (× 16, d = 720)", "train")

    f.node(id="hin", type="tensor", x=36, y=290, w=70, h=26, label="h", dim="50 × 720")
    f.node(id="kv", type="tensor", x=150, y=104, w=110, h=24, kind="frozen", label="prefix K, V (layer ℓ)", dim="from the frozen decoder")
    f.node(id="attn", type="block", kind="train", x=140, y=240, w=200, h=126,
           label=["RMSNorm → attention", "ℓ even: self-attention over", "[prefix ; chunk]", "ℓ odd: cross-attention to", "the prefix K, V", "1.84 M"],
           info="SmolVLA's interleaving (self_attn_every_n_layers = 2).")
    f.node(id="add1", type="op", x=372, y=291, w=24, label=["+"])
    f.node(id="ephase", type="tensor", x=440, y=104, w=80, h=22, kind="side", label="e_phase ∈ ℝ^{720}", dim="")
    f.node(id="cobj", type="tensor", x=560, y=104, w=80, h=22, kind="side", label="c_obj ∈ ℝ^{720}", dim="")
    f.node(id="router", type="block", kind="train", x=430, y=248, w=220, h=110,
           label=["RMSNorm → router (fp32)", "[h ; e_phase ; c_obj]", "ℝ^{2160} → 4 logits", "top-2 per action token", "9 k parameters"],
           info="Routes each action token by its content, the manipulation phase and the target object. Switch load-balancing loss, weight 0.01.")
    for i, sel in enumerate([True, False, True, False]):
        f.node(id=f"e{i + 1}", type="block", kind="train" if sel else "mask", x=700, y=150 + i * 56, w=160, h=42,
               label=[f"expert {i + 1}" + ("  ✓" if sel else ""), "SwiGLU 720 → 2048 → 720"], size=12,
               info="Routed expert initialised from SmolVLA's FFN (up-cycling)." + (" Selected for this token." if sel else ""))
    f.node(id="shared", type="block", kind="train", x=700, y=390, w=160, h=42, label=["shared expert", "always on"], size=12)
    f.node(id="mix", type="op", x=900, y=287, w=30, label=["Σ"], info="Shared output + gate-weighted sum of the 2 selected experts.")
    f.node(id="add2", type="op", x=968, y=289, w=26, label=["+"])
    f.node(id="hout", type="tensor", x=1020, y=290, w=70, h=26, label="h′", dim="→ layer ℓ+1")

    f.text("n_phase", 36, 556, ["e_phase: phase_proj of the decoder output at the CoT phase token when the trace is fed;",
                                "a learned null vector otherwise, which includes every inference step after the curriculum."], size=12)
    f.text("n_obj", 36, 600, ["c_obj: the object projector's context over the K = 4 object tokens (Fig. 1 ②).",
                              "FFN per layer 22.1 M (4 routed + 1 shared); active per token: shared + 2 of 4 = 3× the SmolVLA FFN compute."], size=12)

    f.edge("hin.r", "attn.l", main=True)
    f.edge("kv.b", "attn.t", main=True)
    f.edge("attn.r", "add1.l", main=True)
    f.edge("hin.t", "add1.t", via=[["71", "214"], ["384", "214"]], style="dashed")  # residual
    f.edge("add1.r", "router.l", main=True)
    f.edge("ephase.b", "router.t-60", style="dashed")
    f.edge("cobj.b", "router.t+60", style="dashed")
    for i in range(4):
        f.edge("router.r", f"e{i + 1}.l", main=i in (0, 2), style="solid" if i in (0, 2) else "dashed")
        f.edge(f"e{i + 1}.r", "mix.l", main=i in (0, 2), style="solid" if i in (0, 2) else "dashed")
    f.edge("add1.b", "shared.l", via=[["384", "411"]], main=True)
    f.edge("shared.r", "mix.l", main=True)
    f.edge("mix.r", "add2.l", main=True)
    f.edge("add1.b", "add2.b", via=[["384", "480"], ["981", "480"]], style="dashed")  # residual
    f.edge("add2.r", "hout.l", main=True)

    f.render(
        title="Fig. 2. One layer of the action expert: interleaved attention to the frozen prefix and a mixture-of-experts FFN routed on content, phase and object",
        legend=["train", "frozen", "tensor", ["side", "router conditioning"], ["mask", "expert not selected"]],
        caption="All routed experts start as copies of SmolVLA's FFN; the dense variant (moe_variant = dense) is SmolVLA.",
    )


# =====================================================================================
# Fig. 3: training method
# =====================================================================================
def fig_training() -> None:
    f = Figure("surgicalvla_training", 1940, 880)
    f.panel(16, 52, 690, 760, "① Label-free pseudo-labels (frozen models, run once offline)", "label")
    f.panel(722, 52, 736, 760, "② Supervised fine-tuning (50 k steps)", "train")
    f.panel(1474, 52, 450, 760, "③ Online RL: ReinFlow-GRPO", "rl")

    # ① pseudo-labelling
    f.image("third.jpg", 40, 90, 110, 110, "demonstration, frontal camera")
    f.node(id="instr", type="block", kind="plain", x=170, y=96, w=240, h=56, label=["instruction", "“pick up the black bowl next to …”"], size=12)
    f.node(id="phrase", type="block", kind="frozen", x=440, y=90, w=240, h=70,
           label=["target phrase", "rule + grounded SmolVLM2 check", f"→ “{EX['phrase']}”"],
           info="Rule: the object of the first manipulation verb; SmolVLM2 answers from the first frame; the rule wins on disagreement.")
    f.node(id="acts", type="tensor", x=560, y=250, w=110, h=22, label="", dim="demo actions a_{1:T}")
    f.node(id="sam", type="block", kind="frozen", x=170, y=230, w=330, h=64,
           label=["SAM 3 video tracking", "target and end-effector masks, boxes, scores"])
    f.node(id="dino", type="block", kind="frozen", x=40, y=340, w=150, h=56, label=["DINOv2-S", "target feature"])
    f.node(id="depth", type="block", kind="frozen", x=205, y=340, w=150, h=56, label=["Depth Anything V2-S", "gripper-relative offset"])
    f.node(id="tap", type="block", kind="frozen", x=370, y=340, w=150, h=56, label=["TAPNext++", "point tracks"])
    f.node(id="events", type="block", kind="plain", x=535, y=340, w=150, h=56, label=["gripper events", "+ speed minima → phases"])
    f.node(id="cache", type="block", kind="target", x=40, y=446, w=645, h=112,
           label=["feature cache (Parquet, keyed by episode and frame)",
                  "inputs: K = 4 object descriptors on every frame, from frame t only",
                  "targets: every 4th frame, may use the whole episode, each with a confidence",
                  "CoT traces written from the labels: phase · object · motion",
                  "the manifest pins model hashes, prompts, thresholds, strides and camera"])
    f.image("targets.jpg", 40, 590, 116, 116, "target mask · tracks · gripper")
    f.text("trace", 176, 612, ["trace at t = 36:", EX["trace"]], size=11, muted=False, mono=False)
    f.text("lab_n", 176, 668, ["No human labels. Simulator ground truth only validates the", "pseudo-labels and drives the oracle ablation."], size=11)

    f.edge("img_third.r", "instr.l", via=[["160", "145"], ["160", "124"]])
    f.edge("instr.r", "phrase.l")
    f.edge("phrase.b", "sam.t+100", via=[["560", "200"], ["435", "200"]])
    f.edge("img_third.b", "sam.l", via=[["95", "262"]])
    f.edge("sam.b-140", "dino.t", via=[["195", "318"], ["115", "318"]])
    f.edge("sam.b-30", "depth.t", via=[["305", "318"], ["280", "318"]])
    f.edge("sam.b+110", "tap.t", via=[["445", "318"]])
    f.edge("acts.b", "events.t", via=[["615", "318"], ["610", "318"]])
    for src in ("dino", "depth", "tap", "events"):
        f.edge(f"{src}.b", f"cache.t{ {'dino': -247, 'depth': -82, 'tap': 82, 'events': 247}[src]:+d}")

    # ② SFT
    f.node(id="policy", type="block", kind="train", x=746, y=92, w=690, h=74,
           label=["SurgicalVLA (Fig. 1), one forward pass per sample", "inputs: cameras, ℓ, s_t, cached object descriptors, CoT trace (curriculum), noisy chunk x_τ"],
           info="Frozen VLM; the action expert, object projector, latent queries and heads are trained.")
    cols = {"fm": 790, "aux": 906, "cot": 1022, "ph": 1138, "lat": 1254, "bal": 1370}
    heads = {"aux": ["aux heads", "mean expert out"], "cot": ["LM head", "CoT-token out"], "ph": ["phase head", "e_phase / pool"],
             "lat": ["predictor", "latent-query out"]}
    for k, lab in heads.items():
        f.node(id=f"h_{k}", type="block", kind="train", x=cols[k] - 54, y=212, w=108, h=50, label=lab, size=12)
    losses = {"fm": ["ℒ_{FM}", "‖v_{θ} − (ε − a)‖² · 1"], "aux": ["ℒ_{aux}", "0.1 ×3, 0.02 (rel)"], "cot": ["ℒ_{CoT}", "token CE · 0.1"],
              "ph": ["ℒ_{phase}", "CE · 0.1"], "lat": ["ℒ_{latent}", "1 − cos · 0.1"], "bal": ["ℒ_{balance}", "Switch · 0.01"]}
    for k, lab in losses.items():
        f.node(id=f"l_{k}", type="block", kind="loss", x=cols[k] - 54, y=300, w=108, h=50, label=lab, size=12)
    targets = {"fm": ["ε − a", "demo chunk"], "aux": ["feat · motion", "track · rel"], "cot": ["trace", "tokens"], "ph": ["phase", "labels"],
               "lat": ["z_{t+10}", "stop-grad"]}
    for k, lab in targets.items():
        f.node(id=f"t_{k}", type="block", kind="target", x=cols[k] - 50, y=388, w=100, h=44, label=lab, size=12)
    f.node(id="enc", type="block", kind="frozen", x=cols["lat"] - 54, y=470, w=108, h=50, label=["vision encoder", "frozen"], size=12,
           info="With the vision encoder frozen the target encoder is the encoder itself; an EMA copy (decay 0.99) exists only when it trains.")
    f.image("future.jpg", cols["lat"] - 40, 548, 80, 80, "o_{t+10} (1 s ahead)")
    f.text("sft_total", 746, 490, ["ℒ_{SFT} = ℒ_{FM} + Σ_{k} w_{k} ℒ_{aux,k} + 0.1 ℒ_{CoT}", "          + 0.1 ℒ_{phase} + 0.1 ℒ_{latent} + 0.01 ℒ_{balance}"],
           size=13, muted=False)
    f.text("sft_n", 746, 548, ["Auxiliary losses read the action expert (aux_source = expert),", "so they shape features the policy acts on while the VLM is frozen.",
                               "Batch 24 × 4 accumulation; seed-fixed episode split."], size=11)
    f.text("cur_t", 746, 626, "explicit → latent curriculum", size=12, muted=False, bold=True)
    # curriculum plot: p(explicit CoT) falls 1 -> 0 over the first 25 k of 50 k steps
    x0, y0, w, h = 790, 760, 300, 96
    f.extras.append("\n".join([
        f'<line x1="{x0}" y1="{y0}" x2="{x0 + w}" y2="{y0}" stroke="#555"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 - h}" stroke="#555"/>',
        f'<polyline points="{x0},{y0 - h} {x0 + w / 2},{y0} {x0 + w},{y0}" fill="none" stroke="#7a52a0" stroke-width="2.2"/>',
        f'<polyline points="{x0},{y0} {x0 + w / 2},{y0 - h} {x0 + w},{y0 - h}" fill="none" stroke="#d17a22" stroke-width="2.2" stroke-dasharray="6 4"/>',
        f'<text x="{x0 - 6}" y="{y0 - h + 4}" text-anchor="end" font-family="{FONT}" font-size="11" fill="#555">1</text>',
        f'<text x="{x0 - 6}" y="{y0 + 4}" text-anchor="end" font-family="{FONT}" font-size="11" fill="#555">0</text>',
        f'<text x="{x0 + w / 2}" y="{y0 + 16}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="#555">25 k</text>',
        f'<text x="{x0 + w}" y="{y0 + 16}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="#555">50 k steps</text>',
        f'<text x="{x0 + w + 14}" y="{y0 - h + 4}" font-family="{FONT}" font-size="11" fill="#d17a22">latent only</text>',
        f'<text x="{x0 + 10}" y="{y0 - h - 8}" font-family="{FONT}" font-size="11" fill="#7a52a0">p(feed explicit CoT text)</text>',
    ]))

    f.edge("cache.r", "policy.l", via=[["712", "cache.cy"], ["712", "policy.cy"]], main=True)
    for k in cols:
        if k in heads:
            f.edge(f"policy.b{cols[k] - 1091:+d}", f"h_{k}.t", main=True)
            f.edge(f"h_{k}.b", f"l_{k}.t", main=True)
        else:
            f.edge(f"policy.b{cols[k] - 1091:+d}", f"l_{k}.t", main=True, label="v_{θ}" if k == "fm" else "router stats")
    for k in targets:
        f.edge(f"t_{k}.t", f"l_{k}.b", style="dashed")
    for k in ("aux", "cot", "ph"):
        f.edge("cache.r", f"t_{k}.b", via=[["712", "cache.cy"], ["712", "450"], [f"t_{k}.cx", "450"]], style="dashed")
    f.edge("enc.t", "t_lat.b", style="dashed")
    f.edge("img_future.t", "enc.b")

    # ③ RL
    f.node(id="pi", type="block", kind="train", x=1500, y=96, w=190, h=60, label=["policy π_θ", "starts from the SFT model"])
    f.node(id="ref", type="block", kind="frozen", x=1710, y=478, w=190, h=44, label=["reference π_SFT", "frozen copy"])
    f.node(id="sim", type="block", kind="plain", x=1500, y=212, w=190, h=78,
           label=["LIBERO simulator", "fresh seeded layouts;", "never the 50 evaluation", "init states"])
    f.node(id="roll", type="block", kind="plain", x=1710, y=212, w=190, h=78,
           label=["G = 8 rollouts per start", "SDE sampling, 5 steps,", "learned σ ∈ [0.01, 0.15]", "(ReinFlow noise head)"])
    f.node(id="rew", type="block", kind="target", x=1710, y=330, w=190, h=44, label=["r_{i} = 1[success]"])
    f.node(id="adv", type="block", kind="plain", x=1710, y=410, w=190, h=50, label=["A_{i} = (r_{i} − mean) / std", "within the group"])
    f.node(id="lrl", type="block", kind="loss", x=1500, y=330, w=190, h=130,
           label=["ℒ_{RL}", "−min(ρA, clip(ρ, 0.8, 1.2) A)", "+ 0.01 · KL_{k3}(π_θ ‖ π_SFT)", "− 0.001 · entropy", "ρ from exact SDE", "transition log-probs"])
    f.node(id="modes", type="block", kind="plain", x=1500, y=556, w=400, h=118,
           label=["training.mode", "sequential (default): RL updates only", "joint: + λ_t ℒ_{FM} on demo batches, λ 1 → 0.1 (cosine)",
                  "interleaved: FM steps between RL updates", "200 iterations · lr 1e-6 · chunks of 50 executed"], size=12)
    f.text("rl_n", 1500, 708, ["Evaluation always uses the deterministic ODE (10 Euler steps)", "and the model saved after the fixed final iteration."], size=11)

    f.edge("pi.b", "sim.t", main=True, label="act")
    f.edge("sim.r", "roll.l", main=True)
    f.edge("roll.b", "rew.t", main=True)
    f.edge("rew.b", "adv.t", main=True)
    f.edge("adv.l", "lrl.r+40", main=True)
    f.edge("ref.l", "lrl.r+58", via=[["1700", "ref.cy"], ["1700", "lrl.cy+58"]], style="dashed")
    f.edge("lrl.l", "pi.l", via=[["1488", "lrl.cy"], ["1488", "pi.cy"]], main=True)
    f.edge("policy.r", "pi.l", via=[["1462", "policy.cy"], ["1462", "pi.cy-14"], ["1500", "pi.cy-14"]], main=True)

    f.render(
        title="Fig. 3. Training: label-free pseudo-labels from frozen foundation models, multi-objective SFT with an explicit-to-latent reasoning curriculum, then ReinFlow-GRPO",
        legend=["frozen", "train", ["target", "pseudo-label / target"], "loss", "tensor"],
        caption="Inputs to the policy are past-only; targets may use the whole episode and enter only the losses. "
                "Thick arrows: the main path; dashed: targets.",
    )


if __name__ == "__main__":
    for make in (fig_architecture, fig_expert_layer, fig_training):
        make()
