"""``lerobot-eval`` that also knows SurgicalVLA's ``smolvla_moe`` policy.

LeRobot resolves a policy type through its draccus-registered config class and
the sibling ``modeling_*`` module (``SmolVLAMoEConfig`` ->
``surgicalvla.modules.modeling_smolvla_moe.SmolVLAMoEPolicy``), so importing our
config module is enough; evaluation itself is LeRobot's, unchanged.

    python scripts/lerobot_eval.py <any lerobot-eval arguments>
"""

import surgicalvla.modules.configuration_smolvla_moe  # noqa: F401  (registers "smolvla_moe")
from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()
