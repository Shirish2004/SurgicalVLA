"""infer_features declares images channel-first, as lerobot-eval's checkpoint check expects."""

from types import SimpleNamespace

from lerobot.configs.types import FeatureType

from surgicalvla.data.dataset import channel_first, infer_features


def test_channel_first_uses_names_and_falls_back_to_shape():
    assert channel_first((256, 256, 3), ["height", "width", "channel"]) == (3, 256, 256)
    assert channel_first((256, 256, 3), ["height", "width", "channels"]) == (3, 256, 256)
    assert channel_first((3, 224, 224), ["channels", "height", "width"]) == (3, 224, 224)
    assert channel_first((480, 640, 3)) == (3, 480, 640)
    assert channel_first((3, 480, 640)) == (3, 480, 640)
    assert channel_first((8,)) == (8,)


def test_infer_features_images_are_channel_first():
    meta = SimpleNamespace(features={
        "observation.images.image": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": [8], "names": None},
        "action": {"dtype": "float32", "shape": [7], "names": None},
    })
    f = infer_features(meta)
    assert f["observation.images.image"].type is FeatureType.VISUAL and f["observation.images.image"].shape == (3, 256, 256)
    assert f["observation.state"].shape == (8,) and f["action"].shape == (7,)
