"""Auto direction routing: grayscale DAPI -> dapi2he, stained H&E -> he2dapi."""
import numpy as np

from src.models.infer import detect_direction, resolve_direction


def test_grayscale_routes_to_dapi2he():
    assert detect_direction(np.zeros((32, 32, 3), dtype=np.uint8)) == "dapi2he"
    assert detect_direction(np.zeros((32, 32), dtype=np.uint8)) == "dapi2he"
    assert detect_direction(np.zeros((32, 32, 1), dtype=np.uint8)) == "dapi2he"


def test_stained_image_routes_to_he2dapi():
    he = np.zeros((32, 32, 3), dtype=np.uint8)
    he[..., 0] = 210  # eosin/red
    he[..., 1] = 120
    he[..., 2] = 170  # haematoxylin/blue
    assert detect_direction(he) == "he2dapi"


def test_resolve_passthrough_and_auto():
    gray = np.zeros((16, 16, 3), dtype=np.uint8)
    assert resolve_direction("he2dapi", gray) == "he2dapi"
    assert resolve_direction("dapi2he", gray) == "dapi2he"
    assert resolve_direction("auto", gray) == "dapi2he"
