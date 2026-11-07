import json
from pathlib import Path
from tests.reference_capture import capture


def test_pinned_demo_synthetic_outputs():
    expected = json.loads((Path(__file__).parents[1]/'fixtures'/'demo_regression.json').read_text(encoding='utf-8'))
    assert capture() == expected
