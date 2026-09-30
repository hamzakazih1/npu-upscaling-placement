import pytest

from npu_upscale.chips import parse_chip


@pytest.mark.parametrize("text, name, model, generation, tops", [
    # Windows ProcessorNameString, the SKU used in the study
    ("Snapdragon(R) X Plus - X1P42100 - Qualcomm(R) Oryon(TM) CPU",
     "Snapdragon X Plus", "X1P-42-100", 1, 45),
    ("Snapdragon(R) X Plus - X1P64100 - Qualcomm(R) Oryon(TM) CPU",
     "Snapdragon X Plus", "X1P-64-100", 1, 45),
    ("Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Oryon(TM) CPU",
     "Snapdragon X Elite", "X1E-80-100", 1, 45),
    ("Snapdragon(R) X 12-core X1E78100 @ 3.40 GHz",
     "Snapdragon X Elite", "X1E-78-100", 1, 45),
    ("Snapdragon(R) X - X126100 - Qualcomm(R) Oryon(TM) CPU",
     "Snapdragon X", "X1-26-100", 1, 45),
    ("Snapdragon X1E-00-1DE dev kit", "Snapdragon X Elite", "X1E-00-1DE", 1, 45),
    # Linux device tree
    ("qcom,x1e80100-crd qcom,x1e80100", "Snapdragon X Elite", "X1E-80-100", 1, 45),
    ("qcom,x1p42100", "Snapdragon X Plus", "X1P-42-100", 1, 45),
    # Second generation
    ("Snapdragon(R) X2 Elite - X2E88100", "Snapdragon X2 Elite", "X2E-88-100", 2, 80),
    ("Snapdragon(R) X2 Elite Extreme - X2E-96-100", "Snapdragon X2 Elite", "X2E-96-100", 2, 80),
    ("Snapdragon(R) X2 Plus - X2P42100", "Snapdragon X2 Plus", "X2P-42-100", 2, 80),
    # Marketing names without a model number
    ("Snapdragon X Elite", "Snapdragon X Elite", None, 1, 45),
    ("Snapdragon X Plus", "Snapdragon X Plus", None, 1, 45),
    ("Snapdragon X2 Elite", "Snapdragon X2 Elite", None, 2, 80),
])
def test_snapdragon_x_family(text, name, model, generation, tops):
    chip = parse_chip(text)
    assert chip.is_snapdragon_x
    assert (chip.name, chip.model, chip.generation, chip.npu_tops) == (name, model, generation, tops)


@pytest.mark.parametrize("text", [
    "Intel(R) Core(TM) Ultra 7 155H",
    "AMD Ryzen AI 9 HX 370",
    "Snapdragon 8 Gen 3",
    "ARMv8 (64-bit) Family 8 Model 1 Revision 201, Qualcomm Technologies Inc",
    "",
])
def test_other_chips_are_not_misidentified(text):
    chip = parse_chip(text)
    assert not chip.is_snapdragon_x
    assert chip.slug == "generic"


def test_slug_keys_the_cache_per_sku():
    assert parse_chip("X1P42100").slug == "x1p-42-100"
    assert parse_chip("X1E80100").slug != parse_chip("X1P42100").slug
    assert parse_chip("Snapdragon X Elite").slug == "x1e-unknown"


def test_override(monkeypatch):
    from npu_upscale.chips import detect_chip
    monkeypatch.setenv("NPU_UPSCALE_CHIP", "X1E84100")
    chip = detect_chip()
    assert chip.model == "X1E-84-100" and chip.source == "NPU_UPSCALE_CHIP"
