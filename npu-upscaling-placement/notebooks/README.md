# Notebooks

Both run top to bottom in Google Colab on a T4. Set Runtime → Change runtime
type → GPU first.

**`01_first_model.ipynb`** — exploratory analysis, the first model, and the INT8
export used for the placement and latency measurements. Trains on 90 images from
the DIV2K **validation** split. This model loses to bicubic; see the README.

**`02_corrected_model.ipynb`** — the corrected model with the anchor residual,
trained on 120 images from the DIV2K training split. Checks PSNR against bicubic
every epoch and refuses to export a model that loses to it. Produces the quality
results.

Each downloads its own data, so nothing needs placing by hand.
