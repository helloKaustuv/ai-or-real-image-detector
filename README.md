# 🔍 AI or Real? — Image Detector Website

A website built **entirely in Python** that detects whether an uploaded
image is **AI-generated** or a **real photo**, with a confidence score.

## How it works

1. You upload a JPG / PNG / WebP image on the website.
2. The image is passed to a pretrained Vision Transformer model from
   Hugging Face: [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
3. The site shows the verdict (AI-generated / Real) plus how strongly
   the model scored each side.

No model training is needed for this prototype — the pretrained model
is downloaded automatically on the first run (~350 MB).

## Files

| File           | What it does                                             |
| -------------- | -------------------------------------------------------- |
| `app.py`       | The whole website (Streamlit)                            |
| `detector.py`  | The detection logic, separate from the UI — also runnable from the command line |
| `requirements.txt` | Python dependencies                                  |
| `.streamlit/config.toml` | Theme colours for the website                  |

## The interface

- A monochrome design on a dark background — black, white and greys only
- A dark hero header explaining the site at a glance
- A 3-step "how it works" strip (Upload → AI analysis → Verdict)
- A sidebar with tips for a fair test and a privacy note
- Side-by-side layout after upload: image preview + file details,
  then a verdict card (solid border = real, dashed border = AI-generated)
- Confidence shown as metrics and bars, with a plain-English strength
  guide (strong / moderate / weak signal)
- A "recent checks" history for the session and an explainer section

## Run it locally

```bash
cd ai-image-detector
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

> Tip for a smaller/faster install: install the CPU-only PyTorch build
> first with
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Test the detector without the website

```bash
python detector.py path/to/any-image.jpg
```

## Put it online (free)

**Streamlit Community Cloud** fits this project best:

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and choose the repo and `app.py`.
3. Deploy — Streamlit installs `requirements.txt` and gives you a
   public `*.streamlit.app` link.

Note: the free tier has limited RAM, so the first load can be slow while
the model downloads.

## Limitations — read this before trusting a result

- No AI-image detector is perfect. Edited, cropped, screenshot, or
  heavily compressed images fool detectors often.
- The model was trained on particular generators and photo datasets, so
  newer generators can score differently.
- Treat the output as a **signal, not proof**, especially when the
  confidence shown is below ~65%.

## Ideas to improve it next

- Try other pretrained detectors and show an agreement score
  (e.g. `prithivMLmods/Deep-Fake-Detector-v2-Model`).
- Fine-tune the model on your own set of real vs AI images.
- Add a history of checked images and example test images.
