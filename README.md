# Deep Learning for Maritime Vision Applications: Notebooks

*Torben Globisch · Chair for Visual and Analytic Computing in Ocean Technologies (VACOT)*

Interactive Jupyter/Colab versions of the introductory seminar sessions, written as **show-and-tell** material for the
lecture: the presenter runs them top to bottom, there are no exercises. Each notebook mixes short explanations with
figures in the style of the slides, **animations** you can step through frame by frame, and live demos on **maritime photos**.

## Notebooks

| # | Notebook | Session | Highlights | Open |
|---|---|---|---|---|
| 00 | **Artificial Neural Networks**: neurons, activations, learning | 0 · Part 1 | one running example (iris petals) · neuron playground: decision line + live loss landscape · why non-linearity: a hand-built 2 → 2 → 1 network · gradient-descent animation · backpropagation animation, layer by layer · the training loop step by step on mini-batches of 5 flowers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TorbenGl/visionseminar/blob/main/notebooks/00_neural_networks_basics.ipynb) |
| 01 | **Computer Vision & an MLP on MNIST** | 0 · Part 2 | images as numbers · classification / detection / segmentation / depth run live on boat photos · demo: an MLP on MNIST · why MLPs fail on images: shifting a digit by 2–3 px breaks it (animation) · shuffled-pixel experiment · scaling to 224 × 224 × 3: parameters, memory, measured training time, out of memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TorbenGl/visionseminar/blob/main/notebooks/01_computer_vision_and_mlp_mnist.ipynb) |
| 02 | **Convolutional Neural Networks** | 0 · Part 3 | mini-project: hand-crafted kernels on the 224 × 224 boat photos → **learning the Sobel kernel** by gradient descent (animation) → CNN on MNIST · animated Sobel walkthrough (all 36 positions) · stride/padding & max-pool animations · translation equivariance animation · parameters at 224 × 224: CNN vs. MLP · **the CNN beats the MLP** with fewer parameters · MLP vs. CNN on shifted digits (animation) · ResNet feature hierarchy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TorbenGl/visionseminar/blob/main/notebooks/02_convolutional_neural_networks.ipynb) |
| 03 | **Vision Foundation Models: the CNN era** | 1 · Part 1 | short recap: a kernel learned from data (notebook 02) · stride = K → patches · ImageNet history · plain vs. residual networks experiment · ResNet-50 features of **six vessel types** (our maritime dataset): t-SNE, retrieval · **race: linear probe vs. training from scratch** (animation) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TorbenGl/visionseminar/blob/main/notebooks/03_vision_foundation_models.ipynb) |
| 04 | **Vision Transformers** | 1 · Part 2 | self-attention from scratch + animation · patchification animation · "patch embedding = conv" proof · learned positional embeddings · demo: a mini-ViT trained on MNIST · [CLS] attention through 12 layers, and why it looks at the sky (artifact tokens) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TorbenGl/visionseminar/blob/main/notebooks/04_vision_transformers.ipynb) |
| 05 | **Self-supervised pretraining & transfer** | 1 · Part 3 | mask-ratio animation + real MAE reconstructions · DINO multi-crop & attention heads · DINOv2 PCA: boat segmentation with zero labels · first-layer filters: random vs. supervised vs. self-supervised · frozen benchmarking on the vessel dataset (k-NN & linear) · **race: linear head vs. partial vs. full fine-tuning** with 50 labels per class (animation) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TorbenGl/visionseminar/blob/main/notebooks/05_self_supervised_learning_and_transfer.ipynb) |

> The Colab badges point to the `main` branch of this repository. If the repository is private, Colab asks you to
> authorise GitHub the first time (or use *File → Upload notebook*). For students, either make the repo public or
> share copies via Google Drive.

## Using the notebooks in the lecture

* **GPU:** *Runtime → Change runtime type → T4 GPU*. Notebooks 00 and 02 run fine on CPU; 03–05 really want the GPU.
* **Run everything once before the lecture** (*Runtime → Run all*). This downloads datasets and pretrained weights
  (up to ~600 MB for notebook 05), so nothing stalls on stage.
* **Animations** render as an HTML player: pause it and use the ◀ ▶ step buttons to walk through e.g. the Sobel
  computation one kernel position at a time.
* **No exercises:** every notebook is a demo. 🎤 marks prompts for the presenter: questions for the audience (the answer
  is folded away, click to reveal) or a value worth changing live.
* Figure-drawing code is hidden in collapsed cells (🎨 / 🎞️) so the visible code stays focused on the ideas.
* Approximate run time on a Colab T4: 00 ~1 min · 01 ~1.5 min · 02 ~2.5 min · 03 ~2 min · 04 ~2 min · 05 ~3–4 min (03–05 estimated, not yet timed on a T4).

## How the material was restructured

The two slide decks were turned into six notebooks, three per session, each with a single theme.

All six notebooks are **show-and-tell**: the former exercises are now demos (the solution code runs in place), and the
"try this" prompts became 🎤 questions for the audience with a folded answer.

**Session 0 (ANNs & CNNs)**
* Notebook 00 follows one running example, the iris petals: a single neuron solved by hand in a
  playground (Task A works, Task B can't be solved with one line) → a hand-built 2 → 2 → 1 network shows why hidden layers
  need a non-linearity → gradient descent, backpropagation (layer by layer) and the training loop as animations. The old
  Iris exercise was removed.
* The **Sobel walkthrough** moved here from the Session 1 recap. It is the clearest intuition for convolution, so it now
  sits where convolution is introduced (as one animation instead of five slides).
* Notebook 01 makes the case against MLPs on images with three experiments: moving the digits by 2–3 px (accuracy collapses),
  shuffling the pixels (the MLP doesn't care), and scaling to 224 × 224 × 3 (parameter count, training memory, measured
  time per step, and an out-of-memory attempt for an MLP with the same shape as the MNIST one).
* Notebook 02 is organised as a mini-project: hand-crafted kernels on the 224 × 224 photos → learn the Sobel kernel from
  input–target pairs → a CNN on MNIST with fewer parameters, which also survives the shift experiment from notebook 01.
* Stride/padding is taught **once** (notebook 02) instead of in both decks.
* New experiment that ties the session together: an MLP doesn't care if you **shuffle the pixels** (notebook 01), a CNN does
  (notebook 02). This makes "spatial structure" and "inductive bias" concrete, and it pays off again in the ViT notebook.

**Session 1 (Foundation Models & ViT)**
* **Running example: a small maritime dataset.** 450 photos of six vessel types (sailboat, speedboat, ferry, cruise ship,
  cargo/container ship, fishing vessel) from Open Images, with human-verified labels; 50 training and 25 test photos per
  class. It replaces CIFAR-10 in notebooks 03 and 05, so "learning from data" is shown on maritime data with a realistic number of
  labels. The image ids are embedded in the notebooks and the photos are downloaded at runtime (≈ 90 MB, cached).
* Learning is shown as it happens: an epoch-by-epoch race between a linear probe and a CNN trained from scratch (03), and between
  the four adaptation strategies (05). Notebook 05 also shows what random, supervised and self-supervised backbones learned in
  their very first layer, the bridge back to the Sobel kernel of notebook 02.
* The recap is short and builds the bridge to the new material: one static figure of the kernel that notebook 02 learned
  from data (same photos, same start; no second animation), then stride = K → patches → tokens.
* Every claim gets a small experiment: skip connections (plain vs. residual at depth 8 and 50), the value of pretraining
  (linear probe vs. training from scratch), supervised vs. self-supervised features (frozen benchmark), and the adaptation
  spectrum (linear / MLP head / partial / full fine-tuning).
* Real maritime examples throughout (COCO photos of trawlers, sailboats, lifeboats, ferries, marinas).

## Corrections relative to the slides

While converting I fixed a few inconsistencies; you may want to update the PDFs too:

| Slide | Issue | In the notebooks |
|---|---|---|
| S0 p.16 vs. S1 p.3 | "translation **invariance**" vs. "translation **equivariance**" | convolution is *equivariant*; pooling adds (some) *invariance*, shown in an animation (02) |
| S0 p.16 | ANN total "∼234,752" (weights only, no biases) vs. 235,146 on p.12/18; CNN total "∼108,170" mixes counts with and without biases and leaves out the last FC layer | all counts computed from the actual models: MLP 235,146, CNN 108,618 |
| S1 p.16 | "Human ≈ 95 %" drawn on a **top-1** accuracy chart; the 5.1 % human figure is a **top-5 error** | ILSVRC top-5 error chart with the human 5.1 % line (03) |
| S1 p.16 | "every winning method has been a CNN (until ViT, 2021)": ILSVRC ended in 2017, ViT never competed | phrased as "from then on every winner was a CNN" |
| S1 p.11–12 | the example input switches from 8×8 to 6×6; the stride-2 slide labels its top row "stride 1" | one consistent 7×7 animation comparing stride 1, stride 2 and padding 1 (02) |
| S1 p.19 | `resnet50(pretrained=True)` is deprecated | `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` |

## Running locally

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

## Image credits

The maritime example photos are loaded at runtime from the [COCO 2017 validation set](https://cocodataset.org)
(Flickr images under Creative Commons licences); they are not redistributed in this repository.
The vessel dataset of notebooks 03 and 05 is loaded at runtime from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html)
(annotations CC BY 4.0, photos from Flickr under CC BY 2.0); title, author and Flickr page of every photo are listed in
[`assets/vessel_photo_credits.csv`](assets/vessel_photo_credits.csv).
Pretrained models: torchvision, and on the Hugging Face Hub `google/vit-base-patch16-224`, `facebook/vit-mae-base`,
`facebook/dino-vits8`, `facebook/dinov2-small`, `depth-anything/Depth-Anything-V2-Small-hf`.
