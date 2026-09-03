# Pet Breed Classifier — Transfer Learning with ResNet18

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/pet-classifier/blob/main/notebooks/train.ipynb)

Fine-grained classification of the 37 cat and dog breeds in the Oxford-IIIT Pet
dataset, using an ImageNet-pretrained ResNet18 as a frozen feature extractor
with a linear classification head trained on top. Runs end to end on a free
Colab T4 GPU in under ten minutes.

![Sample predictions on held-out test images](outputs/predictions.png)

## Results

| | |
|---|---|
| **Test accuracy (frozen backbone)** | **XX.X%** |
| Test accuracy (after fine-tuning `layer4`) | XX.X% |
| Trainable parameters | 18,981 of 11,195,493 (0.17%) |
| Random-guess baseline | 2.7% |
| Training time | ~X minutes on a Colab T4 |

## Approach

The classifier is a standard transfer-learning setup:

- **Backbone** — ResNet18 with ImageNet-pretrained weights, entirely frozen.
  Every parameter has `requires_grad = False`, and the optimiser is given only
  the head's parameters, so the pretrained features cannot be modified.
- **Head** — the original 1000-class ImageNet layer is replaced with a single
  `Linear(512, 37)` trained from scratch.
- **Data** — Oxford-IIIT Pet ships only `trainval` and `test` splits, so the
  validation set is carved out of `trainval` with a seeded 85/15 index
  permutation. The test split is untouched until final evaluation.
- **Augmentation** — random resized crop and horizontal flip on the training
  split only; validation and test use a deterministic resize and centre crop so
  reported accuracy reflects the model rather than the luck of a random crop.
- **Normalisation** — ImageNet channel statistics, as required by the
  pretrained backbone.
- **Training** — Adam at `lr=1e-3`, cross-entropy loss, 10 epochs.

## Representation analysis

Beyond the headline accuracy, the notebook asks how much of that accuracy came
from the pretrained representation itself rather than from the trained head.

A **nearest-centroid classifier** provides the comparison: each breed is
represented by the mean embedding of its training images, and test images are
assigned to the closest centroid. It has no learned parameters at all.

| Method | Test accuracy |
|---|---|
| Nearest centroid, cosine | XX.X% |
| Nearest centroid, Euclidean | XX.X% |
| Trained linear head | XX.X% |

The gap between the trained head and the best centroid method measures how much
the learned classifier adds over the geometry that the frozen backbone already
provides. <!-- One or two sentences on what your gap actually shows. A small gap
means the pretrained features already cluster the breeds; a large gap means the
head is doing substantial work. -->

Cross-referencing the confusion matrix against centroid geometry gives a
correlation of **X.XX** between how often two breeds are confused and how close
their class centroids sit in embedding space — <!-- what this tells you: are the
model's mistakes geometrically explicable? -->

The most-confused breed pairs are <!-- e.g. American Bulldog / American Pit Bull
Terrier -->, which are also among the closest in the embedding space.

## Repository structure

```
pet-classifier/
├── README.md
├── requirements.txt
├── notebooks/
│   └── train.ipynb          # data, model, training, evaluation, analysis
└── outputs/
    └── predictions.png      # sample test predictions
```

## Running it

The notebook is written for Google Colab and needs no local setup:

1. Open `notebooks/train.ipynb` in Colab (badge above).
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`. The dataset (~800 MB) downloads on first run.

To run locally instead, `pip install -r requirements.txt` and run the notebook
in Jupyter. A CUDA-capable GPU is optional — the notebook falls back to CPU,
but training will be considerably slower.

## Notes and possible extensions

- Only `layer4` is unfrozen during fine-tuning, at a learning rate an order of
  magnitude below the head's. Earlier layers encode generic edge and texture
  detectors that need no adaptation, and are the most fragile to disturb.
- Model selection is done on the validation split; the test split is read once,
  at the end.
- Further directions: SVD of the embedding matrix to measure its effective rank,
  a random-projection sweep to see how far the 512-d features can be compressed
  before accuracy degrades, and comparing backbones of different depths.

## References

- Parkhi, Vedaldi, Zisserman and Jawahar. *Cats and Dogs*. CVPR 2012. —
  [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- He, Zhang, Ren and Sun. *Deep Residual Learning for Image Recognition*. CVPR
  2016. — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Snell, Swersky and Zemel. *Prototypical Networks for Few-shot Learning*.
  NeurIPS 2017. — [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)
