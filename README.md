# Sequential Modelling for Video Classification

A deep learning and computer vision project exploring how recurrent neural networks learn temporal patterns from short image sequences. The project combines a convolutional neural network (CNN) with three recurrent architectures—Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and a plain Recurrent Neural Network (RNN)—to classify the direction of movement in generated video-like sequences.

> **Project status:** The experiment and report structure have been prepared. Before treating the reported results as final, verify every accuracy value against the executed Colab notebook/results CSV, replace illustrative training-time estimates with measured timings, and complete the required second failure-case analysis.

## Table of contents

- [1. Project overview](#1-project-overview)
- [2. Objectives](#2-objectives)
- [3. Dataset](#3-dataset)
- [4. Methodology](#4-methodology)
- [5. Model architecture](#5-model-architecture)
- [6. Experimental design](#6-experimental-design)
- [7. Evaluation](#7-evaluation)
- [8. Current reported results](#8-current-reported-results)
- [9. Repository structure](#9-repository-structure)
- [10. Tools and technologies](#10-tools-and-technologies)
- [11. Running the project](#11-running-the-project)
- [12. Reproducibility notes](#12-reproducibility-notes)
- [13. Limitations and known follow-up work](#13-limitations-and-known-follow-up-work)
- [14. Future improvements](#14-future-improvements)

## 1. Project overview

A normal image classifier examines one image at a time. It can identify what is present in a frame, but a single frame does not necessarily explain how an object is moving. Video and other sequential data require a model to consider both the visual content of individual frames and the order in which those frames occur.

This project addresses that problem with a CNN–RNN architecture:

1. A CNN extracts spatial features from each frame.
2. A recurrent network processes the ordered feature sequence and learns how those features change over time.
3. A classification head predicts the movement direction.

The task is a four-class classification problem: **Up, Down, Left, and Right**.

The work was designed as a controlled experiment for a sequential-modelling assignment. It compares LSTM, GRU, and plain RNN recurrent layers, and investigates how the number of sampled frames affects classification performance.

## 2. Objectives

The main objectives are to:

- Generate short sequences from handwritten-digit images to create a small video-classification dataset.
- Build a baseline CNN + LSTM model for movement-direction classification.
- Train the recurrent classifier while the CNN feature extractor is frozen, then unfreeze and fine-tune the full network.
- Replace the LSTM with a GRU and a plain RNN, then compare their performance.
- Evaluate the effect of using 4, 8, and 16 frames per sequence.
- Compare test accuracy, training time, and model parameter count.
- Visualise training and validation loss, evaluate predictions with a confusion matrix, and inspect misclassified examples.

## 3. Dataset

### Source data

The sequences are generated from the standard **MNIST handwritten-digit dataset**. MNIST contains grayscale images of handwritten digits, each originally 28 × 28 pixels.

### Generated sequence format

Each digit is placed on a larger **64 × 64 grayscale canvas**. The digit is translated across the canvas over a sequence of frames. The direction of translation supplies the target label.

| Property | Description |
|---|---|
| Source | MNIST handwritten digits |
| Original image size | 28 × 28 pixels |
| Generated frame size | 64 × 64 pixels |
| Sequence length at generation | 16 frames |
| Number of classes | 4 |
| Labels | Up, Down, Left, Right |
| Test-set size reported | 300 sequences (75 per class) |

This is a **custom Moving-MNIST-style dataset**, not the original canonical Moving MNIST benchmark. The motion sequences are generated programmatically from MNIST images for this experiment.

### Dataset splits

The generated examples are divided into training, validation, and test sets. The implementation should preserve separation between the source images used for each split so that frames derived from the same original digit do not leak across splits. The supplied report identifies a test set of 300 sequences, balanced across the four directions. Confirm the exact training and validation counts from the notebook before publishing them as fixed dataset statistics.

### Frame sampling

The generated sequence contains 16 frames. During experiments, the dataset loader samples evenly spaced frames to construct shorter input sequences:

- **4 frames:** a sparse temporal representation.
- **8 frames:** the intermediate setting and recurrent-architecture comparison baseline.
- **16 frames:** the full generated sequence.

The same sequence and label should be used when comparing frame counts, with only the number of sampled frames changing.

## 4. Methodology

The experiment follows a consistent pipeline:

1. Load the MNIST source images.
2. Place each digit on a 64 × 64 canvas.
3. Generate a sequence by translating the digit in one of four directions.
4. Assign the movement direction as the class label.
5. Create training, validation, and test splits.
6. Sample 4, 8, or 16 frames from each sequence, depending on the experiment.
7. Pass each frame through the CNN feature extractor.
8. Pass the ordered frame features through an LSTM, GRU, or plain RNN.
9. Use the final recurrent representation to predict one of the four movement classes.
10. Train, validate, and evaluate each configuration, recording accuracy, parameter count, and elapsed time.
11. Plot learning curves and comparison results, then inspect errors.

### Two-stage training

The training process uses two stages:

**Stage 1 — Frozen CNN:** The CNN feature extractor is frozen while the recurrent layer and classification head are trained. This allows the temporal model to learn from fixed visual features.

**Stage 2 — Fine-tuning:** The CNN is unfrozen and the complete network is trained further. This allows the visual feature extractor to adapt to the movement-classification task.

Record the actual epoch counts, learning rates, batch size, optimiser, loss function, and any early-stopping settings from the executed notebook. Those implementation details should be reported from the notebook rather than inferred from this README.

## 5. Model architecture

The network has three main components.

### 5.1 CNN feature extractor

The CNN processes each frame independently and converts its pixel values into a compact spatial feature representation. The supplied implementation uses convolutional layers with channel progression from 1 to 16 to 32, batch normalisation, ReLU activations, max pooling, and adaptive pooling to a 4 × 4 spatial representation.

The CNN is applied to every frame in the sequence. Its output is then arranged in temporal order and passed to the recurrent layer.

### 5.2 Recurrent sequence model

Three recurrent alternatives are evaluated:

- **LSTM:** Uses input, forget, and output gates together with a cell state to regulate information flow over time.
- **GRU:** Uses update and reset gates. It is a gated recurrent architecture with a simpler structure than an LSTM.
- **Plain RNN:** Uses a recurrent hidden state without the gating mechanisms of LSTM and GRU.

The supplied model configuration uses a recurrent hidden size of 96.

### 5.3 Classification head

The final temporal representation is passed through a classification head with a hidden layer of 64 units and an output layer of 4 units, corresponding to the four movement directions. The predicted class is selected from the model's output scores.

### Parameter-count reference

The following counts were calculated for the supplied CNN + recurrent architecture with hidden size 96 and four output classes. They are reference values and must be checked against the actual model instantiated in the notebook before being presented as measured implementation facts.

| Recurrent layer | Reference parameter count |
|---|---:|
| LSTM | 245,604 |
| GRU | 187,044 |
| Plain RNN | 69,924 |

Parameter counts are architecture-dependent. Changing the CNN, recurrent hidden size, number of recurrent layers, or classification head changes the total.

## 6. Experimental design

The experiments are organised to isolate two factors: recurrent architecture and sequence length.

### Experiment A — Recurrent architecture comparison

Use 8 frames for each model and change only the recurrent architecture:

| Configuration | Recurrent model | Frames |
|---|---|---:|
| A1 | LSTM | 8 |
| A2 | GRU | 8 |
| A3 | Plain RNN | 8 |

This comparison examines whether the additional gating mechanisms in LSTM and GRU provide a measurable advantage over a plain RNN for this relatively simple motion task.

### Experiment B — Number of input frames

Keep the LSTM architecture fixed and vary the number of sampled frames:

| Configuration | Recurrent model | Frames |
|---|---|---:|
| B1 | LSTM | 4 |
| B2 | LSTM | 8 |
| B3 | LSTM | 16 |

This comparison investigates whether a longer temporal sequence improves classification enough to justify the additional recurrent computation.

### Metrics to record

For every configuration, record:

- Test accuracy.
- Total training/experiment elapsed time, with a clear definition of what the timer includes.
- Total trainable and non-trainable parameter counts, or the total model parameter count if that is what the assignment requests.
- Training and validation loss by epoch where applicable.

For a fair comparison, use the same data splits and comparable training settings across configurations. If random initialisation or data order differs, consider repeating runs with multiple random seeds and reporting mean and standard deviation.

## 7. Evaluation

The project uses several complementary evaluation methods.

### Accuracy

Accuracy is the proportion of correctly classified test sequences:

`Accuracy = Correct predictions / Total test examples`

For a test set of 300 examples, one incorrect prediction corresponds to 299 / 300 = **99.67%** accuracy when rounded to two decimal places.

### Training and validation loss

The loss curves show how the model learns over training epochs and whether validation performance follows training performance. A widening gap between training and validation loss can indicate overfitting. The report includes a CNN + LSTM learning-curve figure.

### Confusion matrix

A confusion matrix compares true labels with predicted labels for each of the four directions. It helps identify which directions are most likely to be confused, rather than presenting only one overall accuracy value.

### Failure-case inspection

Misclassified sequences should be examined as ordered frame sequences. For each failure case, report:

- The true direction.
- The predicted direction.
- The sequence frames, preferably displayed in temporal order.
- A short explanation tied to visible evidence in the frames.

Do not attribute an error to a digit's shape, motion speed, or displacement unless the inspected sequence supports that explanation.

## 8. Current reported results

The supplied report currently transcribes the same test accuracy, **99.66%**, for all five configurations. This value is not yet verified against the experiment results CSV. In addition, the supplied baseline confusion matrix shows 299 correct predictions out of 300, which conventionally rounds to **99.67%**. Reconcile this discrepancy using the actual notebook outputs before publishing final results.

The training-time values in the supplied report were explicitly described as illustrative estimates, not measured timings. They must not be presented as actual benchmark results. Replace them with measurements from the Colab runs.

Use the following table as a results template and populate it from the executed experiments:

| Model | Frames | Test accuracy | Elapsed time (s) | Parameters |
|---|---:|---:|---:|---:|
| LSTM | 8 | Verify from results | Insert measured value | 245,604* |
| GRU | 8 | Verify from results | Insert measured value | 187,044* |
| Plain RNN | 8 | Verify from results | Insert measured value | 69,924* |
| LSTM | 4 | Verify from results | Insert measured value | 245,604* |
| LSTM | 16 | Verify from results | Insert measured value | 245,604* |

`*` Reference parameter counts for the described architecture; verify against the instantiated models.

### Figures

The accompanying report is intended to contain these four figures:

1. CNN + LSTM training and validation loss.
2. Performance across recurrent architectures.
3. Effect of frame sequence length on accuracy.
4. Confusion matrix for the CNN + LSTM baseline.

The Markdown report places the figures in a two-column layout. Keep the corresponding image assets in the relative path referenced by the report when exporting it to PDF.

## 9. Repository structure

A suggested GitHub repository layout is:

```text
sequential-modelling-video-classification/
│
├── README.md
├── notebooks/
│   └── sequential_modelling_experiment.ipynb
│
├── reports/
│   ├── Sequential_Modelling_Report.pdf
│   └── Sequential_Modelling_Report.md
│
├── results/
│   ├── experiment_results.csv
│   └── figures/
│       ├── training_validation_loss.png
│       ├── recurrent_architecture_comparison.png
│       ├── frame_count_comparison.png
│       └── confusion_matrix.png
│
├── src/                         # Optional, if code is exported from the notebook
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│
└── .gitignore
```

This is a recommended organisation, not a claim that every file already exists. Adjust the paths and filenames to match the files actually committed to the repository. If the project is implemented entirely in one Colab notebook, the `src/` directory can be omitted.

## 10. Tools and technologies

- **Python:** Main programming language.
- **Google Colab:** Notebook environment used for implementation and experiments.
- **PyTorch:** Model definition, training, and evaluation.
- **TensorFlow/Keras dataset utility:** Used in the workflow to obtain MNIST source data, if retained in the notebook.
- **NumPy:** Numerical operations and sequence generation.
- **Matplotlib:** Training curves, experiment comparisons, and visualisation of sequences.
- **scikit-learn:** Evaluation utilities such as confusion matrix and classification metrics.

The exact package versions should be recorded from the executed notebook if strict reproducibility is required.

## 11. Running the project

The project is designed to run in **Google Colab**. The simplest route is to open the committed notebook in Colab and execute its cells in order.

1. Upload or open `sequential_modelling_experiment.ipynb` in Google Colab.
2. Select a runtime. A GPU runtime may reduce training time, but record the runtime type because timings are hardware-dependent.
3. Run the installation/import and configuration cells.
4. Download/load MNIST and generate the moving-digit sequences.
5. Create the dataset splits and data loaders.
6. Define the CNN + recurrent models.
7. Train the baseline in the frozen-CNN stage and then fine-tune the full network.
8. Run the LSTM, GRU, and plain RNN comparisons using 8 frames.
9. Run the LSTM frame-count experiments using 4, 8, and 16 frames.
10. Evaluate the models, save results, and generate plots and the confusion matrix.
11. Inspect misclassified examples and complete the failure-case analysis.

If the notebook is later converted into a local Python project, create a virtual environment and install the packages imported by the code. The exact dependency list should be taken from the notebook; a `requirements.txt` should not claim package versions that were not tested.

## 12. Reproducibility notes

For a reproducible experiment, document and preserve:

- The MNIST source and the exact split strategy.
- Random seeds used for sequence generation, data splitting, model initialisation, and data-loader shuffling.
- The number of generated sequences in each split.
- The sequence-generation rules, including movement step size and starting-position constraints.
- The frame-sampling indices used for 4-, 8-, and 16-frame experiments.
- Model architecture and hidden size.
- Optimiser, learning rate, batch size, loss function, epoch count, and fine-tuning procedure.
- Colab runtime type and whether a GPU was enabled.
- The exact timing interval used for reported training time.
- The saved model checkpoints, experiment results, and plots.

Training time can vary with the hardware and runtime load. Comparisons are most meaningful when models are run under the same runtime conditions and the timer measures the same stages for every configuration.

## 13. Limitations and known follow-up work

The current task is deliberately controlled and relatively simple. It uses a single handwritten digit moving in a straight, axis-aligned direction on a blank canvas. It does not represent the complexity of natural videos, where objects may change direction, move at different speeds, become occluded, or interact with other objects.

The supplied report also has several items that must be resolved before final submission or publication:

1. **Verify accuracy values.** The five reported values are identical and need to be checked against the experiment results CSV.
2. **Resolve rounding.** The baseline confusion matrix indicates 299/300 correct, which is 99.67% to two decimal places, not 99.66%.
3. **Measure training time.** Replace illustrative timing estimates with actual Colab measurements.
4. **Verify parameter counts.** Recalculate counts from the actual model objects if the architecture differs from the assumed configuration.
5. **Complete two failure cases.** The supplied report documents one misclassification (true Up, predicted Left). Add a second genuine error, or clearly label and separately evaluate a stress-test example if no second test error exists.
6. **Keep explanations aligned with the implementation.** The model uses CNN image features and recurrent layers; explanations should not refer to bounding-box displacement features unless such features are actually implemented.
7. **Include the notebook.** Commit the executed `.ipynb` file so the implementation and outputs can be inspected.

## 14. Future improvements

Potential extensions include:

- Introduce diagonal movement, variable speed, acceleration, and direction changes.
- Generate sequences with multiple digits or partial occlusion.
- Evaluate on a more challenging public video dataset or a carefully selected subset of UCF101.
- Repeat experiments across multiple random seeds and report mean and standard deviation.
- Compare additional temporal models, such as temporal convolutional networks or attention-based sequence models.
- Inspect saliency or feature attribution methods to better understand which frames and regions influence predictions.
- Benchmark inference latency and memory use in addition to training time and parameter count.

## Academic integrity and reporting note

This README documents the project design and the status of the results available in the accompanying report. It intentionally distinguishes verified task details from values that still require confirmation. Do not present placeholder timings, unverified accuracies, or assumed parameter counts as measured results. Update this document after the final notebook run so that the README, report, plots, and committed results file all agree.
