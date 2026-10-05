
"""Train and evaluate a CNN on Fashion-MNIST."""

import time
import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_gpu,
    use_gpu,
)
from dptiny.backend import xp
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import get_fashion_mnist, CLASSES
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

# Random seed


SEED = 42

np.random.seed(SEED)
xp.random.seed(SEED)


# GPU / CPU

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")


# Load Fashion-MNIST

print("Loading Fashion-MNIST...")

X_train, X_test, y_train, y_test = get_fashion_mnist(
    normalize=True,
    flatten=False,
)

# Standardize using TRAINING set statistics only

train_mean = float(X_train.mean())
train_std = float(X_train.std())

print(f"Training mean: {train_mean:.6f}")
print(f"Training std:  {train_std:.6f}")

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std


# CNN baseline from examples/mnist_cnn.py

model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)

if is_gpu():
    model.to_gpu()

# DataLoaders

batch_size = 64

data_loader = DataLoader(
    (X_train, y_train),
    batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    (X_test, y_test),
    batch_size,
    shuffle=False,
)


# Optimizer

optimizer = Adam(
    model,
    lr=0.001,
)

# Training

max_epoch = 10

start_time = time.time()

for epoch in range(max_epoch):

    model.train()

    sum_loss = 0.0
    correct = 0
    total = 0

    for x, t in data_loader:

        x = Variable(x)

        y = model(x)

        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        batch_n = len(t)

        sum_loss += float(loss.data) * batch_n

        pred = y.data.argmax(axis=1)

        correct += int((pred == t).sum())
        total += batch_n

    train_loss = sum_loss / total
    train_accuracy = correct / total

    # Test accuracy

    test_correct = 0
    test_total = 0

    with test_mode(), no_grad():

        for x, t in test_loader:

            y = model(Variable(x))

            pred = y.data.argmax(axis=1)

            test_correct += int((pred == t).sum())
            test_total += len(t)

    test_accuracy = test_correct / test_total

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch + 1:2d}/{max_epoch} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Test Acc: {test_accuracy:.4f} | "
        f"Time: {elapsed:.1f}s"
    )


print("\nTraining completed.")
print(
    f"Total training time: "
    f"{time.time() - start_time:.2f} seconds"
)


# Final evaluation

all_predictions = []
all_labels = []

with test_mode(), no_grad():

    for x, t in test_loader:

        y = model(Variable(x))

        predictions = y.data.argmax(axis=1)

        all_predictions.extend(
            np.asarray(predictions).reshape(-1)
        )

        all_labels.extend(
            np.asarray(t).reshape(-1)
        )

all_predictions = np.asarray(
    all_predictions,
    dtype=np.int32,
)

all_labels = np.asarray(
    all_labels,
    dtype=np.int32,
)

# Confusion matrix

num_classes = 10

cm = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64,
)

for true_label, predicted_label in zip(
    all_labels,
    all_predictions,
):
    cm[true_label, predicted_label] += 1


print("\n10 x 10 CONFUSION MATRIX")
print("=" * 70)

print("Rows = True Class")
print("Columns = Predicted Class\n")

print("             ", end="")

for name in CLASSES:
    print(f"{name[:12]:>13}", end="")

print()

for i, name in enumerate(CLASSES):

    print(f"{name[:12]:>12} ", end="")

    for j in range(num_classes):
        print(f"{cm[i, j]:>13}", end="")

    print()

# Overall accuracy

overall_accuracy = np.trace(cm) / np.sum(cm)

print("\n" + "=" * 70)

print(
    f"Overall test accuracy: "
    f"{overall_accuracy:.4f}"
)

print(
    f"Overall test accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

# Per-class accuracy

print("\nPER-CLASS ACCURACY")
print("=" * 50)

for i, class_name in enumerate(CLASSES):

    total_class = cm[i].sum()
    correct_class = cm[i, i]

    accuracy = correct_class / total_class

    print(
        f"{i}: {class_name:<15} "
        f"{correct_class:>5}/{total_class:<5} "
        f"{accuracy * 100:>6.2f}%"
    )

# Most confused CLASS PAIR
#
# Combine both directions:
#
# A -> B + B -> A

pair_confusion = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64,
)

for i in range(num_classes):

    for j in range(i + 1, num_classes):

        pair_confusion[i, j] = (
            cm[i, j] + cm[j, i]
        )


i, j = np.unravel_index(
    np.argmax(pair_confusion),
    pair_confusion.shape,
)

most_confused_total = pair_confusion[i, j]

print("\nMOST CONFUSED CLASS PAIR")
print("=" * 50)

print(
    f"{CLASSES[i]} <-> {CLASSES[j]}"
)

print(
    f"{CLASSES[i]} -> {CLASSES[j]}: "
    f"{cm[i, j]} errors"
)

print(
    f"{CLASSES[j]} -> {CLASSES[i]}: "
    f"{cm[j, i]} errors"
)

print(
    f"Total confusion: "
    f"{most_confused_total} errors"
)
