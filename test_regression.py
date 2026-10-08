"""Regression test for the MNIST LeNet model.

Loads the pre-trained model parameters and evaluates accuracy on the
MNIST test set to ensure training stability has not regressed.
"""

import os
import pytest
from mxnet import cpu, gluon
from model import MnistModel

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "lenet_mnist.params")
TARGET_ACCURACY = 0.95
BATCH_SIZE = 256


@pytest.fixture(scope="module")
def trained_model():
    """Load the pre-trained LeNet model."""
    net = MnistModel()
    net.load_parameters(PARAMS_PATH, ctx=cpu())
    return net


@pytest.fixture(scope="module")
def test_loader():
    """Prepare the MNIST test-set DataLoader."""
    transforms = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.ToTensor(),
        gluon.data.vision.transforms.Normalize(0.13, 0.31),
    ])
    mnist_test = gluon.data.vision.datasets.MNIST(train=False)
    mnist_test = mnist_test.transform_first(transforms)
    return gluon.data.DataLoader(
        dataset=mnist_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )


def _evaluate_accuracy(model, loader):
    """Return mean accuracy over all batches in *loader*."""
    total_acc = 0.0
    n_batches = 0
    for data, label in loader:
        data = data.as_in_context(cpu())
        label = label.as_in_context(cpu())
        output = model.forward(data)
        total_acc += model.acc(output, label)
        n_batches += 1
    return total_acc / n_batches


def test_model_params_exist():
    """The saved parameter file must be present."""
    assert os.path.isfile(PARAMS_PATH), (
        f"Pre-trained parameters not found at {PARAMS_PATH}"
    )


def test_model_accuracy_above_target(trained_model, test_loader):
    """Model accuracy on the MNIST test set must exceed the target."""
    accuracy = _evaluate_accuracy(trained_model, test_loader)
    assert accuracy > TARGET_ACCURACY, (
        f"Model accuracy {accuracy:.4f} is below target {TARGET_ACCURACY}"
    )
