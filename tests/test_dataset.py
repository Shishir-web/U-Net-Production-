import numpy as np
import pytest
from src.dataset import rgb_mask_to_label, CLASS_RGB


def test_rgb_mask_to_label_shape():
    mock_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    label = rgb_mask_to_label(mock_mask)
    assert label.shape == (256, 256)


def test_rgb_mask_to_label_urban():
    mock_mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_mask[:5] = [0, 255, 255]   # urban → class 0
    mock_mask[5:] = [0, 255, 0]     # forest → class 3
    label = rgb_mask_to_label(mock_mask)
    assert label[0, 0] == 0
    assert label[9, 0] == 3


def test_all_classes_mapped():
    assert len(CLASS_RGB) == 7


def test_label_dtype():
    mock_mask = np.zeros((64, 64, 3), dtype=np.uint8)
    label = rgb_mask_to_label(mock_mask)
    assert label.dtype == np.int64