import pytest
import json
import base64
import numpy as np
from unittest.mock import patch
from PIL import Image
import io
from api.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def make_image_b64():
    img = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def test_segment_returns_200(client):
    mock_out = np.random.rand(1, 7, 256, 256).astype(np.float32)
    with patch("api.app.session") as mock_sess:
        mock_sess.run.return_value = [mock_out]
        resp = client.post('/segment',
            data=json.dumps({'image': make_image_b64()}),
            content_type='application/json')
    assert resp.status_code == 200


def test_segment_response_has_mask(client):
    mock_out = np.random.rand(1, 7, 256, 256).astype(np.float32)
    with patch("api.app.session") as mock_sess:
        mock_sess.run.return_value = [mock_out]
        resp = client.post('/segment',
            data=json.dumps({'image': make_image_b64()}),
            content_type='application/json')
    data = resp.get_json()
    assert 'mask'         in data
    assert 'distribution' in data


def test_segment_distribution_sums_to_100(client):
    mock_out = np.random.rand(1, 7, 256, 256).astype(np.float32)
    with patch("api.app.session") as mock_sess:
        mock_sess.run.return_value = [mock_out]
        resp = client.post('/segment',
            data=json.dumps({'image': make_image_b64()}),
            content_type='application/json')
    dist  = resp.get_json()['distribution']
    total = sum(dist.values())
    assert abs(total - 100.0) < 1.0


def test_segment_missing_image_returns_500(client):
    resp = client.post('/segment',
        data=json.dumps({}),
        content_type='application/json')
    assert resp.status_code == 500