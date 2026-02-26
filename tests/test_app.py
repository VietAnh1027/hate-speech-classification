import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
sys.path.insert(0, root_path)

from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_check_success():
    # Kiểm tra thành công
    response = client.post("/predict", json={"user": "a", "comment": "Tao là trò đùa của chúng mày à ?"})
    print(response)
    data = response.json()

    assert response.status_code == 200
    assert data['user'] == "a"
    assert data['comment'] == "Tao là trò đùa của chúng mày à ?"
    assert "type" in data
    assert "confident" in data
    assert isinstance(data['confident'], (float, int))

def test_check_without_field():
    # Kiểm tra thiếu một trường
    response1 = client.post("/predict", json={"comment": "Tao là trò đùa của chúng mày à ?"})
    response2 = client.post("/predict", json={"user": "a"})
    data1 = response1.json()
    data2 = response2.json()

    assert response1.status_code == 422
    assert response2.status_code == 422
    assert data1['detail'][0]['loc'] == ["body", "user"]
    assert data1['detail'][0]['type'] == "missing"
    assert data2['detail'][0]['loc'] == ["body", "comment"]
    assert data2['detail'][0]['type'] == "missing"

def test_check_empty_cmt():
    response = client.post("/predict", json={"user": "a", "comment": ""})
    assert response.status_code == 422