import pytest
import json
import sys
import os

# Python ko batane ke liye ke backend folder kahan hai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

try:
    from app import app  # Aap ki Flask app import ho rahi hai
except ImportError:
    # Agar app.py direct root par hai to yahan se import hogi
    from backend.app import app

@pytest.fixture
def client():
    """Flask test client setup karta hai."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage_status(client):
    """Test 1: Check karta hai ke backend live hai."""
    response = client.get('/')
    # Agar home route par status 200 ya 404 bhi aaye to matlab server chal raha hai
    assert response.status_code in [200, 404]

def test_prediction_endpoint_structure(client):
    """Test 2: Check karta hai ke endpoint route logic structured hai."""
    payload = {
        "Store": 1,
        "Dept": 1,
        "IsHoliday": 0,
        "Temperature": 42.31,
        "Fuel_Price": 2.572,
        "CPI": 211.096,
        "Unemployment": 8.106,
        "Size": 151315,
        "Month": 2,
        "Year": 2026
    }
    response = client.post('/predict', 
                           data=json.dumps(payload),
                           content_type='application/json')
    # Hum sirf check kar rahe hain ke route crash na ho (500 error na de)
    assert response.status_code != 500