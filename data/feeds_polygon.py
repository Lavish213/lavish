import os
import requests
from dotenv import load_dotenv

# === Load Environment Variables ===
load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

BASE_URL = "https://api.polygon.io"


# === Fetch Polygon Data ===
def fetch_polygon_data(endpoint: str, params: dict = None):
    """
    Fetch data from the Polygon.io API.

    Args:
        endpoint (str): The API endpoint (e.g., '/v2/aggs/ticker/AAPL/prev')
        params (dict): Optional query parameters

    Returns:
        dict or None: JSON response data or None if request fails
    """
    if params is None:
        params = {}

    url = f"{BASE_URL}{endpoint}"
    params["apiKey"] = POLYGON_API_KEY

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching Polygon data: {e}")
        return None


# === Example Usage ===
if __name__ == "__main__":
    print("🔍 Testing Polygon API fetch...")

    endpoint = "/v2/aggs/ticker/AAPL/prev"  # Change 'AAPL' to any ticker
    test_data = fetch_polygon_data(endpoint)

    if test_data:
        print("✅ Successfully fetched Polygon data!")
        print("Sample keys:", list(test_data.keys())[:5])
        print("Ticker:", test_data.get("ticker", "N/A"))
        print("Results count:", len(test_data.get("results", [])))
        print("Full Response Preview:", test_data)
    else:
        print("❌ Failed to fetch Polygon data.")
