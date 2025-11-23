import re
import requests

def get_campaign_id(url):
    try:
        print(f"🔍 Fetching page source from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text

        # Find campaign ID pattern
        match = re.search(r'"campaign_id"\s*:\s*"(\d+)"', html)
        if match:
            campaign_id = match.group(1)
            print(f"✅ Found Campaign ID: {campaign_id}")
            return campaign_id
        else:
            print("❌ Campaign ID not found in page source.")
    except Exception as e:
        print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    # Replace this with your creator page URL
    your_url = "https://www.patreon.com/angelowashington"
    get_campaign_id(your_url)