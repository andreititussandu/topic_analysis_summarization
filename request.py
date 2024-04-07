import requests
import json

def predict_topic(url):
    endpoint_url = "http://localhost:8000/predict"
    data = {"url": url}

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(endpoint_url, json=data, headers=headers)

    print(f"Raw Response Status Code: {response.status_code}")
    print(f"Raw Response Text: {response.text}")

    if response.status_code == 200:
        try:
            result = response.json()
            predicted_topic_label = result.get("predicted_topic_label")
            top_topic_strength = result.get("top_topic_strength")
            print(f"Predicted Topic Label: {predicted_topic_label}")
            print(f"Top Topic Strength: {top_topic_strength}")
        except json.JSONDecodeError:
            print("Error decoding JSON response")
    else:
        try:
            error_message = response.json().get("error")
            print(f"Error: {error_message}")
        except json.JSONDecodeError:
            print("Error decoding JSON error response")

if __name__ == "__main__":
    url_to_predict = "https://www.zdnet.com/article/which-iphone-15-model-should-you-buy-comparing-regular-plus-pro-and-pro-max/"
    predict_topic(url_to_predict)

