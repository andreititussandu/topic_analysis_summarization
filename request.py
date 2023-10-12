import json
import requests

data = {"url": "https://www.mayoclinic.org/drugs-supplements-creatine/art-20347591"}
headers = {"Content-Type": "application/json"}
response = requests.post("http://127.0.0.1:5000/analyze", data=json.dumps(data), headers=headers)

print(response.status_code)
print(response.json())