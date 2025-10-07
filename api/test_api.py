import requests

# Send to API
url = "http://localhost:8000/predict"
files = {'file': open('data/test_image.png', 'rb')}
response = requests.post(url, files=files)

print("API Response:")
print(response.json())
