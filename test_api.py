import requests

url = "http://127.0.0.1:5000/predict"
files = {"file": open("test.wav", "rb")}

response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response Text:", response.text)
