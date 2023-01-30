import requests

url = 'http://localhost:9696/predict'

# url = 'http://localhost:8080/predict'

# data = {'url': 'https://cloudcape.saao.ac.za/index.php/s/iFNh4f35Oy0nT6b/download'}

data = {'url': 'https://cloudcape.saao.ac.za/index.php/s/mUCoY6MeF49csoj/download'}

result = requests.post(url, json=data).json()
print(result)