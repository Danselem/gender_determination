import requests

url = 'http://localhost:9696/predict'

# url = 'http://localhost:8080/predict'

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'}

result = requests.post(url, json=data).json()
print(result)