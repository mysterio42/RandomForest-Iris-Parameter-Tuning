import requests

url = 'http://0.0.0.0:5000/classify'

payload = {
    'sepalLength': 5.4,
    'sepalWidth': 3.7,
    'petalLength': 1.5,
    'petalWidth': 0.2,
}

if __name__ == '__main__':
    r = requests.post(url=url, json=payload)
    print(r.text)
