import requests

class Client:
    def __init__(self) -> None:
        _url = ""

    def send_prediction(self, text : str):
        data = {
            "text" : text
        }
        response = requests.post(url=f"{self._url}/predict", data=data)
        return response.json()
        
    def _test_conn(self):
        request_response = requests.head(self._url)
        status_code = request_response.status_code
        if status_code != 200:
            raise ConnectionError(f"{self._url} sent status code {status_code}")

    def connect(self, url : str):
        self._url = url
        self._test_conn()
        return self

if __name__ == "__main__":
    c = Client()
    c.connect("http://127.0.0.1:5000/")
    print(c.send_prediction("The sand is hotter under direct sunlight"))