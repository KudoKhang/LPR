# importing the requests library
import requests

URL = 'https://5ecb-113-160-224-57.ngrok.io/predict/'
# URL = "http://127.0.0.1:8000/predict/"
myfiles = {'file': open('43C11130.jpg' ,'rb')}
r = requests.post(URL, files=myfiles)
print(r.text)