# importing the requests library
import requests

URL = 'https://cc9d-45-122-238-69.ap.ngrok.io/predict/'
# URL = "http://127.0.0.1:8000/predict/"
myfiles = {'file': open('43C11130.jpg' ,'rb')}
r = requests.post(URL, files=myfiles)
print(r.text)