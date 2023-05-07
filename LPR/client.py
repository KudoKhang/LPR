# importing the requests library
import requests

# myfiles = {'file': open('43C11130.jpg' ,'rb')}
# r = requests.post(URL, files=myfiles)
# print(r.text)


url = "http://127.0.0.1:8000/predict"
myfiles = {"file": open("88H0009.jpg", "rb")}
r = requests.post(url, files=myfiles)

# write raw bytes to file
with open("test.png", "wb") as f:
    f.write(r.content)

# or, convert back to PIL Image
# im = Image.open(io.BytesIO(r.content))
# im.save("test.png")
