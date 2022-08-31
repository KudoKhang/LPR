from LPRPredict import *
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from io import BytesIO
from PIL import Image

def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    image = np.array(image)[:,:,::-1]
    return image

LPRPredictor = LicensePlateRecognition()

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""

app = FastAPI(title="Tensorflow FastAPI Start Pack", description=app_desc)

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    image = read_image_file(await file.read())
    prediction = LPRPredictor.predict(image)

    return prediction


if __name__ == "__main__":
    uvicorn.run(app, debug=True)