import io
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Response, UploadFile
from LicensePlateRecognition import *
from PIL import Image
from starlette.responses import RedirectResponse, StreamingResponse


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    image = np.array(image)[:, :, ::-1]
    return image


LPRPredictor = LicensePlateRecognition()

app_desc = """<h2>Try this app by uploading any image with `predict`</h2>"""

app = FastAPI(title="Lisence Plate Recognition", description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_image_file(await file.read())
    prediction = LPRPredictor.predict(image)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
