from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uuid
from DetectBoundingBox import *
from v5 import lp_char_recog
app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join("uploads", filename)
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    result,cropped_region,bbox=get_bounding_box(upload_path)
    result_path = os.path.join("static/results", "result.jpg")
    cv2.imwrite(result_path, result)
    cropped_path = os.path.join("static/results", "cropped.jpg")
    cv2.imwrite(cropped_path, cropped_region)
    # characters=recognize_character(cropped_path)
    plate_image = cv2.imread(cropped_path)

    characters = lp_char_recog(plate_image)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "original": f"/uploads/{filename}",
        "result": f"/static/results/result.jpg",
        "characters": f"{characters}",
    })
