import sys
import uvicorn, asyncio, fastapi
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi import File, UploadFile
from loguru import logger
from PIL import Image
from io import BytesIO
import numpy as np
import cv2, time, os, yaml
import json
from yolov8.YOLOv8 import *

model_path = "models/last.onnx"
my_yolo = YOLOv8(model_path)
app = fastapi.FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({
                                "returncode": {
                                    "code": 422,
                                    "message": "Request json has syntax error"
                                },
                                "output": {"result": [], "timing": 0.0}
                        }),
    )

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))


@app.post("/image")
async def get_body(file: UploadFile = File(...)):
    bbyte = file.file.read()
    image = load_image_into_numpy_array(bbyte)
    st =  time.time()
    lst_box, lst_score, class_ids = my_yolo(image)
    et = time.time()
    print("Lisence Plate Detection:", et-st)
    output = {"type":"ObjectDetectionPrediction", "predictions": {} }
    for index in range(len(lst_box)):
        res_name = "res_" + (str)(index + 1)
        _res = {}
        _res["score"] = float(lst_score[index])
        _res["labelName"] = 'license plate'
        _res["coordinates"] = {}
        _res["coordinates"]["xmin"] = int(lst_box[index][0])
        _res["coordinates"]["ymin"] = int(lst_box[index][1])
        _res["coordinates"]["xmax"] = int(lst_box[index][2])
        _res["coordinates"]["ymax"] = int(lst_box[index][3])
        output["predictions"][res_name] = _res
        # Serializing json  
        # json_object = json.dumps(output)
        # output = 1
    return output


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=5001)
    uvicorn.run(app, host='localhost', port=6000)