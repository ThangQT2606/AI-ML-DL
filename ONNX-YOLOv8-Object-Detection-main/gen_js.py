from yolov8.YOLOv8 import *
import json, os, cv2
# YOLO
model_path = "models/last.onnx"
my_yolo = YOLOv8(model_path)
img_path = os.path.join(os.path.dirname(__file__), 'images', 'Cars259.png')
img = cv2.imread(img_path, 1)
image = img.copy()
lst_box, lst_score, class_ids = my_yolo(image)
print('class: ',class_ids)
# print(lst_box)
# lst_score = [round(float(lst_score), 3)]
# print(lst_score)
# lst_box = lst_box.round().astype(np.int32).tolist()
# print(lst_box)

combined_img = my_yolo.draw_detections(image)
# del my_yolo.session
# cv2.imshow("result", combined_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()   


output = {
        "type":"ObjectDetectionPrediction", 
        "predictions": {

        } 
    }


if (len(lst_box) >0):
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

else:
    print ('No object detected')

print (output)
json_object = json.dumps(output)
print (json_object)

with open("res.json", "w") as outfile:
    json.dump(output, outfile)