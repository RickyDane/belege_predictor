from easyocr.easyocr import lang
from ultralytics import YOLO
import time, shutil, os, cv2
from enum import Enum
import time
# from paddleocr import PaddleOCR
import pytesseract as tess
from model_config import *
import pandas as pd
import asyncio
import easyocr

class DataMode(Enum):
    none = 0
    belege = 1
    dokumente = 2


class Detector(object):

    def __init__(self):
        try:
            print("Loading model ...")
            self.predictionModel = YOLO(OTH_MODEL)
            print("Loading belege model ...")
            self.predictionBelegeModel = YOLO(BEL_PRED_MODEL)
            # print("Loading zaehlprotokoll model ...")
            # self.predictionZaehlprotokollModel = YOLO(ZAEHL_PRED_MODEL)
            print("Loading ocr model ...")
            self.ocrModel = easyocr.Reader(["en", "de"])
            print("Loading ocr engine ...")
            # self.reader = PaddleOCR(use_angle_cls = True, lang = "german", show_log = False)
            print("Loading done")
        except Exception as ex:
            print(f"# Warnung # : Predictmodel ist noch nicht verfügbar: {ex}")


    def trainBelegeModel(self, modelSize = "s", cache = False, imageSize = 640, resume = False, epochs = BEL_TRAIN_EPOCHS):

        trainingModel = YOLO(f"models/yolov8{modelSize}.pt")

        # if not resume:
        #     resume = False
        #     trainingModel = YOLO(BEL_TRAIN_MODEL)

        #     try:
        #         if shutil.rmtree("runs/detect/train"):
        #             print(" # Previous training data was removed # ")
        #             time.sleep(2)
        #     except Exception as ex:
        #         print(f"No training data yet: {ex}")
        #     time.sleep(2)
        # else:
        #     trainingModel = YOLO(BEL_TRAIN_RESUME_MODEL)
        #     resume = True

        try:
            trainingModel.train(workers = 8, data = BEL_TRAIN_DATA, single_cls = False, nbs = BEL_TRAIN_BATCHSIZE, resume = resume, amp = False, batch = BEL_TRAIN_BATCH, cache = cache, epochs = epochs, imgsz = imageSize, patience = BEL_TRAIN_PATIENCE, exist_ok = True, val = BEL_TRAIN_VAL, verbose = False, device = BEL_TRAIN_DEVICE)
        except Exception as ex:
            print(f"\nProbably cannot resume training | Message:\t{ex}\n")

        try:
            os.remove("models/hpmdetection_belege.pt")
        except Exception as ex:
            print(f"Kein vortrainiertes model vefügbar: {ex}")

        try:
            shutil.copy("runs/detect/train/weights/best.pt", "models/hpmdetection_belege.pt")
        except Exception as ex:
            print(ex)


    def trainZaehlProtokollModel(self, cache = False, imageSize = 640, resume = False):

        if not resume:
            resume = False
            trainingModel = YOLO(ZAEHL_TRAIN_MODEL)

            try:
                if shutil.rmtree("runs/detect/train"):
                    print(" # Previous training data was removed # ")
                    time.sleep(2)
            except Exception as ex:
                print(f"No training data yet: {ex}")
            time.sleep(2)
        else:
            trainingModel = YOLO(ZAEHL_TRAIN_RESUME_MODEL)
            resume = True

        try:
            trainingModel.train(data = ZAEHL_TRAIN_DATA, single_cls = False, nbs = ZAEHL_TRAIN_BATCHSIZE, resume = resume, amp = False, batch = ZAEHL_TRAIN_BATCH, cache = cache, epochs = ZAEHL_TRAIN_EPOCHS, imgsz = imageSize, patience = ZAEHL_TRAIN_PATIENCE, exist_ok = True, val = ZAEHL_TRAIN_VAL, verbose = False, device = ZAEHL_TRAIN_DEVICE)
        except Exception as ex:
            print(f"\nProbably cannot resume training | Message:\t{ex}\n")

        try:
            os.remove("models/hpmdetection_zaehlprotokoll.pt")
        except Exception as ex:
            print(ex)

        try:
            shutil.copy("runs/detect/train/weights/best.pt", "models/hpmdetection_zaehlprotokoll.pt")
        except Exception as ex:
            print(ex)


    def getClassLabel(self, cls) -> str:
        if cls == 0:
            return "Summe"
        elif cls == 1:
            return "Titel"
        elif cls == 2:
            return "Label/Logo"
        else:
            return "Nicht erkannt"


    def predictBelege(self, image, imageName):
        dataFrameClasses = []
        dataFrameValues = []
        model = self.predictionBelegeModel

        result = model.predict(image, workers = 8, device = BEL_PRED_DEVICE, save = True, show = BEL_PRED_DEBUG_SHOW, conf = BEL_PRED_CONF, exist_ok = BEL_PRED_OVERWRITE)

        i = 0

        returnResults = []

        step = ""

        try:
            image = result[0].orig_img

            if BEL_PRED_SAVE == True:
                step = "saving debug image"
                cv2.imwrite(f"hpm_traindata/belege data/test_output_images/{imageName}", image)

            for box in result[0].boxes:

                step = "boxes"
                boxX = box.xyxy[0][0]
                boxX2 = box.xyxy[0][2]
                boxY = box.xyxy[0][1]
                boxY2 = box.xyxy[0][3]
                boxCls = int(box.cls[0])

                step = "cropping"

                cropImage = image[int(boxY):int(boxY2), int(boxX):int(boxX2)]

                step = "image processing"

                cropImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
                cropImage = cv2.resize(cropImage, None, fx = 2.0, fy = 2.0, interpolation = cv2.INTER_LANCZOS4)
                # cropImage = cv2.GaussianBlur(cropImage, (3, 3), 0)
                threshold, cropImage = cv2.threshold(cropImage, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cropImage = cv2.adaptiveThreshold(cropImage, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 87, 11)

                if BEL_PRED_DEBUG_SAVE:
                    step = "saving"
                    cv2.imwrite(f"hpm_traindata/belege data/crop_result_output/{self.getClassLabel(boxCls)}_{i}_{imageName}", cropImage)
                    i += 1

                step = "reading"

                #region PaddleOCR
                # ocrResult = self.reader.ocr(cropImage, cls = False)
                # textResult = [line[1][0] for line in ocrResult[0]]
                # print(textResult)

                # if boxCls == 0:
                #     if len(textResult) > 1:
                #         textResult = str(textResult[0]).replace(",", ".").replace("-", "") + "." + str(textResult[1]).replace(",", ".").replace("-", "")
                #     elif len(textResult) < 1:
                #         textResult = "-"
                #     else:
                #         textResult = str(textResult[0]).replace(",", ".").replace("-", "")
                # elif boxCls == 1:
                #     if len(textResult) > 1:
                #         tempText = ""
                #         for text in textResult:
                #             tempText += text + " "
                #         textResult = tempText
                #     elif len(textResult) < 1:
                #         textResult = "-"
                #     else:
                #         textResult = str(textResult[0]).replace("-", "")
                # else:
                #     textResult = "-"

                # print(textResult)
                #endregion

                #region ocr
                # ocrResult = tess.image_to_string(cropImage, lang="eng", config="--psm 13 --oem 1 -c tessedit_char_blacklist=$§%!()/&?#+*;:_-/").replace(",", ".").replace("\n", "")
                ocrResult = self.ocrModel.readtext(cropImage, detail=0, paragraph=True, decoder="greedy", batch_size=4)
                print(ocrResult)
                ocrResult = ocrResult[0] # easyocr specific
                textResult = ocrResult
                # print(textResult)
                #endregion

                step = "appending"
                returnResults.append([boxCls, self.getClassLabel(boxCls), textResult])
                # dataFrameClasses.append(self.getClassLabel(boxCls))
                # dataFrameValues.append(textResult)

                # dataFrame = {
                #     "Image": imageName,
                #     "Classes": dataFrameClasses,
                #     "Values": dataFrameValues
                # }
                returnResults
                # print(pd.DataFrame(dataFrame))

        except Exception as ex:
            print(f"Something went wrong getting the bboxes on step {step}: {ex}")
            return list()

        return returnResults


    def predictZaehlProtokoll(self, image, imageName, dataMode, processReturn = None):
        model = self.predictionZaehlprotokollModel

        result = model.predict(image, device = BEL_PRED_DEVICE, save = BEL_PRED_SAVE, show = BEL_PRED_DEBUG_SHOW, conf = BEL_PRED_CONF, exist_ok = BEL_PRED_OVERWRITE)

        i = 0

        returnResults = []

        step = ""

        try:
            image = result[0].orig_img

            if ZAEHL_PRED_SAVE == True:
                step = "saving debug image"
                cv2.imwrite(f"hpm_traindata/zaehlprotokoll data/test_output_images/{imageName}", image)

            for box in result[0].boxes:

                step = "boxes"
                boxX = box.xyxy[0][0]
                boxX2 = box.xyxy[0][2]
                boxY = box.xyxy[0][1]
                boxY2 = box.xyxy[0][3]
                boxCls = int(box.cls[0])

                step = "cropping"

                cropImage = image[int(boxY):int(boxY2), int(boxX):int(boxX2)]

                step = "image processing"

                cropImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
                cropImage = cv2.resize(cropImage, None, fx = 2.0, fy = 2.0, interpolation = cv2.INTER_LANCZOS4)
                # cropImage = cv2.GaussianBlur(cropImage, (3, 3), 0)
                # threshold, cropImage = cv2.threshold(cropImage, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cropImage = cv2.adaptiveThreshold(cropImage, 245, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 87, 11)

                if ZAEHL_PRED_DEBUG_SAVE:
                    step = "saving"
                    cv2.imwrite(f"hpm_traindata/zaehlprotokoll data/crop_result_output/{self.getClassLabel(boxCls)}_{i}_{imageName}", cropImage)
                    i += 1

                step = "reading"

                ocrResult = self.reader.ocr(cropImage, cls = False)
                textResult = [line[1][0] for line in ocrResult[0]]

                print(textResult)

                step = "appending"
                returnResults.append([boxCls, self.getClassLabel(boxCls), textResult])

        except Exception as ex:
            print(f"Something went wrong getting the bboxes on step {step}: {ex}")
            return float(0)

        return returnResults
