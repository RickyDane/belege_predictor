from yolov8_utils import *
import torch, gc
from model_config import *
import resource

gc.collect()
torch.cuda.empty_cache()

# soft, hard = resource.getrlimit(resource.RLIMIT_AS)
# resource.setrlimit(resource.RLIMIT_AS, (int(hard / 3), hard))

# Belege training
Detector().trainBelegeModel(resume = False, modelSize = "x", imageSize = 640, epochs = 1250)

# Zaehlprotokoll training
# hpmDetector.trainZaehlProtokollModel(resume = False)
