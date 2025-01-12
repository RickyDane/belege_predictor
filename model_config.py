# Model configurations
#  for predicting and training

# General configurations
modelPath = "models"
trainPath  = "runs/detect/train/weights"

othPredModel = "belege.pt"

belegeBestPredModel = "belege.pt"
belegeLastPredModel = "last.pt"

zaehlprotokollBestPredModel = "zaehlprotokoll.pt"
zaehlprotokollLastPredModel = "last.pt"

CLEAR_DEBUG_CROP = True

# Other prediction
OTH_MODEL_DEVICE = "cpu"
OTH_CONF = 0.75
OTH_MODEL = f"{modelPath}/{othPredModel}"

# Belege prediction
BEL_PRED_DEVICE = "cpu"
BEL_PRED_CONF = 0.75
BEL_PRED_MODEL = f"{modelPath}/{belegeBestPredModel}"
BEL_PRED_DEBUG_SHOW = False
BEL_PRED_SAVE = True
BEL_PRED_DEBUG_SAVE = False
BEL_PRED_OVERWRITE = True

# Belege training
BEL_TRAIN_DATA = "dataset/belege_dataset.yaml"
BEL_TRAIN_DEVICE = 0
BEL_TRAIN_RESUME_MODEL = f"{trainPath}/{belegeLastPredModel}"
BEL_TRAIN_MODEL = "models/yolov8x.pt"
BEL_TRAIN_BATCH = 2
BEL_TRAIN_BATCHSIZE = 16
BEL_TRAIN_PATIENCE = 50     # => 0 - 50
BEL_TRAIN_VAL = False
BEL_TRAIN_EPOCHS = 2500

# Zaehlprotokoll prediction
ZAEHL_PRED_DEVICE = "mps"
ZAEHL_PRED_CONF = 0.75
ZAEHL_PRED_MODEL = f"{modelPath}/{zaehlprotokollBestPredModel}"
ZAEHL_PRED_DEBUG_SHOW = False
ZAEHL_PRED_SAVE = True
ZAEHL_PRED_DEBUG_SAVE = True
ZAEHL_PRED_OVERWRITE = True

# Zaehlprotokoll training
ZAEHL_TRAIN_DATA = "dataset/zaehlprotokoll_dataset.yaml"
ZAEHL_TRAIN_DEVICE = 0
ZAEHL_TRAIN_RESUME_MODEL = f"{trainPath}/{zaehlprotokollLastPredModel}"
ZAEHL_TRAIN_MODEL = "models/yolov8m.pt"
ZAEHL_TRAIN_BATCH = 4
ZAEHL_TRAIN_BATCHSIZE = 32
ZAEHL_TRAIN_PATIENCE = 50     # => 0 - 50
ZAEHL_TRAIN_VAL = False
ZAEHL_TRAIN_EPOCHS = 1000
