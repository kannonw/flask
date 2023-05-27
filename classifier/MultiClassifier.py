from torch import load
from torch.nn.functional import softmax
from classifier.model import UniversalClassifier
from classifier.utils import load_checkpoint, get_image_transformed
from PIL import Image
import numpy as np
from io import BytesIO

classes = ["Blood", "BrainMRI", "ChestCT", "ChestXRay", "KneeMRI", "KneeXRay", "Liver", "Ocular", "Unexpected"]

model = UniversalClassifier(num_classes=9).eval()
load_checkpoint(load("./classifier/proto_4.pth.tar", map_location ='cpu'), model)

def get_dictionary(pred):
    return dict(zip(classes, pred.astype(float).round(decimals=2)))

def PredictDisease(img_bytes):
    img_np = np.array(Image.open(BytesIO(img_bytes)).convert("RGB")) # Convert image bytes into a matrix
    img_tensor = get_image_transformed(img_np)

    prediction = softmax(model(img_tensor.unsqueeze(0)), dim=1).squeeze().detach().numpy() # Get prediction and classification using softmax

    pred_dic = get_dictionary(prediction)
    label_idx = np.argmax(prediction)

    return label_idx, pred_dic
