import torch
from torch.nn.functional import softmax
from classifier.model import UniversalClassifier
from classifier.utils import load_checkpoint, get_image_transformed
from PIL import Image
import numpy as np
import io

classes = ["Blood", "BrainMRI", "ChestCT", "ChestXRay", "KneeMRI", "KneeXRay", "Ocular", "Unexpected"]

model = UniversalClassifier().eval()
load_checkpoint(torch.load("./classifier/proto_3.pth.tar", map_location ='cpu'), model)

def get_dictionary(pred):
    print(pred.round(decimals=4).tolist())
    return dict(zip(classes, pred.round(decimals=4)))

def main(img_bytes):
    img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB")) # Convert image bytes into a matrix
    img_tensor = get_image_transformed(img_np)

    prediction = softmax(model(img_tensor.unsqueeze(0)), dim=1).squeeze().detach().numpy() # Get prediction and classification using softmax

    pred_dic = get_dictionary(prediction)

    return pred_dic
