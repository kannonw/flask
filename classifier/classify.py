import torch
from torch.nn.functional import softmax
from classifier.model import UniversalClassifier
from classifier.utils import load_checkpoint, get_image_transformed
from PIL import Image
import numpy as np
import io

classes = ["Blood", "BrainMRI", "ChestCT", "ChestXRay", "KneeMRI", "KneeXRay", "Ocular"]

model = UniversalClassifier().eval()
load_checkpoint(torch.load("./classifier/my_checkpoint.pth.tar", map_location ='cpu'), model)

def get_dictionary(pred):
    return dict(zip(classes, np.trunc(pred[0]).tolist()))

def main(img_bytes):
    img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB")) # Convert image bytes into a matrix
    img_tensor = get_image_transformed(img_np)

    prediction = softmax(model(img_tensor.unsqueeze(0))).detach().numpy() # Get prediction and classification using softmax

    pred_dic = get_dictionary(prediction)

    return pred_dic
