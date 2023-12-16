"""
1. Jyothi Vishnu Vardhan Kolla
2. Karan Shah

CS-7180 Fall 2023 semester.

This is the main function which is the heart of system.
"""

# This file loads the model and makes predictions.
from PIL import Image
import sys
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as T
from utils import detect, plot_results
from model import DETR
import torch
from evaluate import visualize_images
from utils import mean_average_precision

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# The main function which takes command line arguements and exegutes the system.
def main(argv):
    url = 'https://storage.googleapis.com/kagglesdsdata/datasets/857191/1462296/coco2017/val2017/000000000724.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20231211%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231211T155929Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=9b76c73f732b0b9288ed589d923b00ce06f7476f84cbe6766a59060a455dfe89af7a4a9911cb7e27c219e596e388373de70fd6170260ee5436b593729e0482e481354e869ae6a6b01a5f9a3ecfc2b0ea4979e1f69480a7251442d6f3f73835738424849154abc9ddbc2aaa3db3a67dd8365ba253b5e95a875b9fc62490efeb4f5e3697201b7c4de471c99dcc44761bc404fba0a659f485df359d9e988693ab6d4ecfe3b98c52a0c5e9a08336c260107ba0d78076908a96655c68e486811fbc404b33f2c513e2880ac71848965bd938c861c2fe1813fe61eccd782862a9158b7fa415d355b780ed9fbf5deaff4ce21e757b54c6241cb4fff4875a0553087d2174'
    im = Image.open(requests.get(url, stream=True).raw)
    model = DETR(num_classes=91)
    state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
    model.load_state_dict(state_dict)
    model.eval()

    scores, boxes = detect(im, model, transform)
    plot_results(im, scores, boxes)

    true_boxes = argv[2]
    if int(argv[1]) == 1:
        visualize_images(model, transform)
        mean_average_precision(boxes, true_boxes)

if __name__ == "__main__":
    main(sys.argv)