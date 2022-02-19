import torch
from model import get_model_instance_segmentation
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2

def infer(image):
    
    
    transform = transforms.ToTensor()
    tensor = torch.unsqueeze(transform(image),0).to(torch.device('cuda'))

    output = model(tensor)[0]
    scores = output['scores'].cpu().detach().numpy()
    masks = output['masks'].cpu().detach().numpy()

    return masks[0,0,]




if __name__ == '__main__':
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load('10E_model_weight.pth'))
    model.to(device=torch.device('cuda'))
    model.eval()
    
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()

        mask = infer(frame)

        cv2.imshow(frame)
        cv2.waitKey(1)