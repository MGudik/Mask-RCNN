import torch
from model import get_model_instance_segmentation
import torch
from PIL import Image
import torchvision.transforms as transforms

def infer(image):
    model = get_model_instance_segmentation(10)
    model.eval()
    
    image = Image.open(image)
    transform = transforms.ToTensor()
    tensor = torch.unsqueeze(transform(image),0)

    output = model(tensor)

    print(output)



if __name__ == '__main__':
    infer('img.jpg')