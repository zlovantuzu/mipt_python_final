import torch
from torchvision import transforms
from cell_counter.cnn_deep_learning.model import load_trained_model

model = load_trained_model()
model.eval()

def count_cells_cnn(img):
    image = img.convert('L')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    return int(output.item())