 Sample Project

Setup Elastic Search,  Kibana, Flask APP
--------------------------------------------------------

step 1:  docker-compose.yml file
step 2:  Run the above file to steup the containers
         docker-compose up -d   


Stop the containers

docker-compose down


kibana UI URL : http://localhost:5601/app

Elastic Search URl : http://localhost:9200

Run the application Container
-----------------------------------------------------------

Local Run
-----------------------------------

pip install -r requirements.txt

docker build -t elk_proj_flaskapp:latest .



Swagger Json
-----------------------------------------------------------

http://192.168.0.100:9097/swagger.json

=====================================================================
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()

# Define the transform for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
])

def change_background(image_path, background_color=(255, 255, 255)):
    # Load the image
    input_image = cv2.imread(image_path)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Transform the image for the model
    image_tensor = transform(input_image_rgb).unsqueeze(0).to(device)

    # Perform segmentation
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    
    # Get the predicted segmentation mask
    output_predictions = output.argmax(0).cpu().numpy()

    # Create a mask for the foreground
    mask = (output_predictions == 15)  # 15 is the label for 'person' in COCO dataset
    mask = np.stack([mask] * 3, axis=-1)  # Convert to 3 channels

    # Create the new background
    background = np.full_like(input_image_rgb, background_color)
    
    # Change the background
    combined = np.where(mask, input_image_rgb, background)

    return combined

# Change the background of an input image
output_image = change_background('path/to/your/image.jpg', background_color=(255, 255, 255))

# Display the output
plt.imshow(output_image)
plt.axis('off')
plt.title('Background Changed')
plt.show()
