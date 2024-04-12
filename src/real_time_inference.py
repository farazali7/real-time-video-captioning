import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from PIL import Image
from transformers import BertTokenizer

model_path = 'results/student_model.pt'
model = torch.load(model_path,map_location=torch.device('cpu'))
model.eval()

class BGR2RGBTransform:
    def __call__(self, x):
        return x[[2, 1, 0], ...]

def image_transform():
    crop_size = 224
    trans = [
        ToTensor(),
        Resize(crop_size, interpolation=Image.BICUBIC),
        CenterCrop(crop_size),
        BGR2RGBTransform(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
    ]
    return Compose(trans)

transform = image_transform()

cap = cv2.VideoCapture(0)
frames = []
latest_caption = ''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Increment the frame counter
    frame_counter += 1

    # Every 5th frame, process and reset the counter
    # Can modify this to control frame rate
    if frame_counter == 3:
        frame_pil = Image.fromarray(frame)
        transformed_frame = transform(frame_pil).unsqueeze(0)
        frames.append(transformed_frame)
        
        # Reset frame counter
        frame_counter = 0

    if len(frames) == 6:
        input_tensor = torch.cat(frames, dim=0).unsqueeze(0)
        caption = model.greedy_decode(input_tensor,max_len=25)
        preds = [tokenizer.decode(c.tolist(), skip_special_tokens=True) for c in caption]
        latest_caption = preds[0]
        frames.clear()
    
    # Overlay the latest caption on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 0, 255)  # Red color
    thickness = 6
    text_size = cv2.getTextSize(latest_caption, font, font_scale, thickness)[0]
    position = ((frame.shape[1] - text_size[0]) // 2, frame.shape[0] - 40)
    cv2.putText(frame, latest_caption, position, font, font_scale, color, thickness)
    

    # Display the frame with the caption
    cv2.imshow('Webcam Live with Caption', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()