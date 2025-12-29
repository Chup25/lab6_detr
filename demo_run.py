import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
model.eval()

url = "https://images.unsplash.com/photo-1546182990-dffeafbe841d?auto=format&fit=crop&w=800&q=60"
img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

import torchvision.transforms as T
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
inp = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(inp)

probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # без "no-object"
keep = probas.max(-1).values > 0.7                      # поріг

boxes = outputs['pred_boxes'][0, keep].cpu()
scores = probas[keep].max(-1).values.cpu()

w, h = img.size
boxes_xyxy = torch.zeros_like(boxes)
boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

plt.figure(figsize=(10, 7))
plt.imshow(img)
ax = plt.gca()

for (x0, y0, x1, y1), sc in zip(boxes_xyxy, scores):
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.text(x0, y0, f"{sc:.2f}", bbox=dict(facecolor="white", alpha=0.7))

plt.axis("off")
plt.savefig("result.png", bbox_inches="tight", dpi=200)
print("Готово: збережено result.png")