from models.resnet import resnet18
import os

model = resnet18()
checkpoint = os.path.join("checkpoints/resnet18-Baseline", 'model_%d.pth' % 14)
model.load_state_dict(checkpoint["model"])

