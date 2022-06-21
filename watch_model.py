from model_modified import Yolov1
from torchsummary import summary
import torch 
import torch.nn as nn

model_path = '/Users/josefinabalzer/Desktop/SS22/BA/Output/DOTA_135/overfit_DOTA_135.pth.tar'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Yolov1(split_size=7, num_boxes=2, num_classes=18).to(DEVICE)

#x = torch.randn((1,3,896,896))
#m = model(x)
#print(m.shape)

if DEVICE == 'cuda':
	checkpoint = torch.load(model_path)
else:
	checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

S = 14
B = 2
C = 18


#model.load_state_dict(checkpoint['state_dict'])
model.fcs = nn.Sequential(
	nn.Flatten(),
	nn.Linear(1024 * S * S, 4096), # Hier liegt das Problem! Wenn sich S ändert, passt der Linear Layer nicht mehr. Dieses Problem wird wird nicht von der grundlegenden Architektur beeinflusst 
	nn.Dropout(0.5),
	nn.LeakyReLU(0.1),
	nn.Linear(4096, S * S * (C + B * 5)), # classification layer?
)
model.to(DEVICE)

summary(model, (3,896,896))
#summary(model, (3,1792,1792))

