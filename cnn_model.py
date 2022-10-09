import jittor as jt
from jittor import nn
from jittor.nn import CrossEntropyLoss
class CNN_Net(nn.Module):
	def __init__ (self, feature = 256):
		super(CNN_Net, self).__init__()
		self.conv1 = nn.Conv (1, feature, kernel_size=(2,768), padding=0)
		self.conv2 = nn.Conv (1, feature, kernel_size=(3,768), padding=0)
		self.conv3 = nn.Conv(1, feature, kernel_size=(4,768), padding=0)
		self.conv4 = nn.Conv(1, feature, kernel_size=(5,768), padding=0)
		self.relu = nn.Relu()
		self.fc1 = nn.Linear (feature*4, feature*2)
		self.fc2 = nn.Linear (feature*2, 5)
	def execute (self, input_ids, labels) : 
		x=input_ids
		x1=self.relu(self.conv1(x))
		x2=self.relu(self.conv2(x))
		x3=self.relu(self.conv3(x))
		x4=self.relu(self.conv4(x))
		x4=jt.max(x4,dim=2,keepdims=True)
		x3=jt.max(x3,dim=2,keepdims=True)
		x2=jt.max(x2,dim=2,keepdims=True)
		x1=jt.max(x1,dim=2,keepdims=True)
		x=jt.concat((x1,x2,x3,x4),dim=2)
		x=jt.reshape(x, [x.shape[0], -1])
		x=self.fc2(self.relu(self.fc1(x)))
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(x.view(-1, 5), labels.view(-1))
		return {
			"loss": loss,
			"logits": x,
			}

model = CNN_Net(256)
num_params = sum(param.numel() for param in model.parameters())
print(num_params)