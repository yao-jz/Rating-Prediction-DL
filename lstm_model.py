from jittor import nn
from jittor.nn import CrossEntropyLoss
class LSTM_Net(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, num_layers, bidirectional=False):
		super(LSTM_Net, self).__init__()
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
		self.classifier = nn.Linear(hidden_dim, 5)
	def execute(self, input_ids=None, labels=None):
		outputs, _ = self.lstm(input_ids)
		outputs = outputs[-1,:,:]
		outputs=self.classifier(outputs)
		loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(outputs.view(-1, 5), labels.view(-1))
		return {
			"loss": loss,
			"logits": outputs,
			}

model = LSTM_Net(
	embedding_dim=768,
	hidden_dim=2048,
	num_layers=1,
	bidirectional=False
)
num_params = sum(param.numel() for param in model.parameters())
print(num_params)