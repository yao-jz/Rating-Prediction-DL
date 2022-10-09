from sklearn.svm import LinearSVC
import joblib
from sklearn import tree
import json
# 加载计图，以及其他通用包
import jittor as jt
import time
from tokenization import BertTokenizer
from lstm_model import LSTM_Net
from bert_model import BertClassification, BertConfig
from jittor.optim import AdamW
from tqdm import tqdm
import random
import numpy as np
from cnn_model import CNN_Net
import sys
def exit():
	sys.exit(0)
def predict(outputs):
	probabilities = jt.nn.softmax(outputs["logits"], dim=1)
	predictions = jt.argmax(probabilities, dim=1)[0]
	return predictions
T=10

class MyDataset(jt.dataset.Dataset):
	def __init__(self, encodings, labels):
		super().__init__()
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: jt.array(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = jt.array(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)
print("loading data")
jt.flags.use_cuda = 1
train_data = open("train.source", "r").read().split("\n")
test_data = open("test.source","r").read().split("\n")
train_label = [int(float(i)) for i in open("train.target","r").read().split("\n")]
test_label = [int(float(i)) for i in open("test.target","r").read().split("\n")]
vocab_file = "vocab.txt"
tokenizer = BertTokenizer(do_lower_case=True, model_max_length=512, vocab_file=vocab_file)
configuration = BertConfig()
bert = BertClassification(configuration)
bert.load_state_dict(jt.load("jittorhub://pretrained_bert.bin")) 
embed = bert.bert.embeddings
print("start bagging...")
train_length = len(train_data)
for i in tqdm(range(5,10)):
	print("i = ",i)
	model = CNN_Net(256)
	optim = AdamW(model.parameters(), lr=1e-5)
	epoch = 5
	model.train()
	sub_x_train = []
	sub_y_train = []
	for j in range(train_length):
		index = int(random.random()*train_length)
		sub_x_train.append(train_data[index])
		sub_y_train.append(train_label[index])
	print("start tokenize")
	train_encodings = tokenizer(list(sub_x_train), padding=True)
	print("tokenizer over")
	train_dataset = MyDataset(train_encodings, list(sub_y_train)).set_attrs(batch_size=2, shuffle=False)
	for epoch_i in range(epoch):
		print('Epoch %s/%s' % (epoch_i + 1, epoch))
		correct = 0
		count = 0
		epoch_loss = list()
		pbar = tqdm(train_dataset, total=len(train_dataset)//train_dataset.batch_size)
		for batch in pbar:
			optim.zero_grad()
			input_ids = embed.execute(batch['input_ids'])
			input_ids = input_ids.reshape(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[2])
			labels = batch['labels']
			outputs = model(input_ids=input_ids, labels=labels)
			loss = outputs['loss']
			optim.step(loss)
			predictions = predict(outputs)
			correct += predictions.equal(labels.reshape(-1)).sum().item()
			count += len(labels)
			accuracy = correct * 1.0 / count
			pbar.set_postfix({
				'Loss': '{:.3f}'.format(loss.item()),
				'Accuracy': '{:.3f}'.format(accuracy)
			})
			epoch_loss.append(loss.item())
		pbar.close()
		jt.save(model.state_dict(), "cnn_epoch" + str(epoch_i) + "_bagging_"+str(i)+".pkl")