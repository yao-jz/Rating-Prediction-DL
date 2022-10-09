import json
# 加载计图，以及其他通用包
import jittor as jt
import time
from tokenization import BertTokenizer
from lstm_model import LSTM_Net
from bert_model import BertClassification, BertConfig
from jittor.optim import AdamW
from tqdm import tqdm
import sys
def exit():
	sys.exit(0)
# 预测
def predict(outputs):
	probabilities = jt.nn.softmax(outputs["logits"], dim=1)
	predictions = jt.argmax(probabilities, dim=1)[0]
	return predictions
# 开启 GPU 加速
jt.flags.use_cuda = 1
test_data = open("test.source","r").read().split("\n")
test_label = [int(float(i)) for i in open("test.target","r").read().split("\n")]
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
vocab_file = "vocab.txt"
tokenizer = BertTokenizer(do_lower_case=True, model_max_length=512, vocab_file=vocab_file)
print("start tokenize")
test_encodings = tokenizer(list(test_data), padding=True)
print("tokenizer over")
test_dataset = MyDataset(test_encodings, list(test_label)).set_attrs(batch_size=1, shuffle=False)
configuration = BertConfig()
bert = BertClassification(configuration)
bert.load_state_dict(jt.load("jittorhub://pretrained_bert.bin")) 
embed = bert.bert.embeddings
for i in range(10):
	model = LSTM_Net(
		embedding_dim=768,
		hidden_dim=2048,
		num_layers=1,
		bidirectional=False
	)
	model.load_state_dict(jt.load("lstm_"+str(i)+".pkl"))
	model.eval()
	with jt.no_grad():  # 关闭梯度
		correct = 0
		count = 0
		record = {"labels":list(), "predictions":list()}
		pbar = tqdm(test_dataset, total=len(test_dataset)//test_dataset.batch_size)
		for batch in pbar:
			input_ids = embed.execute(batch['input_ids'])
			labels = batch['labels']
			outputs = model(input_ids=input_ids, labels=labels)
			loss = outputs['loss']
			predictions = predict(outputs)
			correct += predictions.equal(labels.reshape(-1)).sum().item()
			count += len(labels)
			accuracy = correct * 1.0 / count
			pbar.set_postfix({
				'loss': '{:.3f}'.format(loss.item()),
				'accuracy': '{:.3f}'.format(accuracy)
			})
			record["labels"] += list(labels.reshape(-1).numpy())
			record["predictions"] += list(predictions.numpy())
			
		pbar.close()
	for k in record.keys():
		record[k]=str(record[k])
	time.sleep(0.3)
	print(u"测试集上准确率: %s%%" % round(accuracy*100,4))
	json.dump(record, open("lstm_test_result_"+str(i)+".json","w"))

