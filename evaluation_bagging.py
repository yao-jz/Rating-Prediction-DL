import json
import jittor as jt
from tokenization import BertTokenizer
from cnn_model import CNN_Net
from bert_model import BertClassification, BertConfig
from tqdm import tqdm
from collections import defaultdict
def predict(outputs):
	probabilities = jt.nn.softmax(outputs["logits"], dim=1)
	predictions = jt.argmax(probabilities, dim=1)[0]
	return predictions
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

print("evaluating bagging tree")
model_list = [CNN_Net(256) for i in range(10)]
for i in range(10):
	model_list[i].load_state_dict(jt.load(str(i)+".pkl"))
	model_list[i].eval()
with jt.no_grad():
	record = {"labels":[], "predictions":[]}
	correct = 0
	count = 0
	pbar = tqdm(test_dataset, total=len(test_dataset)//test_dataset.batch_size)
	for batch in pbar:
		label = defaultdict(int)
		input_ids = embed.execute(batch['input_ids'])
		input_ids = input_ids.reshape(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[2])
		labels = batch['labels']
		for i in range(10):
			outputs = model_list[i](input_ids=input_ids, labels=labels)
			predictions = predict(outputs)
			label[list(predictions.numpy())[0]]	+= 1
		this_pred = max(label.items())[0]
		if this_pred == list(labels.reshape(-1).numpy())[0]:
			correct += 1
		count += 1
		accuracy = correct * 1.0 / count
		pbar.set_postfix({
			'Accuracy': '{:.3f}'.format(accuracy)
		})
		
		record["labels"] += list(labels.reshape(-1).numpy())
		record["predictions"] += [this_pred]
	pbar.close()
for k in record.keys():
	record[k]=str(record[k])
print(u"测试集上准确率: %s%%" % round(accuracy*100,4))
print(record)
json.dump(record, open("bagging_result.json","w"))