import json
import jittor as jt
from tokenization import BertTokenizer
from bert_model import BertClassification, BertConfig
from tqdm import tqdm
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
for i in range(10):
	model = BertClassification(configuration)
	model.load_state_dict(jt.load("bert_"+str(i)+".pkl"))
	model.eval()
	with jt.no_grad():
		correct = 0
		count = 0
		record = {"labels":[], "predictions":[]}
		pbar = tqdm(test_dataset, total=len(test_dataset)//test_dataset.batch_size)
		for batch in pbar:
			input_ids = batch['input_ids']
			attention_mask = batch['attention_mask']
			labels = batch['labels']
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
	print(u"测试集上准确率: %s%%" % round(accuracy*100,4))
	json.dump(record, open("bert_test_result_"+str(i)+".json","w"))