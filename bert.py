import json
# 加载计图，以及其他通用包
import jittor as jt
import time
from tokenization import BertTokenizer
from bert_model import BertClassification, BertConfig
from jittor.optim import AdamW
from tqdm import tqdm
def predict(outputs):
    probabilities = jt.nn.softmax(outputs["logits"], dim=1)
    predictions = jt.argmax(probabilities, dim=1)[0]
    return predictions
jt.flags.use_cuda = 1
train_data = open("train.source", "r").read().split("\n")
test_data = open("test.source","r").read().split("\n")
train_label = [int(float(i)) for i in open("train.target","r").read().split("\n")]
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
train_encodings = tokenizer(list(train_data), padding=True)
test_encodings = tokenizer(list(test_data), padding=True)
train_dataset = MyDataset(train_encodings, list(train_label)).set_attrs(batch_size=4, shuffle=False)
test_dataset = MyDataset(test_encodings, list(test_label)).set_attrs(batch_size=1, shuffle=False)
configuration = BertConfig()
model = BertClassification(configuration)
model.load_state_dict(jt.load("jittorhub://pretrained_bert.bin")) # 加载预训练的参数
optim = AdamW(model.parameters(), lr=1e-5)
epoch = 10
model.train()
train_loss = list()
train_accuracies = list()
for epoch_i in range(epoch):
    print('Epoch %s/%s' % (epoch_i + 1, epoch))
    time.sleep(0.3)
    correct = 0
    count = 0
    epoch_loss = list()
    pbar = tqdm(train_dataset, total=len(train_dataset)//train_dataset.batch_size)
    for batch in pbar:
        optim.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
    jt.save(model.state_dict(), "bert_" + str(epoch_i) + ".pkl")
    train_loss += epoch_loss
    train_accuracies.append(accuracy)
result = {}
result["train_loss"] = train_loss
result["train_accuracies"]  = train_accuracies
json.dump(result, open("./bert_train_result.json", "w"))
