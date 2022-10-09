from sklearn.metrics import f1_score, mean_squared_error,mean_absolute_error,accuracy_score, recall_score, precision_score
from math import sqrt
from sklearn.metrics import r2_score
import numpy as np
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bad", type=str)
args = parser.parse_args()
print(args)
prefix = "cnn_test_result_"
suffix  = "_bagging_0"
for bagging in range(10):
	suffix = "_bagging_"+str(bagging)
	for i in range(5):
		print("cp cnn_epoch"+str(i)+"_bagging_"+str(bagging)+".pkl " + str(bagging) + ".pkl")
		result = json.load(open(prefix + str(i)+".json"))
		labels = eval(result["labels"])
		pred = eval(result["predictions"])
		print(mean_squared_error(labels, pred))
		print(sqrt(mean_squared_error(labels, pred)))
		print(mean_absolute_error(labels, pred))
		print(accuracy_score(labels, pred))
		print(f1_score(labels, pred,average="macro"))
		print("*"*20)


# result = json.load(open("bagging_result.json"))
# labels = eval(result["labels"])
# pred = eval(result["predictions"])
# print(mean_squared_error(labels, pred))
# print(sqrt(mean_squared_error(labels, pred)))
# print(mean_absolute_error(labels, pred))
# print(accuracy_score(labels, pred))
# print(f1_score(labels, pred,average="macro"))
# print("*"*20)