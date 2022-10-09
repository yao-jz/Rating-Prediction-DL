s = open("source_total.txt", "r").read().split("\n")
t = open("target_total.txt", "r").read().split("\n")
def write_list(l, f):
	file = open(f, "w")
	for i in l[:-1]:
		file.write(i+"\n")
	file.write(l[-1])
	file.close()
# tl = 40000
tl=int(0.8*len(s))
write_list(s[:tl],"train.source")
write_list(t[:tl],"train.target")
write_list(s[tl:],"test.source")
write_list(t[tl:],"test.target")