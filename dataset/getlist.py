
from random import shuffle
datalist = [i for i in range(30000)]
shuffle(datalist)

train = open('train.txt','w')
val = open('val.txt','w')
test = open('test.txt','w')

for i in range(30000):
    if i<25000:
        train.write(str(datalist[i])+'\n')
    elif i <27500:
        val.write(str(datalist[i])+'\n')
    else:
        test.write(str(datalist[i])+'\n')
train.close()
val.close()
test.close()