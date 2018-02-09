import pickle
import numpy as np

f = open('./data/EN.Categories.txt','r', encoding='utf-8', errors='ignore')
g = open('./data/data.txt', 'wb')
train = open('./data/train.txt', 'w')
val = open("./data/val.txt", "w")
test = open("./data/test.txt", "w")
nr = 1

val_set_x = []
val_set_y = []

test_set_x = []
test_set_y = []

train_set_x = []
train_set_y = []

train_set = np.array([])
val_set = np.array([])
test_set = np.array([])


for line in f.readlines():
    nr = nr + 1
    rest = line.split("\t")
    if (3 < len(rest)):

        #lowercase all characters
        rest[3] = rest[3].lower()

        #process codes
        if (rest[0] == '00'):
            rest[0] = 1
        if (rest[0] == '11'):
            rest[0] = 2
        if (rest[0] == '12'):
            rest[0] = 3
        if (rest[0] == '13'):
            rest[0] = 4
        if (rest[0] == '16'):
            rest[0] = 5
        if (rest[0] == '17'):
            rest[0] = 6
        if (rest[0] == '18'):
            rest[0] = 7
        if (rest[0] == '21'):
            rest[0] = 8
        if (rest[0] == '22'):
            rest[0] = 9
        if (rest[0] == '23'):
            rest[0] = 10
        if (rest[0] == '24'):
            rest[0] = 11
        if (rest[0] == '25'):
            rest[0] = 12
        if (rest[0] == '26'):
            rest[0] = 13
        if (rest[0] == '27'):
            rest[0] = 14
        if (rest[0] == '28'):
            rest[0] = 15
        if (rest[0] == '31'):
            rest[0] = 16
        if (rest[0] == '32'):
            rest[0] = 17
        if (rest[0] == '33'):
            rest[0] = 18
        if (rest[0] == '34'):
            rest[0] = 19
        if (rest[0] == '35'):
            rest[0] = 20
        if (rest[0] == '36'):
            rest[0] = 21
        if (rest[0] == '37'):
            rest[0] = 22
        if (rest[0] == '41'):
            rest[0] = 23
        if (rest[0] == '51'):
            rest[0] = 24
        if (rest[0] == '52'):
            rest[0] = 25
        if (rest[0] == '54'):
            rest[0] = 26
        if (rest[0] == '61'):
            rest[0] = 27
        if (rest[0] == '62'):
            rest[0] = 28
        if (rest[0] == '63'):
            rest[0] = 29
        if (rest[0] == '64'):
            rest[0] = 30
        if (rest[0] == '65'):
            rest[0] = 31
        if (rest[0] == '66'):
            rest[0] = 32
        if (rest[0] == '67'):
            rest[0] = 33
        if (rest[0] == '71'):
            rest[0] = 34
        if (rest[0] == '72'):
            rest[0] = 35
        if (rest[0] == '84'):
            rest[0] = 36
        if (rest[0] == '97'):
            rest[0] = 37
        if (rest[0] == '99'):
            rest[0] = 38

        #put data
        if (nr % 11 == 0):
            val_set_xx = []
            val_set_xx = val_set_xx + [0]*256
            i = 0;

            for c in rest[3]:
                val_set_xx[i] = ord(c) / 1000
                i = i + 1

            val_set_x.append(val_set_xx)
            val_set_y.append(int(rest[0]))
            #np.insert(val_set, t)
        elif (nr % 11 == 1):
            test_set_xx = []
            test_set_xx = test_set_xx + [0]*256
            i = 0;

            for c in rest[3]:
                test_set_xx[i] = ord(c) / 1000
                i = i + 1

            test_set_x.append(test_set_xx)
            test_set_y.append(int(rest[0]))
        else:
            train_set_xx = []
            train_set_xx = train_set_xx + [0]*256
            i = 0;

            for c in rest[3]:
                train_set_xx[i] = ord(c) / 1000
                i = i + 1

            train_set_x.append(train_set_xx)
            train_set_y.append(int(rest[0]))


t = [test_set_x, test_set_y]

for t_x, t_y in list(zip(test_set_x, test_set_y)):
    test.write(''.join('%s' % t_x))
    test.write('\n')
    test.write(''.join('%s' % t_y))
    test.write('\n\n')


val_set_xxx = np.asarray(val_set_x)
val_set_yyy = np.asarray(val_set_y)
test_set_xxx = np.asarray(test_set_x)
test_set_yyy = np.asarray(test_set_y)
train_set_xxx = np.asarray(train_set_x)
train_set_yyy = np.asarray(train_set_y)

val_set = (val_set_xxx, val_set_yyy)
train_set = (train_set_xxx, train_set_yyy)
test_set = (test_set_xxx, test_set_yyy)

dataset = (train_set, val_set, test_set)
pickle.dump(dataset, g)

f.close()
g.close()
test.close()

print ("Fin")