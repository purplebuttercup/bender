import pickle
import numpy as np

import ClassMap
import Teams

f = open('./data/EN.Categories.txt','r', encoding='utf-8', errors='ignore')
g = open('./data/data.txt', 'wb')
h = open('./data/Op.team.big.txt','r', encoding='utf-8', errors='ignore')
test = open("./data/test.txt", "w")
nr = 1
categ = []
categ = categ + [0]*39*40

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

        code = rest[2]
        print(code)
        for hline in h.readlines():

            hrest = hline.split("\t")
            if (1 < len(hrest)):

                hcode = hrest[1]
                if (code == hcode):
                    # get team index
                    name = hrest[2]
                    if (name != 'NULL'):
                        name = name.split(" ")
                        nr = name[0][-2:]
                        nr = int(nr)

                        categ[rest[0]*40 + nr] = categ[rest[0]*40 + nr] + 1

        h.seek(0)
nrl = 1
nrc = 1
teams = Teams.giveTeams(h)
print (teams)
test.write('\"stat_name\",\"stat_label\",\"stat_qtr\",\"stat_year\",\"pt_group\",\"qtr_result\"')
test.write('\n')
for ca in categ:
    if (teams[nrl-1] != 0):
        test.write('\"' + ''.join('%s' % ClassMap.classes[nrc]) + '\"')
        test.write(',')
        test.write('\"New\"')
        test.write(',')
        #if (ca != 0)
        test.write(''.join('%s' % nrl))
        test.write(',')
        test.write('2008')
        test.write(',')
        test.write('\"All\"')
        test.write(',')
        test.write(''.join('%s' % ca))
        test.write('\n')

    if (nrl % 40 == 0 and nrl != 0):
        nrl = 0
        nrc = nrc + 1

    nrl = nrl + 1
   # test.write(''.join('%s' % t_y))
   # test.write('\n\n')


f.close()
h.close()
g.close()
test.close()

print ("Fin")