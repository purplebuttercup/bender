
#h = open('./data/Op.team.big.txt','r', encoding='utf-8', errors='ignore')


#for line in h.readlines():
def giveTeams(h):
    teams = []
    teams = teams + [0] * 40

    for line in h.readlines():
        rest = line.split("\t")
        if (2 < len(rest)):
            #get team index
            name = rest[2]
            if (name != 'NULL'):
                name = name.split(" ")
                nr = name[0][-2:]
                #print (nr)
                nr = int(nr)

                #place team in vector
                teams[nr] = 1

    return teams

#h.close()
#print(teams)
#print ("Fin")