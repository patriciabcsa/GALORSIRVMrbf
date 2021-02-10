
import csv
import pandas as pd
import numpy 
'''
with open('BD/psd_DEAP/testFile_07.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('BD/psd_DEAP_segunda/testFile_07.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)
'''

dfData = pd.read_csv('BD/psd_DEAP_segunda/testFile_03.csv')

op=dfData['op']


for n,i in enumerate(op):
   if i == 1:
       op[n] = 1
   if i == 2:
       op[n] = 2
   if i == 3:
       op[n] = 1
   if i == 4:
       op[n] = 2

for a in op:
    print(a)
numpy.savetxt("BD/psd_DEAP_segunda/arousallabel.csv",op,delimiter=",")


'''

with open('BD/psd_DEAP_segunda/testFile_00.csv', 'w') as csvFile:
		#writer = csv.writer(csvFile)
		csvFile.writelines(['op',str(a)])
csvFile.close()
'''
