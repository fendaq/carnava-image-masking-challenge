total = open("test128x128_100064") 
total_line = total.readlines()

data1 = total_line[0:20000]
data2 = total_line[20000:40000]
data3 = total_line[40000:60000]
data4 = total_line[60000:80000]
data5 = total_line[80000:100064]

file1 = open("test128x128_100064_1", "w")
for i in range(len(data1)):
    file1.write(data1[i])
file1.close()

file2 = open("test128x128_100064_2", "w")
for i in range(len(data2)):
    file2.write(data2[i])
file2.close()

file3 = open("test128x128_100064_3", "w")
for i in range(len(data3)):
    file3.write(data3[i])
file3.close()

file4 = open("test128x128_100064_4", "w")
for i in range(len(data4)):
    file4.write(data4[i])
file4.close()

file5 = open("test128x128_100064_5", "w")
for i in range(len(data5)):
    file5.write(data5[i])
file5.close()