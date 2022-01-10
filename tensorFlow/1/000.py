list1 = (1,2,3,4,5,6,7,8,9,0)
list2 = (0,9,8,7,6,5,4,3,2,1)

for i in zip(list1, list2) :
    print(i)

for i, j in zip(list1, list2) :
    print(i, '\\', j)