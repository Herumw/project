a={1:1,7:4,5:1}
b=sorted(a.items(),key=lambda student:student[0],reverse=True) #reverse说明是从高到低排序，默认为false
print(type(b[0]))