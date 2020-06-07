import matplotlib.pyplot as plt
x=["1","2","4","8","12","24","48","64"]
y=[3.37,3.09,3.05,3.06,3.26,3.69,4.94,5.99]
y = [i/15 for i in y]
plt.xlabel("No. of Clusters")
plt.ylabel("Training Time Per Epoch(s)")
plt.title("Citeseer")
plt.plot(x,y)
plt.show()
axes = plt.gca()
axes.set_ylim([0.5,1])
#to change y axis scaling