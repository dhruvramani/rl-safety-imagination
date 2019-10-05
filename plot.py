import matplotlib.pyplot as plt

unsafe_count = []
safe_count = []
with open("./unsafe_state_count_False.txt", "r") as f:
	for i in f:
		unsafe_count.append(int(i))

with open("./unsafe_state_count_True.txt", "r") as f:
	for i in f:
		safe_count.append(int(i))
'''
plt.plot(unsafe_count, label="Unsafe Policy")
plt.plot(safe_count, label="Safe Policy")
plt.ylabel('Unsafe States Visited')
plt.xlabel('Timesteps')
#plt.show()
plt.savefig('state_count.png')
'''

#import tensorflow as tf
from tensorboardX import SummaryWriter

writer = SummaryWriter('./plots/')

for i in range(len(safe_count)):
	writer.add_scalars("Unsafe States Visited", {"Safe Policy" : safe_count[i], 'Unsafe Policy' : unsafe_count[i],}, i)
writer.close()
