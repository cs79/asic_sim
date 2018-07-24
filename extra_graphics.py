# throwaway plots for illustration on Medium

import pandas as pd
import matplotlib.pyplot as plt

# "Ideal" distribution - 45 degree line
ideal_x = [i / 10 for i in list(range(0, 10))] + [1]
ideal_y = ideal_x
ideal_df = pd.DataFrame(index=ideal_x, data={'% of network hashpower': ideal_y})
ax = ideal_df.plot(linewidth=2)
ax.set_xlabel('% of miners controlling')
ax.set_ylabel('% of hashpower controlled')
ax.grid(which='both')
plt.savefig('C:/Users/cloud/Dropbox/Projects/asic_sim/ideal_dist.png')
plt.close()

# "Worst-case" distribution - right angle
wc_x = [i / 100 for i in list(range(0, 100))] + [1]
wc_y = [0, .99] + [.99 + (i * (1 - .99) / 99) for i in range(1, 100)]
wc_df = pd.DataFrame(index=wc_x, data={'% of network hashpower': wc_y})
ax2 = wc_df.plot(linewidth=2)
ax2.set_xlabel('% of miners controlling')
ax2.set_ylabel('% of hashpower controlled')
ax2.grid(which='both')
plt.savefig('C:/Users/cloud/Dropbox/Projects/asic_sim/wc_dist.png')
plt.close()
