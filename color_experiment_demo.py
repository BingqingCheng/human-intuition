import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Slider, Button, RadioButtons
from math import pi

# initialize
[ vr0, vg0, vb0 ] = [ random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
[ vr, vg, vb ] = [ 0.0, 0.0, 0.0]
memory = []
memory.append([ -1.0, vr0, vg0, vb0, 1. ])

# set up the plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

# this is for drawing a heart curve
tl = np.arange(0.0, 1.*pi, 2.*pi/1000.)
tr = np.arange(pi, 2.*pi, 2.*pi/1000.)
a0 = 16
b0 = 13
b1 = -5
b2 = -2
b3 = -1
xl = a0*(np.sin(tl))**3.0
yl = b0*np.cos(tl)+b1*np.cos(2.*tl)+b2*np.cos(3.*tl)+b3*np.cos(4.*tl)
xr = a0*(np.sin(tr))**3.0
yr = b0*np.cos(tr)+b1*np.cos(2.*tr)+b2*np.cos(3.*tr)+b3*np.cos(4.*tr)

# color function
def setcolor(a1,a2,a3):
    cf1 = np.sin(a1*pi)
    cf2 = abs(np.cos(a2*pi) + np.sin(a2*pi))/2.
    cf3 = abs(np.cos(a3*pi) + np.cos(-a2*pi))/(np.exp(a3/pi) + np.exp(a2/pi))/2.
    return cf1,cf2,cf3

# similarity function
def calsim(col1, col2):
    return int(100.*(1.0-np.linalg.norm(np.array(col1)-np.array(col2))/np.sqrt(3.)))

# set up the plot
r,  = plt.plot(xr, yr, lw=20, color='black')
r.set_color([vr0, vg0, vb0])
sim = calsim([vr0, vg0, vb0],[vr, vg, vb])
l, = plt.plot(xl, yl, lw=20, color='black', label="change")
sim = calsim([vr0, vg0, vb0],[vr, vg, vb])
ax.text(-15, 17, "Target", size=12, rotation=0.,
         ha="center", va="center")
ax.text(15, 17, "Your blend", size=12, rotation=0.,
         ha="center", va="center")

plt.axis([-25, 25, -35, 20])
plt.xticks([])
plt.yticks([])
plt.title('Blend the three elements (fire, water, earth) to get the target color')

# add gagets
axcolor = 'lightgoldenrodyellow'
axfire = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axwater = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axearth = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

sfire = Slider(axfire, 'Fire', 0, 1, valinit=0)
swater = Slider(axwater, 'Water',0, 1, valinit=0)
searth = Slider(axearth, 'Earth',0, 1, valinit=0)

resetax = plt.axes([0.5, 0.025, 0.4, 0.04])
button = Button(resetax, 'Try this blend', color=axcolor, hovercolor='0.975')


def confirm(event):
    [ vr, vg, vb ] = setcolor(sfire.val,swater.val,searth.val)
    l.set_color([vr, vg, vb])
    sim = calsim([vr0, vg0, vb0],[vr, vg, vb])
    ax.text(0, -27, "similarity: "+str(sim)+"%", size=20, rotation=0.,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 1., 1.),
                   )
         )
    fig.canvas.draw_idle()
    timestamp=time.mktime(time.gmtime())
    memory.append([ timestamp, vr, vg, vb, sim/100. ])
button.on_clicked(confirm)

# exit botton
resetax = plt.axes([0.15, 0.025, 0.1, 0.05])
exitbutton = Button(resetax, 'Exit', color='red', hovercolor='0.975')

def record(event):
    timestamp=time.mktime(time.gmtime())
    np.savetxt("memory-his-"+str(timestamp)+".dat",memory,fmt='%d %4.8f %4.8f %4.8f %4.2f')
    exit()  
exitbutton.on_clicked(record)

plt.show()
