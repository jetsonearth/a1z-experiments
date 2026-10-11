#!/usr/bin/env python3
"""README figures for the DoD 4 comparison (2026-10-11). Numbers are copied from the Jetson run logs:
runs/20261011-112045-eval (HIL-SERL), runs/20261011-162501-eval (HG-DAgger), runs/20261011-144850-eval (BC, first 30),
runs/20261011-100404-train and runs/2026101-{150244,153747,160606}-dagger (collection curves)."""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[3]/'docs'/'images'
SURF='#fcfcfb';INK='#0b0b0b';INK2='#52514e';GRID='#e4e3df'
COL={'HIL-SERL':'#2a78d6','HG-DAgger':'#eb6834','BC':'#1baf7a'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK2,'xtick.color':INK2,
                     'ytick.color':INK2,'axes.facecolor':SURF,'figure.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False})


def wilson(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h


def results():
    rows=[('HIL-SERL',90,100,3.9,252),('HG-DAgger',42,100,8.6,77),('BC',3,30,None,None)]
    fig,ax=plt.subplots(1,3,figsize=(11,2.8),gridspec_kw={'wspace':0.45})
    names=[r[0] for r in rows][::-1]
    for i,(nm,k,n,t,h) in enumerate(rows[::-1]):
        lo,hi=wilson(k,n);p=k/n
        ax[0].plot([lo*100,hi*100],[i,i],color=COL[nm],lw=2,solid_capstyle='round')
        ax[0].plot(p*100,i,'o',ms=9,color=COL[nm],mec=SURF,mew=2)
        ax[0].text(hi*100+3,i,f'{k}/{n}',va='center',color=INK,fontsize=10)
        if t:ax[1].barh(i,t,height=0.5,color=COL[nm]);ax[1].text(t+0.2,i,f'{t:.1f} s',va='center',color=INK)
        else:ax[1].text(0.2,i,'3 successes only',va='center',color=INK2,fontsize=9)
        if h:ax[2].barh(i,h,height=0.5,color=COL[nm]);ax[2].text(h+5,i,f'~{h}',va='center',color=INK)
        else:ax[2].text(5,i,'not measured',va='center',color=INK2,fontsize=9)
    for a,title,xl in [(ax[0],'Success rate (95% CI)','% of fixed starts, USB-verified'),(ax[1],'Median time to insert','seconds, successful trials'),
                       (ax[2],'Successful insertions / hour','incl. resets and grasp checks')]:
        a.set_yticks(range(len(names)),names,color=INK);a.set_title(title,loc='left',color=INK,fontsize=11,pad=10);a.set_xlabel(xl)
        a.grid(axis='x',color=GRID,lw=0.8);a.set_axisbelow(True);a.tick_params(length=0);a.set_ylim(-0.6,len(names)-0.4)
    ax[0].set_xlim(0,110);ax[1].set_xlim(0,11);ax[2].set_xlim(0,300)
    fig.savefig(OUT/'result-comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)


def collection():
    rl=[(428,3,25),(779,5,25),(1067,9,25),(1329,8,25),(1464,17,25),(1527,20,25),(1534,22,25),(1534,24,25),(1534,4,4)]
    dg=[(1008,3,50),(1160,0,12),(1589,4,24)]   # per round: the model changes only between rounds
    fig,ax=plt.subplots(figsize=(7.5,3.8))
    for nm,pts,lab in [('HIL-SERL',rl,'per 25 episodes'),('HG-DAgger',dg,'per round')]:
        x=[p[0] for p in pts];y=[100*p[1]/p[2] for p in pts]
        ax.plot(x,y,'-',color=COL[nm],lw=2);ax.plot(x,y,'o',color=COL[nm],ms=7,mec=SURF,mew=2,label=f'{nm} ({lab})')
    ax.annotate('RL keeps improving with\nno extra human steps',xy=(1534,96),xytext=(980,80),color=INK2,fontsize=9,arrowprops=dict(arrowstyle='-',color=INK2,lw=0.8))
    ax.set_xlabel('cumulative human (takeover) steps during collection');ax.set_ylabel('% episodes solved with no takeover')
    ax.set_title('Autonomy while collecting data, against human effort',loc='left',color=INK,fontsize=11,pad=10)
    ax.set_ylim(-3,105);ax.set_xlim(0,1700);ax.grid(color=GRID,lw=0.8);ax.set_axisbelow(True);ax.tick_params(length=0)
    ax.legend(frameon=False,loc='upper left',labelcolor=INK)
    fig.savefig(OUT/'collection-curve.png',dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);results();collection();print('wrote',OUT)
