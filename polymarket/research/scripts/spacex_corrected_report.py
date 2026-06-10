"""
Corrected mirror of the colleague's ev_report front page: distribution chart + KEY TRADE
METRICS box + DISTRIBUTION PERCENTILES box, using the BEST-ASK liquidity-weighted lognormal
(taker convention) and the corrected 13.076B share count. Single-hump, honest tails.

ENV: run from the repo root with system python (numpy>=2 / scipy / matplotlib; research venv lacks scipy):
  python3 polymarket/research/scripts/spacex_corrected_report.py
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SHARES=13_075_865_175
def c2p(c): return (c*1e12)/SHARES
def p2c(p): return (p*SHARES)/1e12
IPO=135.0; PERP=165.7; IPO_CAP=p2c(IPO)

continuous=[(1.0,99.2,1.4),(1.2,98.3,1.8),(1.4,96.6,3.6),(1.6,91.5,8.8),(1.8,78.0,23.0),
 (2.0,64.0,37.0),(2.2,46.0,55.0),(2.4,30.0,71.0),(2.6,15.0,86.0),(2.8,11.0,90.0),
 (3.0,7.0,94.0),(3.2,3.6,96.7),(3.4,2.6,97.8),(3.6,2.4,98.5),(3.8,1.8,98.5),(4.0,1.3,99.0)]
strikes=np.array([c[0] for c in continuous])
yes_ask=np.array([c[1] for c in continuous])/100             # BEST-ASK survivor (taker)
yes_bid=np.array([100-c[2] for c in continuous])/100
spread=np.maximum(yes_ask-yes_bid,0.001); w=1/spread**2; w/=w.sum()

def lnS(K,mu,sg): return 1-norm.cdf((np.log(np.maximum(K,1e-9))-mu)/sg)
mu,sg=minimize(lambda p:np.sum(w*(lnS(strikes,*p)-yes_ask)**2),[np.log(2.1),0.25],method="Nelder-Mead").x

xf=np.linspace(0.01,5.0,20000); dx=xf[1]-xf[0]
pdf=np.maximum(np.gradient(1-lnS(xf,mu,sg),xf),0); pdf/=np.trapezoid(pdf,xf)
price=c2p(xf); pdf_price=pdf*(SHARES/1e12)
cdf=np.cumsum(pdf)*dx
mean=c2p(np.trapezoid(xf*pdf,xf)); med=c2p(xf[np.searchsorted(cdf,0.5)]); mode=c2p(xf[np.argmax(pdf)])
sd=c2p(np.sqrt(np.trapezoid((xf-np.trapezoid(xf*pdf,xf))**2*pdf,xf)))
pwin=float(lnS(IPO_CAP,mu,sg)); ploss=1-pwin
mw=xf>=IPO_CAP; ml=xf<IPO_CAP
ew=c2p(np.trapezoid(xf[mw]*pdf[mw],xf[mw])/pwin); el=c2p(np.trapezoid(xf[ml]*pdf[ml],xf[ml])/ploss)
ag=ew-IPO; al=IPO-el; ev=pwin*ag-ploss*al; kelly=pwin-ploss*al/ag
pct={p:c2p(xf[np.searchsorted(cdf,p/100)]) for p in [1,5,10,25,50,75,90,95,99]}

fig=plt.figure(figsize=(11,11))
fig.text(0.5,0.965,"SpaceX IPO Day-1 Close — Corrected Construction (best-ask, liquidity-weighted)",
         ha="center",fontsize=15,fontweight="bold",color="#1B3A5C")
fig.text(0.5,0.945,"Single-mode lognormal on the 16-strike best-ask survivor | 13.076B shares | "
         "mirrors the colleague report with the multi-peak artifact removed",ha="center",fontsize=9,color="#666")
gs=GridSpec(2,2,figure=fig,height_ratios=[1.15,1],hspace=0.28,wspace=0.18,top=0.91,bottom=0.05,left=0.08,right=0.95)

# --- distribution chart (his style) ---
ax=fig.add_subplot(gs[0,:])
ax.fill_between(price,pdf_price,where=price>=IPO,alpha=.18,color="#00A651")
ax.fill_between(price,pdf_price,where=price<IPO,alpha=.18,color="#CC0000")
ax.plot(price,pdf_price,color="#2E75B6",lw=2)
ax.axvline(IPO,color="#00A651",lw=2.5);ax.axvline(PERP,color="#FF6600",lw=2,ls="--")
ax.axvline(mean,color="#CC0000",lw=1.4,alpha=.8);ax.axvline(med,color="#9933CC",lw=1.4,ls="--",alpha=.8)
ymax=pdf_price.max()
ax.annotate("IPO $135",(IPO,ymax*.78),color="#00A651",fontsize=9,fontweight="bold",ha="center")
ax.annotate(f"Perp ${PERP:.0f}",(PERP,ymax*.9),color="#FF6600",fontsize=9,fontweight="bold",ha="center")
ax.annotate(f"Mean ${mean:.0f}",(mean+3,ymax*.62),color="#CC0000",fontsize=8)
ax.annotate(f"Median ${med:.0f}",(med-32,ymax*.5),color="#9933CC",fontsize=8)
ax.text(232,ymax*.72,f"P(close > $135) = {pwin*100:.0f}%",fontsize=13,fontweight="bold",color="#00A651",
        bbox=dict(boxstyle="round,pad=0.4",fc="#E8F5E9",ec="#00A651"))
ax.text(78,ymax*.72,f"P(< $135) = {ploss*100:.0f}%",fontsize=10,color="#CC0000",
        bbox=dict(boxstyle="round,pad=0.3",fc="#FFEBEE",ec="#CC0000"))
ax.set_xlim(50,350);ax.set_xlabel("Day-1 Closing Price ($/share)",fontsize=11)
ax.set_ylabel("Probability Density",fontsize=11)
ax.set_title("Crowd-Implied Day-1 Closing Price Distribution (single-mode, no artifact)",fontsize=12,fontweight="bold",color="#1B3A5C")
ax.grid(alpha=.2)

# --- KEY TRADE METRICS box ---
axm=fig.add_subplot(gs[1,0]);axm.axis("off")
metrics=f"""KEY TRADE METRICS  (corrected)
{'-'*34}

Entry Price:          ${IPO:.2f}
Expected Close:       ${mean:.1f}
Median Close:         ${med:.1f}
Mode:                 ${mode:.1f}
Perp Confirmation:    ${PERP:.1f}

P(close > $135):      {pwin*100:.1f}%
P(close < $135):      {ploss*100:.1f}%

If win:  avg gain     +${ag:.1f}  (+{ag/IPO*100:.0f}%)
If loss: avg loss     -${al:.1f}  (-{al/IPO*100:.0f}%)

Expected Value:       +${ev:.1f}/sh (+{ev/IPO*100:.1f}%)
Kelly Fraction:       {kelly*100:.0f}%
Std Deviation:        ${sd:.1f}/sh

CONSTRUCTION: best-ask, liquidity-weighted"""
axm.text(0.04,0.97,metrics,transform=axm.transAxes,fontsize=9.5,va="top",family="monospace",
         bbox=dict(boxstyle="round",fc="#E8F5E9",ec="#00A651",lw=2))

# --- DISTRIBUTION PERCENTILES box (corrected vs colleague original) ---
axp=fig.add_subplot(gs[1,1]);axp.axis("off")
orig={1:72,5:113,10:124,25:140,50:164,75:187,90:215,95:235,99:308}
lab={1:"extreme crash",5:"bad day",10:"weak open",25:"below exp.",50:"median/base",
     75:"good pop",90:"strong pop",95:"euphoric",99:"extreme rip"}
lines=["DISTRIBUTION PERCENTILES","-"*40,f"{'':4s}{'corrected':>10s}{'colleague':>11s}",""]
for p in [1,5,10,25,50,75,90,95,99]:
    lines.append(f"P{p:<3d} ${pct[p]:>7.0f}   ${orig[p]:>7.0f}   {lab[p]}")
lines+=["","corrected tails (P1 $96 / P99 $279) are","tighter: colleague's $72/$308 were","inflated by the PCHIP+anchor artifact."]
axp.text(0.04,0.97,"\n".join(lines),transform=axp.transAxes,fontsize=9,va="top",family="monospace",
         bbox=dict(boxstyle="round",fc="#FFF8E1",ec="#FF9800",lw=2))

import os; os.makedirs("polymarket/research/data/analysis/plots/overview",exist_ok=True)
out="polymarket/research/data/analysis/plots/overview/spacex_corrected_report.png"
plt.savefig(out,dpi=140,bbox_inches="tight",facecolor="white")
print("saved",out)
print(f"best-ask 13.076B: P(win)={pwin*100:.1f}% mean=${mean:.1f} median=${med:.1f} mode=${mode:.1f} "
      f"sd=${sd:.1f} EV=+${ev:.1f} kelly={kelly*100:.0f}%")
print("percentiles:",{p:round(pct[p]) for p in pct})
