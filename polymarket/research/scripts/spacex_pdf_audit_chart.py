"""
SpaceX PDF methodology-audit figure (4 panels) — reproduces the figure embedded in
polymarket/research/notes/overview/data_quality/spacex_pdf_construction_audit.md

Panels: (A) the colleague's shipped PCHIP density with its spurious local maxima marked;
(B) the assumption-free interval-mass histogram + the liquidity-weighted unimodal lognormal;
(C) ladder-implied vs standalone bucket-market overlay; (D) per-strike bid-ask spread/mid.

NOTE: this audit figure intentionally uses the colleague's 13.091B share convention (SHARES_M
below) so panel A replicates their shipped chart exactly. The production builder
(spacex_pdf_builder_v2.py) uses the corrected S-1/A count of 13.076B.

ENV: run with system python (numpy>=2 / scipy / matplotlib); the research venv lacks scipy.
  python3 polymarket/research/scripts/spacex_pdf_audit_chart.py
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.signal import argrelextrema
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHARES_M = 13091
def c2p(c): return (c*1e12)/(SHARES_M*1e6)
def p2c(p): return (p*SHARES_M*1e6)/1e12

continuous_raw = [(1.0,99.2,1.4),(1.2,98.3,1.8),(1.4,96.6,3.6),(1.6,91.5,8.8),
    (1.8,78.0,23.0),(2.0,64.0,37.0),(2.2,46.0,55.0),(2.4,30.0,71.0),(2.6,15.0,86.0),
    (2.8,11.0,90.0),(3.0,7.0,94.0),(3.2,3.6,96.7),(3.4,2.6,97.8),(3.6,2.4,98.5),
    (3.8,1.8,98.5),(4.0,1.3,99.0)]
strikes=np.array([c[0] for c in continuous_raw])
yes_ask=np.array([c[1] for c in continuous_raw])/100
yes_bid=np.array([(100-c[2]) for c in continuous_raw])/100
mids=(yes_ask+yes_bid)/2; spread=yes_ask-yes_bid

bucketed=[("<1.0T",0.0,1.0,0.7,99.4),("1.0-1.5",1.0,1.5,3.9,97.0),("1.5-2.0",1.5,2.0,40.0,62.0),
    ("2.0-2.5",2.0,2.5,43.0,58.0),("2.5-3.0",2.5,3.0,12.7,88.8),("3.0-3.5",3.0,3.5,4.1,97.3),("3.5T+",3.5,5.0,1.2,99.1)]
b_lab=[b[0] for b in bucketed]; b_lo=np.array([b[1] for b in bucketed]); b_hi=np.array([b[2] for b in bucketed])
b_mid=np.array([(b[3]+100-b[4])/2 for b in bucketed])/100; b_norm=b_mid/b_mid.sum()

# PCHIP
x=np.concatenate([[0.0],strikes,[4.5,5.0]]); s=np.concatenate([[1.0],mids,[0.005,0.001]])
for i in range(1,len(s)):
    if s[i]>=s[i-1]: s[i]=s[i-1]-0.0001
sp=PchipInterpolator(x,s); xf=np.linspace(0.01,5.0,10000)
pdf=np.maximum(-sp.derivative()(xf),0); pdf/=np.trapezoid(pdf,xf)
pdf_price=pdf*(SHARES_M*1e6/1e12); price=c2p(xf)
pk=[i for i in argrelextrema(pdf,np.greater,order=50)[0] if pdf[i]>0.05*pdf.max()]

# lognormal fit (liquidity-weighted)
w=1/np.maximum(spread,0.01)**2; w/=w.sum()
def lnS(K,mu,sg): return 1-norm.cdf((np.log(K)-mu)/sg)
r=minimize(lambda p:np.sum(w*(lnS(strikes,*p)-mids)**2),[np.log(2.1),0.25],method="Nelder-Mead")
mu,sg=r.x
ln_pdf=np.gradient(1-lnS(xf,mu,sg),xf); ln_pdf=np.maximum(ln_pdf,0); ln_pdf/=np.trapezoid(ln_pdf,xf)
ln_pdf_price=ln_pdf*(SHARES_M*1e6/1e12)

# continuous-implied bucket probs
impl=np.array([np.trapezoid(pdf[(xf>=lo)&(xf<hi)],xf[(xf>=lo)&(xf<hi)]) for lo,hi in zip(b_lo,b_hi)])

fig,ax=plt.subplots(2,2,figsize=(15,11))
fig.suptitle("SpaceX IPO crowd PDF — methodology audit (data: Polymarket 2026-06-07, 13.091B sh)",fontsize=14,fontweight="bold")

# A: shipped PCHIP w/ spurious peaks
a=ax[0,0]
a.fill_between(price,pdf_price,alpha=.25,color="#2E75B6"); a.plot(price,pdf_price,color="#2E75B6",lw=2)
for i in pk: a.axvline(c2p(xf[i]),color="red",ls=":",alpha=.7); a.annotate(f"${c2p(xf[i]):.0f}",(c2p(xf[i]),pdf_price[i]),color="red",fontsize=8,ha="center")
a.axvline(135,color="green",lw=2,label="IPO $135"); a.set_xlim(50,350)
a.set_title(f"A. Shipped PCHIP PDF: {len(pk)} spurious local maxima (red)",fontweight="bold")
a.set_xlabel("$/share"); a.set_ylabel("density"); a.legend(fontsize=8); a.grid(alpha=.2)

# B: honest histogram + unimodal fit
b=ax[0,1]
mass=-np.diff(mids); pe=c2p(strikes); centers=(pe[:-1]+pe[1:])/2; widths=np.diff(pe)
heights=mass/widths  # probability density per $/share = mass / price-bin width
b.bar(centers,heights,width=widths*0.92,alpha=.45,color="#888",label="interval-mass histogram (raw data)")
b.plot(price,ln_pdf_price,color="#CC0000",lw=2.5,label=f"liquidity-wtd lognormal (unimodal)\nmean ${c2p(np.exp(mu+sg**2/2)):.0f} med ${c2p(np.exp(mu)):.0f}")
b.axvline(135,color="green",lw=2,label="IPO $135"); b.set_xlim(50,350)
b.set_title("B. What the data actually says: ONE broad hump",fontweight="bold")
b.set_xlabel("$/share"); b.set_ylabel("density"); b.legend(fontsize=8); b.grid(alpha=.2)

# C: bucket overlay
c=ax[1,0]; xp=np.arange(len(b_lab)); ww=0.38
c.bar(xp-ww/2,impl*100,ww,label="continuous-ladder implied",color="steelblue",alpha=.85)
c.bar(xp+ww/2,b_norm*100,ww,label="bucket market (normalized)",color="coral",alpha=.85)
for i in range(len(b_lab)):
    g=(impl[i]-b_norm[i])*100
    c.annotate(f"{g:+.1f}",(xp[i],max(impl[i],b_norm[i])*100+0.6),ha="center",fontsize=8,
               color="red" if abs(g)>5 else ("orange" if abs(g)>3 else "green"),fontweight="bold")
c.set_xticks(xp); c.set_xticklabels(b_lab,rotation=40,ha="right",fontsize=8)
c.set_title("C. Cross-market overlay: ladder-implied vs bucket market (gap pp)",fontweight="bold")
c.set_ylabel("probability %"); c.legend(fontsize=8); c.grid(alpha=.2,axis="y")

# D: liquidity proxy
d=ax[1,1]; ratio=spread/np.maximum(mids,1e-6)
cols=["#CC0000" if rr>0.15 else "#2E75B6" for rr in ratio]
d.bar(range(len(strikes)),ratio,color=cols,alpha=.8)
d.axhline(0.15,color="red",ls="--",alpha=.5,label="noise-dominated threshold")
d.set_xticks(range(len(strikes))); d.set_xticklabels([f">{k:.1f}" for k in strikes],rotation=45,fontsize=7)
d.set_title("D. Per-strike liquidity proxy: bid-ask spread / mid (red = noisy tail)",fontweight="bold")
d.set_xlabel("strike ($T)"); d.set_ylabel("spread / mid"); d.legend(fontsize=8); d.grid(alpha=.2,axis="y")

plt.tight_layout(rect=[0,0,1,0.97])
import os; os.makedirs("polymarket/research/data/analysis/plots/overview",exist_ok=True)
out="polymarket/research/data/analysis/plots/overview/spacex_pdf_artifact_audit.png"
plt.savefig(out,dpi=140,bbox_inches="tight",facecolor="white")
print("saved",out)
print(f"bucket gaps (ladder-impl - bucket-mkt, pp): "+", ".join(f"{l}:{(impl[i]-b_norm[i])*100:+.1f}" for i,l in enumerate(b_lab)))
