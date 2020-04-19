#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:38:27 2020

@author: mralarcon
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter, LogLocator


# constants
beta = 8.629e-6         # cgs
k = 8.617332e-5	        # eV/K
h = 6.6261e-27          # cm2 g s-1
c = 2.99792458e10       # cm s-1

# atomic data given in Tabla 1
df = pd.DataFrame(
    {'i': ['3P1', '3P2', '3P2', '1D2', '1D2', '1D2', '1S0', '1S0', '1S0', '1S0'],
     'j': ['3P0', '3P0', '3P1', '3P0', '3P1', '3P2', '3P0', '3P1', '3P2', '1D2'],
     'Transicion': ['3P1-3P0', '3P2-3P0', '3P2-3P1',
                    '1D2-3P0', '1D2-3P1', '1D2-3P2',
                    '1S0-3P0', '1S0-3P1', '1S0-3P2', '1S0-1D2'],
     'lambda(A)': [2052800, 764300, 1217700, 6527.2, 6548.1,
                   6583.5, 3058.3, 3062.8, 3070.6, 5754.6],
     'A(1/s)': [2.083e-6, 1.12e-12, 7.42e-6, 5.253e-7, 9.851e-4,
                2.914e-3, 0., 3.185e-2, 1.547e-4, 1.136],
     'O(5000K)': [0.3591, 0.1435, 1.3731, 0.2788, 0.7517,
               1.4455, 0.0335, 0.0936, 0.1632, 0.3894],
     'O(10000K)': [0.38, 0.188, 1.45, 0.286, 0.76, 1.46,
                0.0333, 0.0996, 0.172, 0.522],
     'O(15000K)': [0.3888, 0.2219, 1.4734, 0.2889, 0.7711,
                1.4834, 0.0339, 0.1028, 0.1773, 0.5729],
     'O(20000K)': [0.3950, 0.2460, 1.49, 0.291, 0.779, 1.5,
                0.0343, 0.105, 0.181, 0.609],
     'gi': [3., 5., 5., 5., 5., 5., 1., 1., 1., 1.],
     'gj': [1., 1., 3., 1., 3., 5., 1., 1., 5., 5.],
      })

# user plotting function
def plotter(ax, x, y, xlabel, ylabel, color, label, fontsize,
            lstyle='solid', aph=1, xs='log',ys='log',legcol=1,legloc=0):
    # Format plt.axes for plotting the light curve
    ax.plot(x,y,color=color,label=label,linestyle=lstyle,alpha=aph)
    if label != None: ax.legend(loc=legloc,fontsize=fontsize,frameon=False,ncol=legcol)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)

  
    ax.minorticks_on()
    ax.tick_params(axis='both',direction='in',which='minor',
                   length=2,width=.5,labelsize=fontsize)
    ax.tick_params(axis='both',direction='in',which='major',
                   length=5,width=1,labelsize=fontsize)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    
    if xs == 'log':
        ax.set_xscale('log')
        locmin = LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                              numticks=100)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(NullFormatter())
        
    if ys == 'log':
        ax.set_yscale('log')
        locmin = LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                              numticks=100)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())


# collisional transition rates function
Te_df = [5000, 10000, 15000, 20000] 
def q(Te_list, df_transition):
    Te_array = np.array(Te_list)
    Eji = 1.2398e4 / df_transition['lambda(A)']
    O_df = df_transition.loc[:,'O(5000K)':'O(20000K)'].values
    O = []
    for o in O_df: O.append(np.interp(Te_array, Te_df, o))
    O = np.array(O)
    qij = beta * O / np.outer(df_transition['gi'].values,np.sqrt(Te_array))
    qji = beta * O / np.outer(df_transition['gj'].values,np.sqrt(Te_array)) * \
        np.exp(-np.outer(Eji,1/(k*Te_array)))
    return qij, qji

        
# relative population function
def rel_populations(ne, Te):
    M = []
    
    # calculate matrix elements
    for indx,l in enumerate(levels[:-1]):
        Mi = np.zeros(len(levels))
        
        lev_i = df[df['i'] == l]
        lev_j = df[df['j'] == l]
        if lev_i.size != 0: qi = q(Te, lev_i)
        if lev_j.size != 0: qj = q(Te, lev_j)
        
        i = 0
        for _,li in lev_i.iterrows():
            Mi[levels.index(li['j'])] = ne * qi[1][i]
            Mi[indx] -= ne * qi[0][i] + li['A(1/s)']
            i += 1
        j = 0
        for _,lj in lev_j.iterrows():
            Mi[levels.index(lj['i'])] = ne * qj[0][j] + lj['A(1/s)']
            Mi[indx] -= ne * qj[1][j]
            j += 1
        M.append(Mi)
    M.append(np.ones(len(levels)))
    
    # define and solve linear system
    M = np.array(M)
    B = np.append(np.zeros(len(levels)-1),np.array([1]))
    N = np.linalg.solve(M, B)
    return N    

# a) critical density T = 10000K
Te = [10000]
levels = ['3P0', '3P1', '3P2', '1D2', '1S0']
print('\nCritical density')
for l in levels[1:]: 
    A = np.sum(df[df['i'] == l]['A(1/s)'])
    qij, _ = q(Te, df[df['i'] == l])
    _, qji = q(Te, df[df['j'] == l])
    Q = np.sum(qij) + np.sum(qji)
    nc = A / Q
    print(r'Level %s: %.3e cm^-3' % (l,nc))
    

# b) relative population ne = [1,10^9] cm^-3, T = 10000K
ne_list = np.logspace(0,9,500)
Te = [10000]

n = []
for ne in ne_list:
    n.append(rel_populations(ne, Te))
n = np.array(n)

# plot
fig = plt.figure(1,figsize=(6,5))
plt.clf()
ax = plt.subplot(1,1,1)
colormap = plt.cm.get_cmap('gnuplot', 255)
co = colormap(np.linspace(0, 1, len(levels)+1))
ylabel = r'$n_i ~ (cm^{-3})$'; xlabel = r'$n_e ~ (cm^{-3})$'
fontsize = 14
for il,l in enumerate(levels):
    label = r'$' + '^' + l[:-1] + '_' + l[-1] + '$'
    plotter(ax, ne_list, n[:,il], xlabel, ylabel, co[il], label, fontsize,legcol=2)

plt.tight_layout()
#plt.show()
print('\nSaving plot b.pdf...')
plt.savefig('b.pdf',dpi=500)
print('Done!')


# c) relative population ne = 100 cm^-3, T = [5000,15000] K
ne = 100
Te_list = np.linspace(5000,15000,500)

T = []
for Te in Te_list:
    T.append(rel_populations(ne, [Te]))
T = np.array(T)

# plot
fig = plt.figure(2,figsize=(6,5))
plt.clf()
ax = plt.subplot(1,1,1)
xlabel = r'$T_e ~ (K)$'

for il,l in enumerate(levels):
    label = r'$' + '^' + l[:-1] + '_' + l[-1] + '$'
    plotter(ax, Te_list, T[:,il], xlabel, ylabel, co[il], label,fontsize,
            xs='lin',legcol=2, legloc=[0.03,0.23])

plt.tight_layout()
#plt.show()
print('\nSaving plot c.pdf...')
plt.savefig('c.pdf',dpi=500)
print('Done!')


# emission coefficients
def j(transition, N):
    trans_row = df[df['Transicion']==transition]
    initial_level = trans_row['i'].to_string(index=False)[1:]
    n = N[levels.index(initial_level)]
    nu = c / trans_row['lambda(A)'] * 1e8
    A = trans_row['A(1/s)']
    j = h * nu / (4*np.pi) * n * A
    return np.float(j)

def j_mod(ne, Te):
    Te_a = np.array(Te)
    return 8.23 * np.exp(2.5e4/Te_a) / (1+4.4e-3 * ne / Te_a**(1/2))

# d) fraction
ne_list = [1, 1e5, 1e9]
Te_list = np.linspace(5000,20000,100)
transitions = ['1D2-3P1', '1D2-3P2', '1S0-1D2']

# plot
fig = plt.figure(3,figsize=(6,5))
plt.clf()
ax = plt.subplot(1,1,1)
co = colormap(np.linspace(0, 1, 4))
ylabel = r'$\frac{j_{\lambda6548}+j_{\lambda6583}}{j_{\lambda5755}}$'
xlabel = r'$T_e ~ (K)$'
lab = ['100', '10^5', '10^9']

for e,ne in enumerate(ne_list):

    trans_list = []    
    for t in transitions:
        jnn = []
        for Te in Te_list:
            N = rel_populations(ne, [Te])
            jnn.append(j(t, N))
        trans_list.append(jnn)
        
    trans = np.array(trans_list)
    jc = (trans[0] + trans[1]) / trans[2]  

    label = r'$%s ~ cm^{-3}$' % lab[e]
    plotter(ax, Te_list, jc, xlabel, ylabel, co[e], label, fontsize, xs='lin')
    plotter(ax, Te_list, j_mod(ne,Te_list), xlabel, ylabel, co[e], None, 
            fontsize, lstyle='dashed', xs='lin')

plt.tight_layout()
#plt.show()
print('\nSaving plot d.pdf...')
plt.savefig('d.pdf',dpi=500)
print('Done!')