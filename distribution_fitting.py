import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import os

def best_fit_distribution(distributions,data, bins = 200, ax = None):
    '''Model data by finding distribution to data'''
    y,x = np.histogram(data, bins = bins, density = True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    DISTRIBUTIONS = distributions  
    for distribution in DISTRIBUTIONS:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                params = distribution.fit(data)
                arg = params[:-2]
                loc = params[:-2]
                scale = params[-1]
                pdf = distribution.pdf(x, loc = loc, scale = scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                cdf = distribution.cdf(x, loc = loc, scale = scale, *arg)
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass
    return (best_distribution.name, best_params,best_sse)

def make_cdf_pdf(dist, params, size = 10000):
    '''Generate disrtibutions Probability Distribution Function & Cumilative Distribution Function'''

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    cdf_y = dist.cdf(x, loc=loc, scale=scale, *arg)
    cdf = pd.Series(cdf_y,x)
    pdf_y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(pdf_y, x)
    return (cdf,pdf)

def make_ecdf(data):
    '''Empirical Cumilative Distribution Plot'''
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/ len(x)
    ecdf = pd.Series(y,x)    
    return ecdf

def legend_name(dist, dist_params):
    '''Function to generate the legend in the plot'''
    param_names = (dist.shapes + ', loc, scale').split(', ') if dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, dist_params)])
    dist_str = '{}({})'.format(dist.name, param_str)
    return dist_str

def get_file_name():
    for root, dirs, files in os.walk(".", topdown=False):
        for f_name in files:
            if f_name.endswith('.csv'):
                csv_files.append(f_name)
    return csv_files

csv_files = []
get_file_name()
csv_files.sort()
print('The CSV FILES ARE :')
print(csv_files)
best_dists = []
second_best_dists = []
third_best_dists = []

for f_name in csv_files:
    #Distributions to check (89 distributions from the statsmodel)
    distributions =  [st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy ] 
    pdf = pd.read_csv(f_name)
    data = pdf["Residual(kbps)"]

    #Finding the best distribution

    best_fit_name, best_fit_params,best_sse = best_fit_distribution(distributions,data, 200)
    best_dist = getattr(st, best_fit_name)
    best_dists.append(best_dist.name)
    print(best_fit_name)

    #Second best distribution

    distributions.remove(best_dist)
    second_best_fit_name, second_best_fit_params,second_best_sse = best_fit_distribution(distributions,data, 200)
    second_best_dist = getattr(st, second_best_fit_name)
    second_best_dists.append(second_best_dist.name)
    print(second_best_fit_name)

    #Third distribution
   
    distributions.remove(second_best_dist)
    third_best_fit_name, third_best_fit_params,third_best_sse = best_fit_distribution(distributions,data, 200)
    third_best_dist = getattr(st, third_best_fit_name)
    third_best_dists.append(third_best_dist.name)
    print(third_best_fit_name)

    '''Generating CDF and PDF of the data'''
    first_cdf,first_pdf = make_cdf_pdf(best_dist, best_fit_params)
    second_cdf,second_pdf = make_cdf_pdf(second_best_dist, second_best_fit_params)
    third_cdf,third_pdf = make_cdf_pdf(third_best_dist, third_best_fit_params)
    ecdf = make_ecdf(data)

    #Plotting CDF for top 3 distributions
    plt.figure(figsize=(12,8))
    plt.margins(0.02)
    ax = first_cdf.plot(lw=2,label = legend_name(best_dist,best_fit_params) + ", error : " + str(best_sse), legend=True, color = 'r', alpha = 0.5)
    ax = second_cdf.plot(lw=2, label = legend_name(second_best_dist,second_best_fit_params) + ", error : " + str(second_best_sse), legend=True, color = 'g', alpha = 0.5)
    ax = first_cdf.plot(lw=2, label = legend_name(third_best_dist,third_best_fit_params) + ", error : " + str(third_best_sse), legend=True, color = 'b', alpha = 0.5 , style = '--')
    ax = ecdf.plot(lw=2, label ='Empirical CDF', legend = True, style = '-', alpha = 0.5)
    ax.legend(loc=2, fontsize = 9)
    ax.set_title(u'Error distribution CDF \n')
    ax.set_xlabel(u'Residual')
    ax.set_ylabel(u'Probability')
    plt.savefig(f_name + "_CDF.svg",format='svg', dpi=200,bbox_inches='tight')
    plt.close()
    #Plotting PDF for top 3 distributions
    plt.figure(figsize=(12,8))
    plt.margins(0.02)
    ax = first_pdf.plot(lw=2, label = legend_name(best_dist,best_fit_params), legend=True, color = 'r')
    ax = second_pdf.plot(lw=2, label = legend_name(second_best_dist,second_best_fit_params), legend=True, color = 'g')
    ax = third_pdf.plot(lw=2, label = legend_name(third_best_dist,third_best_fit_params), legend=True, color = 'b')
    data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True)
    ax.legend(loc=2, fontsize = 9)
    ax.set_title(u'Error distribution PDF \n')
    ax.set_xlabel(u'Residual')
    ax.set_ylabel(u'Probability')
    plt.savefig(f_name + "_PDF.svg",format='svg', dpi=200,bbox_inches='tight')
    plt.close()
print("Best Distributions")
print(best_dists)
print("Second Best Distributions")
print(second_best_dists)
print("Third Best Distributions")
print(third_best_dists)













