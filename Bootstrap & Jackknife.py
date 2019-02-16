# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 21:40:54 2019

@author: Brian Cote
"""

###bootstrap

def Bootstrap(data,B):
    corr_boots = np.zeros(B)
    t_stats = np.zeros(B)
    
    for i in range(0,B):
        #Set up sample vector, sample 31 random integers uniformly
        index = np.random.randint(0,len(data),len(data))
        #Sample our values
        sample = data[index,:]
        #Get our value
        corr_boots[i] = pearsonr(sample[:,0], sample[:,1])[0]
        
        corr_boot_se = np.zeros(B)
        for j in range(0,B):
            index1 = np.random.randint(0,14,14)
            sample2 = sample[index1,:]
            corr_boot_se[j] = pearsonr(sample2[:,0], sample2[:,1])[0]
        se_boot = np.std(sample2, ddof=1)
        
        #Getting our t-statistic
        t_stats[i] = (corr_boots[i] - corr_overall)/se_boot
    
    #Bias
    bootBias = np.mean(corr_boots) - corr_overall
  
    #Overall Bootstrap Standard Error
    se = 0
    for i in range(0,B):
        se = se + (corr_boots[i]-np.mean(corr_boots))**2
    bootSE = ((1/(B-1))*se)**(1/2)
    
    return(np.mean(corr_boots),bootBias, bootSE, np.sort(t_stats))
    
res = Bootstrap(data,500)
print(res[0])
print(res[1])
print(res[2])


###jackknife

def jackknife(data):
    
    #Setup
    n = len(data)
    jk = np.zeros(n)
    corr_jk = np.zeros(n)
    

    for i in range(0,n):
        #Simply just gotta delete the ith observation for the ith jackknife sample
        jk = np.delete(data,(i),axis=0)
       
        corr_jk[i] = pearsonr(jk[:,0],jk[:,1])[0]
        
    #Jackknife Quenouille's Bias
    jk_bias = (n-1)*(np.mean(corr_jk) - corr_overall)
    
    #Jackknife's Variance
    se_corr = 0
    for i in range(0,n):
        se_corr = se_corr + (corr_jk[i] - corr_overall)**2
    jk_var = ((n-1)/n)*se_corr
    
    return(jk_bias, np.sqrt(jk_var), np.mean(corr_jk))

print(jackknife(data))    

###CI for bootstrap

def CI(res, alpha,B):
    t1 = res[3][int(round(B*(1-(alpha/2))))]
    t2 = res[3][int(round(B*alpha/2))]
    UL = res[0] - t2*res[2]
    LU = res[0] - t1*res[2]

    return(LU, UL)
print(CI(res,0.05,500))
