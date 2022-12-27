##### Util functions using ecl data#####
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from Sim1_Util import plot_2D, roundhalf, normal_dist
from scipy.stats import ttest_ind

def sample_job(jobs_data, keys = None, key_prob = None, sample_rm = False):
    '''sample one job_ot from the job_data
       sample_rm: whether to remove sampled job'''
    if isinstance(keys,list):
        job_key = np.random.choice(keys,1, p=key_prob)[0]
        job_ot = random.sample(jobs_data[job_key],1)[0]
        if sample_rm:
            jobs_data[job_key].remove(job_ot)
    else:
        job_ot = random.sample(jobs_data,1)[0]
        if sample_rm:
            jobs_data.remove(job_ot)
    return job_ot

def data_to_csv(path, newpath):
    '''creates a copy data that has all nan values removed'''
    data = pd.read_csv(path)
    data = data.copy()
    for key in data.keys():
        for i in range(len(data[str(key)])):
            if math.isnan(data.loc[i,[str(key)]]):
                data.loc[i,[str(key)]] = -1
            else:
                thing = float(data.loc[i,[str(key)]])
                #if thing <= 0:
                    #print(str(key), i, thing)
                data.loc[i,[str(key)]] = thing

    data.to_csv(newpath)

def csv_to_data(path, attr=None):
    '''read the ptcl data csv and return the desired attr array'''
    data = pd.read_csv(path)
    data_f = data.copy()
    if not attr == None:
        if attr == 'op_num':
             return sorted(list(set(data_f[attr])),reverse=True)
        else: return data_f[attr]
    else:
        return data_f


def csv_to_jobs(path, mul=1):
    '''read the ptcl data csv and product job dict by type'''
    data_f = pd.read_csv(path)
    data_f = data_f.copy()

    jobs = dict() #list of each type of ptcl
    jobs_prob = dict() #proportion of each type of ptcl

    for key in list(data_f.keys())[1:]:
        if '.5' not in key:
            jobs[key] = []
            jobs_prob[key] = 0
            for i in range(len(data_f[key])):
                if data_f.loc[i,key] > 0:
                    '''is rounding up/down the job_ot acceptable?'''
                    jobs[key].append(max(roundhalf(mul*data_f.loc[i,key]),1))
                elif data_f.loc[i,key] == -1:
                    break
    assert(len(list(jobs.keys()))==64)

    total_jobs = 0 #total number of ptcls
    total_ot = 0
    for key in jobs.keys():
        total_jobs += len(jobs[key])
        for ot in jobs[key]:
            total_ot += ot
    for key in jobs.keys():
        jobs_prob[key] = len(jobs[key])/total_jobs

    #average operating time of all ptcls
    mean_ot = total_ot/total_jobs
    return jobs, jobs_prob, mean_ot, total_ot, total_jobs

def res_to_csv(test_res, colnames, rownum, path):
    '''produce a csv file using the test data'''
    mat = np.zeros((rownum,len(colnames))) 
    for i in range(len(test_res)):
        pack = []
        for col in colnames:
            if col in test_res[i]: #col is a key of test_param_i
                if col == 'eps':
                    test_res[i][col] = round(test_res[i][col],2)
                pack.append(test_res[i][col])
            else:
                pack.append('-99999')
                print(f'{i}th test_param is missing',col)
        mat[i, :] = np.array(pack)

    cg_l = dict()
    for i in range(len(colnames)):
        cg_l[colnames[i]] = mat[:,i]
    cg_l_df = pd.DataFrame(data = cg_l)
    cg_l_df.to_csv(path)
    print('csv done')
    
def plot_scatter(test_res, pause_probs = [0,1], save = (False,''), code=''):
    '''plot scatter avg_wt vs. eps plots. pause prob = 0 and 1'''
    data_list = [0]*len(pause_probs)
    for i in range(len(pause_probs)):
        data_list[i] = ([test_case for test_case in test_res if test_case['pause_prob']==pause_probs[i]])
    
    if data_list[0][0]['fix_lam']== -1:
        lam = data_list[0][0]['cus_lam']
    else:
        lam = data_list[0][0]['fix_lam']
    scale = lam*data_list[0][0]['cus_num']
    #data containers for plotting
    plot_info_0 = [];plot_info_1 = []
    if len(pause_probs)==2:
        colors = ['b','r']
    else:
        colors = ['b','g','r','m','c']
    
    if len(pause_probs) > len(colors):
        styles = [f'{c}o' for c in colors]
        styles += ['o']*(len(pause_probs)-len(colors))
    else:
        styles = [f'{c}o' for c in colors[:len(pause_probs)]]
    assert(len(styles)==len(pause_probs))
    
    legs = [f'P={p}' for p in pause_probs]
    #plot_info.append([[data.eps,data.cus_num],'bo',''])
    plot_info = []
    for i in range(len(pause_probs)):
        plot_info.append([[[x['eps'] for x in data_list[i]],[x['avg_wt'] for x in data_list[i]]],styles[i],legs[i]])
    
    title = f'{code}Avg. Wait Time vs. Eps, scale={scale}'
    input_label = 'op. surplus %'
    output_label = 'avg. waiting time (hour)'

    plot_2D(plot_info,title,input_label,output_label,bloc='best',save=save)
    

def get_multiplier(k, tol = 0.01, length = 300, check = False):
    '''returns the multiplier s.t. (mean_ot'+1)=k*(mean_ot+1) within tol-difference'''

    path = "ecldata/ecldata_without_float.csv"
    jobs, jobs_prob, mean_ot_s, total_ot, total_jobs = csv_to_jobs(path, mul=1)
    '''rounding up/down lambda acceptable?'''
    
    f_lam = int((1/k)*roundhalf(total_jobs/length))
    
    c = (k*(mean_ot_s+1)-1)/mean_ot_s
    jobs_c, jobs_prob_c, mean_ot_c, total_ot_c, total_jobs_c = csv_to_jobs(path, mul=c)
    count = 1
    
    while(abs(k-(mean_ot_c+1)/(mean_ot_s+1)) > tol):
        
        if k-(mean_ot_c+1)/(mean_ot_s+1) > tol:
            c += tol
            jobs_c, jobs_prob_c, mean_ot_c, total_ot_c, total_jobs_c = csv_to_jobs(path, mul=c)

        elif (mean_ot_c+1)/(mean_ot_s+1)-k > tol:
            c -= tol
            jobs_c, jobs_prob_c, mean_ot_c, total_ot_c, total_jobs_c = csv_to_jobs(path, mul=c)
        count += 1
    if check:
        print('c = '+str(round(c,3))+' k\' = '+str(round((mean_ot_c+1)/(mean_ot_s+1),3))) 
        print(f'used {count} iterations')
    f_rls = roundhalf(mean_ot_c/2)
    return c, f_lam, f_rls
    

def get_random_data(jobs, show=False):
    '''generate random data similar to ecl's 
       using t-test for 20 times, requiring p<0.05 chance is less than 10%'''
    jobss = []
    for key in jobs.keys():
        if len(jobs[key]) >0:
            jobss += jobs[key]
    count = 0
    pall = []
    for k in range(20):
        random_job = []
        for i in range(len(jobss)-800):
            random_job.append(np.abs(normal_dist(7.5, 5, up = 57.5, low = -42.5)))
        for i in range(200):
            random_job.append(np.abs(normal_dist(20, 10, up = 50, low = -20)))
        for i in range(200):
            random_job.append(normal_dist(30, 10, up = 50, low = 1))
        for i in range(200):
            random_job.append(normal_dist(30, 10, up = 70, low = 1))
        for i in range(100):
            random_job.append(normal_dist(40, 10, up = 100, low = 1))
        for i in range(100):
            random_job.append(normal_dist(65, 80, up = 319, low = 1))
        for j in range(len(random_job)):
            if random_job[j] == 0:
                random_job[j] = normal_dist(65, 80, up = 319, low = 1)
        (stat,p) = ttest_ind(jobss,random_job)
        pall.append(p)
        if (p < 0.05):
            count += 1
    assert(len(random_job)==len(jobss))
    assert(count <= 2)
    if show:
        print('mean p=',np.mean(pall))
        print('ECL mean',np.mean(jobss))
        print('Normal mean',np.mean(random_job))

        print(f'2-sample t-test value={stat}, p={p}')
        
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10, 6))
        ax[0].hist(jobss, bins=[5*i for i in range(20)],color='b',label='ECL data')
        ax[0].legend(fontsize=14)
        ax[1].hist(random_job, bins=[5*i for i in range(20)],color='m',label='Normal random data')
        ax[1].legend(fontsize=14)
        plt.show()
    return random_job