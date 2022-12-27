##### Running batches of tests #####
import sys
sys.path.append('../')
import numpy as np
from Sim13_ecl_Init import initialize, sim_run
from Sim13_ecl_Util import csv_to_jobs, res_to_csv, plot_scatter
from Sim13_Classes import operator
from Sim1_Util import plot_2D, roundhalf
import time

##### Batch-testing set-up function #####
def test_batch_wpic(jobs, jobs_prob, mean_ot, total_ot, total_jobs, 
                    epss, time_span = 300, f_lam = 10, f_rls = 8, poisson = False,
                    var_op = False, cus_num=1, pause_probs = [0,1], code = 't', 
                    f_op = None, runs = 10, plots = [True, True, True, False]):
    '''test a job batch based on eps values using data from path and pre-set param'''
    assert(isinstance(epss, np.ndarray))
    
    test_sets = setup(epss, x0 = 0, mean_ot = mean_ot, total = total_jobs,
                  f_lam = f_lam, f_rls = f_rls, runs = runs,
                  poisson = poisson, var_op = var_op, cycle_num = 300, 
                  cus_num=cus_num, pause_probs = pause_probs, f_op = f_op)
    rownum = 1*len(epss)*len(pause_probs)
    assert(len(test_sets)==rownum)

    print('total jobs =',total_jobs)
    print(f'f_lam = {f_lam}, originally',total_jobs/time_span)
    print(f'f_rls time = {f_rls} hours')
    print(f'mean ot = {mean_ot} hours')
    print(f'test size = {len(test_sets)}')
    print(f'poisson = {poisson}')
    print(f'var_op = {var_op}')
    print(f'cus_num = {cus_num}')
    print(f'pause_prob = {pause_probs}')
    print(f'job_scale = {f_lam*cus_num}')

    test_res = test_func(test_sets, cycle = 24, cycle_num = 300,
                         mean_ot = mean_ot, total = total_jobs,
                         jobs_data = jobs, jobs_prob = jobs_prob, sample_rm = False)
    assert(len(test_res)==rownum)


    get_mean(test_res, 'avg_qsz', buffer_time = 24*30, edit = True)
    get_mean(test_res, 'avg_qpc', buffer_time = 24*30, edit = True)
    get_mean(test_res, 'avg_wt', buffer_time = 24*30, edit = True)
    colnames =  ['op_num','cus_num','pause_prob','fix_lam','fix_rls', 'total_ptcls',
                 'multi_run','eps','bf_time','avg_qsz','avg_qpc','avg_wt','job_scale','mean_ot']

    cus = test_res[0]['cus_num']; lam = test_res[0]['fix_lam']
    scale = test_res[0]['job_scale']
    if test_res[0]['poisson']:
        pois = 'pois'
    else:
        pois = 'nopois'
    if test_res[0]['poisson']:
        var = 'var'
    else:
        var = 'novar'

    new_path = f'level_curve_dt/{code}_{f_lam}ecl_{pois}_{var}_{cus}cus.csv'
    res_to_csv(test_res, colnames, rownum, new_path)
    if plots[0]:
        plot_scatter(test_res, pause_probs = pause_probs, 
                     save=(True,f'pic/{code}_avgwt_{f_lam}ecl_{pois}_{var}_{cus}cus_{runs}runs'))
    
    test_res_prob = [0]*len(pause_probs)
    for i in range(len(pause_probs)):
        test_res_prob[i] = [test_case for test_case in test_res if test_case['pause_prob']==pause_probs[i]]

    mean_ot = roundhalf(mean_ot)
    runs = test_res[0]['multi_run']
    batch = int(np.ceil(len(epss)/4))
    for i in range(batch): 
        slice_prob = [0]*len(pause_probs)
        P_prob = [0]*len(pause_probs)
        
        for j in range(len(pause_probs)):
            slice_prob[j] = test_res_prob[j][i*4:min((i+1)*4,len(epss))]
            P_prob[j] = slice_prob[j][0]['pause_prob']
            assert(P_prob[j]==pause_probs[j])
            if plots[1]:
                plot_multi_QueuesTime(slice_prob[j], attr='eps', 
                                  title = f'Avg.Ot={mean_ot}+1, job scale={f_lam*cus_num}, P={P_prob[j]}, {runs}runs', buffer_time = 24*30, 
                                  plot = True, 
                                  save=(True,f'pic/QueueTime/{code}_s{i}_P={P_prob[j]}_{pois}_{var}_{scale}scale'))
            if plots[2]:
                plot_multi_WaitTime(slice_prob[j], attr='eps', 
                                title = f'Avg.Ot={mean_ot}+1, job scale={f_lam*cus_num}, P={P_prob[j]}, {runs}runs', buffer_time = 24*30, 
                                plot = True, 
                                save=(True,f'pic/WaitTime/{code}_s{i}_P={P_prob[j]}_{pois}_{var}_{scale}scale'))
            if plots[3]:
                plot_multi_OutstandTime(slice_prob[j], attr='eps', 
                                title = f'Avg.Ot={mean_ot}+1, job scale={f_lam*cus_num}, P={P_prob[j]}, {runs}runs', buffer_time = 24*30, 
                                plot = True, 
                                save=(True,f'pic/OutstandTime/{code}_s{i}_P={P_prob[j]}_{pois}_{var}_{scale}scale'))



def setup(epss, x0 = 0, mean_ot = None, total = None,
          f_lam = -1, f_rls = -1, runs = 10,
          poisson = False, var_op = False, cycle_num = None, 
          cus_num = 1, pause_probs = [0,1], f_op = None):
    assert(isinstance(epss, np.ndarray))
    '''test_sets = [test_param_0(P=0), test_param_0(P=1), 
                    test_param_1(P=0), test_param_1(P=1),
                    ...]'''
    test_sets = []
    if f_lam > 24 and not poisson:
        print(f'Error, fix lambda > limit={cycle} for non-poisson')
        f_lam = 24
    for i in range(len(epss)):
        '''op_num = [HO = (1+eps[i])HI]'''
        eps = round(epss[i],2)
        if not f_op == None:
            assert(len(f_op) == len(epss))
            assert(f_op[0] >= f_op[1] and epss[0] >= epss[1])
            op_num_tmp = f_op[i]
        else:
            op_num_tmp = (1+eps)*f_lam*cus_num*(mean_ot+1)/24 
        
        if var_op:  #increase op_num by 1 according to op_num_tmp decimal parts
            assert((not cycle_num == None) and (cycle_num >= 100))
            ops_num = [int(op_num_tmp)]*cycle_num
            part = int(100*(op_num_tmp - int(op_num_tmp)))
            if part > 0:
                for j in range(cycle_num//100+1):
                    temp = ops_num[j*100:min((j+1)*100, cycle_num)]
                    if len(temp) >= part:
                        temp[(100-part):] = [int(op_num_tmp)+1]*part
                        ops_num[j*100:min((j+1)*100, cycle_num)] = temp
            for prob in pause_probs:
                test_sets.append({'x0':x0, 'op_num':round(op_num_tmp,2), 'cus_th':total, 
                            'pause_prob':prob, 'ops_num':ops_num, 'cus_num': cus_num,
                            'fix_lam':f_lam, 'fix_rls':f_rls, 'poisson':poisson, 
                            'var_op':var_op, 'multi_run':runs, 'eps':eps, 'job_scale':f_lam*cus_num})

        else: 
            op_num = roundhalf(op_num_tmp)
            for prob in pause_probs:
                test_sets.append({'x0':x0, 'op_num':op_num, 'cus_th':total, 'pause_prob':prob, 
                            'cus_num': cus_num, 
                            'fix_lam':f_lam, 'fix_rls':f_rls, 'poisson':poisson, 
                            'var_op':var_op, 'multi_run':runs, 'eps':eps, 'job_scale':f_lam*cus_num, 'mean_ot':mean_ot})

    print('setup completed\n')
    return test_sets

def test_runs(x0 = 0, op_num = 1, cus_num = 1,
                cycle = 24, cycle_num = 100,
                pause_prob = 0, mean_ot = None, total = None,
                jobs_data = None, jobs_prob = None, sample_rm = False,
                f_lam = -1, f_rls = -1, runs = 10,
                poisson = False, var_op = False, 
                buffer_time = 0, count = False):
    '''testing one number of op., each test takes avg of 10 runs'''
    if f_lam > cycle and not poisson:
        print(f'Error, fix lambda > limit={cycle} for non-poisson')
        f_lam = cycle
    avg_qsz = []; avg_wt = []
    if count:
        avg_finished = []; avg_waiting = []
        avg_active = []; avg_paused = []
    if var_op:  #increase op_num by 1 according to op_num_tmp decimal parts
        assert((not cycle_num == None) and (cycle_num >= 100))
        ops_num = [int(op_num)]*cycle_num
        part = int(100*(op_num - int(op_num)))
        if part > 0:
            for j in range(cycle_num//100+1):
                temp = ops_num[j*100:min((j+1)*100, cycle_num)]
                if len(temp) >= part:
                    temp[(100-part):] = [int(op_num)+1]*part
                    ops_num[j*100:min((j+1)*100, cycle_num)] = temp
        op_num = ops_num[0]
    else:
        assert(isinstance(op_num,int))
        ops_num = None

    tmp_qsz = []; tmp_wt = []; operators = []
    if count:
        tmp_finished = []; tmp_waiting = []
        tmp_active = []; tmp_paused = []

    for i in range(runs): #10 runs and taking avg
        customers, operators, q_backlog, q_thread = initialize(x0, op_num, cus_num, mean_ot, total,
                                                  pause_prob, f_rls, 
                                                  jobs_data, jobs_prob, sample_rm,
                                                  show = False)

        if count:
            waiting_times, outstand, ptcl_fin, ptcl_come, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount = sim_run(customers, operators, q_backlog, q_thread,
                  cycle, cycle_num, pause_prob, f_lam, f_rls, 
                  jobs_data, jobs_prob, sample_rm, 
                  poisson, ops_num = ops_num, 
                  show = False, show_fin = False,count_sta = count)
        else:
            waiting_times, outstand, ptcl_fin, ptcl_come, backlog_sz, thread_sz = sim_run(customers, 
                  operators, q_backlog, q_thread,
                  cycle, cycle_num, pause_prob, f_lam, f_rls,
                  jobs_data, jobs_prob, sample_rm,
                  poisson, ops_num = ops_num, 
                  show = False, show_fin = False)
        tmp_qsz.append(np.mean([x for (x,y) in thread_sz if y >= buffer_time]))
        tmp_wt.append(np.mean([x for (x,y) in waiting_times if y >= buffer_time]))
        if count:
            tmp_finished.append(ptcl_fincount[-1][0])
            tmp_waiting.append([x for ((x,y,z),t) in ptcl_stacount if t >= buffer_time])
            tmp_active.append([y for ((x,y,z),t) in ptcl_stacount if t >= buffer_time])
            tmp_paused.append([z for ((x,y,z),t) in ptcl_stacount if t >= buffer_time])

        avg_qsz.append(np.mean(tmp_qsz))
        avg_wt.append(np.mean(tmp_wt))
        q_limit = q_thread.len
        if count:
            avg_finished.append(np.mean(tmp_finished))
            avg_waiting.append(np.mean(tmp_waiting))
            avg_active.append(np.mean(tmp_active))
            avg_paused.append(np.mean(tmp_paused))
    if count:
        return op_num, avg_qsz, q_limit, avg_wt, avg_finished, avg_waiting, avg_active, avg_paused
    else:
        return op_num, avg_qsz, q_limit, avg_wt
    

def single_run(x0 = 0, eps = 0, cus_num = 1, 
                cycle = 24, cycle_num = 100,
                pause_prob = 0, mean_ot = None, total = None,
                jobs_data = None, jobs_prob = None, sample_rm = False,
                f_lam = -1, f_rls = -1, 
                poisson = False, var_op = False, 
                count = False, f_op = None, wk_ratio = False):
    '''testing the data for a single simulation'''
    if f_lam > cycle and not poisson:
        print(f'Error, fix lambda > limit={cycle} for non-poisson')
        f_lam = cycle
    if not f_op == None:
        op_num = f_op
    else:
        op_num = (1+eps)*(mean_ot+1)*cus_num*f_lam/24
    print(f'op_num = {op_num}')
    
    if var_op:  #increase op_num by 1 according to op_num_tmp decimal parts
        assert((not cycle_num == None) and (cycle_num >= 100))
        ops_num = [int(op_num)]*cycle_num
        part = int(100*(op_num - int(op_num)))
        if part > 0:
            for j in range(cycle_num//100+1):
                temp = ops_num[j*100:min((j+1)*100, cycle_num)]
                if len(temp) >= part:
                    temp[(100-part):] = [int(op_num)+1]*part
                    ops_num[j*100:min((j+1)*100, cycle_num)] = temp
        op_num = ops_num[0]
    else:
        op_num = roundhalf(op_num)
        ops_num = None


    customers, operators, q_backlog, q_thread = initialize(x0, op_num, cus_num, mean_ot, total,
                                              pause_prob, f_rls, 
                                              jobs_data, jobs_prob, sample_rm,
                                              show = False)
    q_limit = q_thread.len

    if count:
        if wk_ratio:
            waiting_times, outstand, work_ratios, ptcl_fin, ptcl_num, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount = sim_run(customers, operators, q_backlog, q_thread,
              cycle, cycle_num, pause_prob, f_lam, f_rls, 
              jobs_data, jobs_prob, sample_rm, poisson, ops_num = ops_num, 
              show = False, show_fin = False, count_sta = count, wk_ratio = wk_ratio)
            return waiting_times, outstand, work_ratios, ptcl_fin, ptcl_num, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount, q_limit
        else:
            waiting_times, outstand, ptcl_fin, ptcl_num, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount = sim_run(customers, operators, q_backlog, q_thread,
              cycle, cycle_num, pause_prob, f_lam, f_rls, 
              jobs_data, jobs_prob, sample_rm, poisson, ops_num = ops_num, 
              show = False, show_fin = False, count_sta = count, wk_ratio = wk_ratio)
            return waiting_times, outstand, ptcl_fin, ptcl_num, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount, q_limit
    
    else:
        if wk_ratio:
            waiting_times, outstand, work_ratios, ptcl_fin, ptcl_num, backlog_sz, thread_sz = sim_run(customers, 
              operators, q_backlog, q_thread,
              cycle, cycle_num, pause_prob, f_lam, f_rls,
              jobs_data, jobs_prob, sample_rm, poisson, ops_num = ops_num, 
              show = False, show_fin = False, wk_ratio = wk_ratio)
            return waiting_times, outstand, work_ratios, ptcl_fin, ptcl_num, backlog_sz, thread_sz, q_limit
        else:
            waiting_times, outstand, ptcl_fin, ptcl_num, backlog_sz, thread_sz = sim_run(customers, 
              operators, q_backlog, q_thread,
              cycle, cycle_num, pause_prob, f_lam, f_rls,
              jobs_data, jobs_prob, sample_rm, poisson, ops_num = ops_num, 
              show = False, show_fin = False, wk_ratio = wk_ratio)
            return waiting_times, outstand, ptcl_fin, ptcl_num, backlog_sz, thread_sz, q_limit
    
    
def test_func(test_data, cycle = 24, cycle_num = 300,
              mean_ot = None, total = None,
              jobs_data = None, jobs_prob = None, sample_rm = False):
    '''A wrap-up helper function to do multiple runs with different parameters
    test_data = [test_param_1, test_param_2, ...]
    test_param_n = dict{x0, op_num, cus_th, cus_num, cus_lam, pause_prob, ot, multi_run}
    multi_run = # of runs for this set of parameter (take data avg. afterwards)
    '''
    count = 0
    for test_param in test_data:
        assert(isinstance(test_param,dict))
        x0 = test_param['x0']; op_num = test_param['op_num']
        cus_th = test_param['cus_th']; pause_prob = test_param['pause_prob']
        poisson = test_param['poisson']; var_op = test_param['var_op']
        cus_num = test_param['cus_num']
        if var_op:
            ops_num = test_param['ops_num']

        if 'fix_lam' in test_param:
            f_lam = test_param['fix_lam']
            if f_lam > cycle and not poisson:
                print(f'Error, fix lambda > limit={cycle} for non-poisson')
                f_lam = cycle
        else:
            f_lam = -1

        if 'fix_rls' in test_param:
            f_rls = test_param['fix_rls']
        else:
            f_rls = -1

        tmp_wt = []; tmp_bsz = []; tmp_qsz = []; tmp_outstand = []
        tmp_finished = []; tmp_waiting = []; tmp_active = []; tmp_paused = []
        tmp_ptcl_num = []
        t0 = time.time()
        for i in range(test_param['multi_run']):

            customers, operators, q_backlog, q_thread = initialize(x0, int(op_num), cus_num, mean_ot, total,
                                                      pause_prob, f_rls, 
                                                      jobs_data, jobs_prob, sample_rm,
                                                      show = False)
            if var_op:
                wt_times, outstand, ptcl_fin, ptcl_num, bg_sz, th_sz, ptcl_fin, ptcl_sta, op_sta = sim_run(customers, operators, q_backlog, q_thread,
                                               cycle, cycle_num, pause_prob, f_lam, f_rls, 
                                               jobs_data, jobs_prob, sample_rm,
                                               poisson, ops_num = ops_num, 
                                               show = False, show_fin = False,count_sta = True)
            else:
                wt_times, outstand, ptcl_fin, ptcl_num, bg_sz, th_sz, ptcl_fin, ptcl_sta, op_sta = sim_run(customers, operators, q_backlog, q_thread,
                                               cycle, cycle_num, pause_prob, f_lam, f_rls, 
                                               jobs_data, jobs_prob, sample_rm,
                                               poisson, 
                                               show = False, show_fin = False,count_sta = True)

            tmp_wt.append(wt_times); tmp_ptcl_num.append(ptcl_num)
            tmp_outstand.append(outstand)
            tmp_bsz.append(np.array([x for (x,y) in bg_sz]))
            tmp_qsz.append(np.array([x for (x,y) in th_sz]))
            tmp_finished.append(np.array([x for (x,y) in ptcl_fin]))
            tmp_waiting.append(np.array([x for ((x,y,z),t) in ptcl_sta]))
            tmp_active.append(np.array([y for ((x,y,z),t) in ptcl_sta]))
            tmp_paused.append(np.array([z for ((x,y,z),t) in ptcl_sta]))
        
        test_param['time'] = np.arange(-1,cycle*cycle_num,1)
        test_param['total_ptcls'] = np.mean(tmp_ptcl_num)
        test_param['bsz'] = np.mean(tmp_bsz,axis=0)
        test_param['qsz'] = np.mean(tmp_qsz,axis=0)
        test_param['qlim'] = q_thread.len
        test_param['finished'] = np.mean(tmp_finished,axis=0)
        test_param['waiting'] = np.mean(tmp_waiting,axis=0)
        test_param['active'] = np.mean(tmp_active,axis=0) 
        test_param['paused'] = np.mean(tmp_paused,axis=0)
        test_param['wt'] = tmp_wt
        test_param['outstand'] = tmp_outstand
        if mean_ot is not None:
            test_param['mean_ot'] = mean_ot
        t1 = time.time()
        count += 1
        print(f'##### {count}/{len(test_data)} finished #####')
        print(f'##### time used: {round(t1-t0,2)} sec #####')
    
    return test_data 

##### Auxiliary functions for level-curve tests #####

def get_mean(test_set, attr, buffer_time = 0, edit = False):
    '''computes 'attr' starting from buffer_time'''
    
    for test_param in test_set:
        test_param['bf_time'] = buffer_time
        
    if attr == 'avg_qsz':
        for test_param in test_set:
            
            if 'avg_qsz' in test_param and not edit:
                continue
            else:
                test_param[attr] = np.mean(test_param['qsz'][buffer_time:])
    
    elif attr == 'avg_qpc':
        for test_param in test_set:
            if 'avg_qpc' in test_param and not edit:
                continue
            else:
                if 'avg_qsz' in test_param:
                    assert(test_param['avg_qsz'] <= test_param['qlim'])
                    test_param[attr] = test_param['avg_qsz']/test_param['qlim']
                else:
                    test_param['avg_qsz'] = np.mean(test_param['qsz'][buffer_time:])
                    assert(test_param['avg_qsz'] <= test_param['qlim'])
                    test_param[attr] = test_param['avg_qsz']/test_param['qlim']
            
    elif attr == 'avg_wt':
        for test_param in test_set:
            if 'avg_wt' in test_param and not edit:
                continue
            else:
                tmp_avg = []
                for run in test_param['wt']:
                    tmp_avg.append(np.mean([x for (x,y) in run if y >= buffer_time]))
                test_param[attr] = np.mean(tmp_avg)

##### Plotting batches of tests #####

def plot_multi_QueuesTime(test_data, attr, title = None, buffer_time = 0, plot = True, save = (False,'')):
    '''plots queue(s) size v. time
    time = test_param['time'],
    queue_data = test_param['qsz']
    '''
    mns = []; data_info = []
    for test_param in test_data:
        if attr == 'job_scale':
            if test_param['fix_lam'] > 0:
                data_info.append([[test_param['time'][buffer_time:],
                                   test_param['qsz'][buffer_time:]],
                                  '-','{0}={1}'.format(attr,test_param[attr])])
            elif test_param['cus_lam'] > 0:
                data_info.append([[test_param['time'][buffer_time:],
                                   test_param['qsz'][buffer_time:]],
                         '-','{0}={1}'.format(attr,test_param['cus_num']*test_param['cus_lam'])])
        else:
            data_info.append([[test_param['time'][buffer_time:],
                               test_param['qsz'][buffer_time:]],
                              '-',f'{attr}={test_param[attr]}'])
        if plot:
            mn = round(np.mean(test_param['qsz'][buffer_time:]),2)
        else:
            mn = np.mean(test_param['qsz'][buffer_time:])
        mns.append(mn)
        data_info.append([[[test_param['time'][0],test_param['time'][-1]],[mn,mn]],':',f'mean={mn}'])
    if plot:
        text = title if not(title==None) else 'Queue size v. Time'
        plot_2D(data_info,
                title = text,
                input_label = 'Time(hour)',
                output_label = 'num. of ptcls',
                axis_bounds=None,xscale=None,yscale=None,save=save)
            
    else:
        return mns

def plot_multi_WaitTime(test_data, attr, title = None, buffer_time = 0, plot = True, save = (False,'')):
    '''plots waiting time v. time
    time = test_param['time'],
    waitingtime_data = test_param['wt']=[[data for run1], [data ofr run2], ...]
    '''
    mns = []; data_info = []
    for test_param in test_data:
        tmp_mn = []; tmp_wt = []; tmp_tm = []
        for run in test_param['wt']:
            tmp_mn.append(np.mean([x for (x,y) in run if y >= buffer_time]))
            tmp_wt += [x for (x,y) in run if y >= buffer_time]
            tmp_tm += [y for (x,y) in run if y >= buffer_time]
        if attr == 'job_scale':
            if test_param['fix_lam'] > 0:
                data_info.append([[tmp_tm,tmp_wt],'o','{0}={1}'.format(attr,test_param[attr])])
            elif test_param['cus_lam'] > 0:
                data_info.append([[tmp_tm,tmp_wt],'o','{0}={1}'.format(attr,test_param['cus_num']*test_param['cus_lam'])])
        else:
            data_info.append([[tmp_tm,tmp_wt],'o',f'{attr}={test_param[attr]}'])
        if plot:
            mn = round(np.mean(tmp_mn),2)
        else:
            mn = np.mean(tmp_mn)
        mns.append(mn)
        data_info.append([[[test_param['time'][0],test_param['time'][-1]],[mn,mn]],':',f'mean={mn}'])
    if plot:
        text = title if not(title==None) else 'Wait time v. Time'
        plot_2D(data_info,
                title = text,
                input_label = 'Time(hour)',
                output_label = 'wait time(hour)',
                axis_bounds=None,xscale=None,yscale=None,save=save)

    else:
        return mns

def plot_multi_OutstandTime(test_data, attr, title = None, buffer_time = 0, plot = True, save = (False,'')):
    '''plots outstanding time v. time
    time = test_param['time'],
    outstandingtime_data = test_param['outstand']=[[data for run1], [data ofr run2], ...]
    '''
    mns = []; data_info = []
    for test_param in test_data:
        tmp_mn = []; tmp_wt = []; tmp_tm = []
        for run in test_param['outstand']:
            tmp_mn.append(np.mean([x for (x,y) in run if y >= buffer_time]))
            tmp_wt += [x for (x,y) in run if y >= buffer_time]
            tmp_tm += [y for (x,y) in run if y >= buffer_time]
        if attr == 'job_scale':
            if test_param['fix_lam'] > 0:
                data_info.append([[tmp_tm,tmp_wt],'o','{0}={1}'.format(attr,test_param[attr])])
            elif test_param['cus_lam'] > 0:
                data_info.append([[tmp_tm,tmp_wt],'o','{0}={1}'.format(attr,test_param['cus_num']*test_param['cus_lam'])])
        else:
            data_info.append([[tmp_tm,tmp_wt],'o',f'{attr}={test_param[attr]}'])
        if plot:
            mn = round(np.mean(tmp_mn),2)
        else:
            mn = np.mean(tmp_mn)
        mns.append(mn)
        data_info.append([[[test_param['time'][0],test_param['time'][-1]],[mn,mn]],':',f'mean={mn}'])
    if plot:
        text = title if not(title==None) else 'Outstanding time v. Time'
        plot_2D(data_info,
                title = text,
                input_label = 'Time(hour)',
                output_label = 'outstanding(hour)',
                axis_bounds=None,xscale=None,yscale=None,save=save)

    else:
        return mns