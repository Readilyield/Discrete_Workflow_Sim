##### Running batches of tests #####
import numpy as np
from Sim13_Init import initialize, sim_run
from Sim13_Classes import operator
from Sim1_Util import plot_2D
import time
import copy

def setup(cus_nums, epss, x0 = 0, ot = 30, cus_th = 10, cus_lam = 1,
          f_ot = -1, f_lam = -1, f_rls = -1, runs = 10,
          poisson = False, var_op = False, cycle_num = None):
    assert(isinstance(cus_nums, np.ndarray))
    assert(isinstance(epss, np.ndarray))
    '''test_sets = [test_param_0(P=0), test_param_0(P=1), 
                    test_param_1(P=0), test_param_1(P=1),
                    ...]'''
    test_sets = []
    if f_lam > 24 and not poisson:
        print(f'Error, fix lambda > limit={cycle} for non-poisson')
        f_lam = 24
    for j in range(len(epss)):
        '''op_num = [HO = (1+eps[j])HI]'''
        eps = round(epss[j],2)
        for i in range(len(cus_nums)):
            if f_ot == -1:
                op_num_tmp = (1+eps)*max(cus_lam,f_lam)*cus_nums[i]*(ot+1)/24
            elif f_ot > 0: 
                op_num_tmp = (1+eps)*max(cus_lam,f_lam)*cus_nums[i]*(f_ot+1)/24
            
            if var_op:  #increase op_num by 1 according to op_num_tmp decimal parts
                assert((not cycle_num == None) and (cycle_num >= 100))
                ops_num = [int(op_num_tmp)]*cycle_num
                test = copy.copy(ops_num)
                part = int(100*(op_num_tmp - int(op_num_tmp)))
                if part > 0:
                    
                    for k in range(cycle_num//100+1):
                        temp = ops_num[k*100:min((k+1)*100, cycle_num)]
                        if len(temp) >= part:
                           
                            temp[(100-part):] = [int(op_num_tmp)+1]*part
                            ops_num[k*100:min((k+1)*100, cycle_num)] = temp
                test_sets.append({'x0':x0, 'op_num':round(op_num_tmp,2), 'cus_th':cus_th,
                                'cus_num':cus_nums[i], 'cus_lam':cus_lam, 
                                'pause_prob':0, 'ops_num':ops_num, 'ot':ot, 
                                'fix_lam':f_lam, 'fix_ot':f_ot, 'fix_rls':f_rls, 
                                'poisson':poisson, 'var_op':var_op, 'multi_run':runs, 
                                'eps':eps, 'job_scale':max(f_lam,cus_lam)*cus_num})
                test_sets.append({'x0':x0, 'op_num':round(op_num_tmp,2), 'cus_th':cus_th,
                                'cus_num':cus_nums[i], 'cus_lam':cus_lam,
                                'pause_prob':1, 'ops_num':ops_num, 'ot':ot, 
                                'fix_lam':f_lam, 'fix_ot':f_ot, 'fix_rls':f_rls, 
                                'poisson':poisson, 'var_op':var_op, 'multi_run':runs, 
                                'eps':eps, 'job_scale':max(f_lam,cus_lam)*cus_num})
            else:
                op_num = roundhalf(op_num_tmp)
                test_sets.append({'x0':x0, 'op_num':op_num, 'cus_th':cus_th, 
                                'cus_num':cus_nums[i], 'cus_lam':cus_lam,
                                'pause_prob':0, 'ot':ot, 
                                'fix_lam':f_lam, 'fix_ot':f_ot, 'fix_rls':f_rls, 
                                'poisson':poisson, 'var_op':var_op, 'multi_run':runs, 
                                'eps':eps, 'job_scale':max(f_lam,cus_lam)*cus_num})
                test_sets.append({'x0':x0, 'op_num':op_num, 'cus_th':cus_th, 
                                'cus_num':cus_nums[i], 'cus_lam':cus_lam, 
                                'pause_prob':1, 'ot':ot,  
                                'fix_lam':f_lam, 'fix_ot':f_ot, 'fix_rls':f_rls, 
                                'poisson':poisson, 'var_op':var_op, 'multi_run':runs, 
                                'eps':eps, 'job_scale':max(f_lam,cus_lam)*cus_num})

    print('setup completed\n')
    return test_sets


def test_runs(x0 = 10, op_nums = [], cus_th = 5, 
                cus_num = 10, cus_lam = 5, 
                cycle = 24, cycle_num = 8,
                pause_prob = 0, ot = -1, 
                f_ot = -1, f_lam = -1, f_rls = -1, runs = 10,
                buffer_time = 0, count = False):
    '''testing different numbers of op., each test takes avg of 10 runs'''
    if f_lam > cycle and not poisson:
        print(f'Error, fix lambda > limit={cycle} for non-poisson')
        f_lam = cycle
    avg_qsz = []; avg_wt = []
    if count:
        avg_finished = []; avg_waiting = []
        avg_active = []; avg_paused = []
    for op_num in op_nums:
        customers, operators, q_backlog, q_thread = initialize(x0, 0, 
                                                      cus_th, cus_num, pause_prob, ot, 
                                                      f_ot, f_rls, show = False)
        tmp_qsz = []; tmp_wt = []; operators = []
        if count:
            tmp_finished = []; tmp_waiting = []
            tmp_active = []; tmp_paused = []

        for i in range(op_num):
                operators.append(operator())
        for i in range(runs): #10 runs and taking avg
            if count:
                waiting_times, outstand, ptcl_fin, ptcl_come, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount = sim_run(customers, operators, q_backlog, q_thread,
                      cus_lam, cycle, cycle_num,
                      pause_prob, ot, f_ot, f_lam, f_rls, 
                      show = False, show_fin = False,count_sta = count)
            else:
                waiting_times, outstand, ptcl_fin, ptcl_come, backlog_sz, thread_s = sim_run(customers, 
                      operators, q_backlog, q_thread,
                      cus_lam, cycle, cycle_num,
                      pause_prob, ot, f_ot, f_lam, f_rls, 
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
        return op_nums, avg_qsz, q_limit, avg_wt, avg_finished, avg_waiting, avg_active, avg_paused
    else:
        return op_nums, avg_qsz, q_limit, avg_wt
    

def test_func(test_data, cycle = 24, cycle_num = 300):
    '''A wrap-up helper function to do multiple runs with different parameters
    test_data = [test_param_1, test_param_2, ...]
    test_param_n = dict{x0, op_num, cus_th, cus_num, cus_lam, pause_prob, ot, multi_run}
    multi_run = # of runs for this set of parameter (take data avg. afterwards)
    '''
    count = 0
    for test_param in test_data:
        assert(isinstance(test_param,dict))
        x0 = test_param['x0']; op_num = test_param['op_num']
        cus_th = test_param['cus_th']; cus_num = test_param['cus_num']
        cus_lam = test_param['cus_lam']; pause_prob = test_param['pause_prob']; 
        poisson = test_param['poisson']; var_op = test_param['var_op']
        ot = test_param['ot']
        if var_op:
            ops_num = test_param['ops_num']
        
        if 'fix_lam' in test_param:
            f_lam = test_param['fix_lam']
            if f_lam > cycle and not poisson:
                print(f'Error, fix lambda > limit={cycle} for non-poisson')
                f_lam = cycle
        else:
            f_lam = -1
        if 'fix_ot' in test_param:
            f_ot = test_param['fix_ot']
        else:
            f_ot = -1
        if 'fix_rls' in test_param:
            f_rls = test_param['fix_rls']
        else:
            f_rls = -1

        tmp_wt = []; tmp_bsz = []; tmp_qsz = []
        tmp_finished = []; tmp_waiting = []; tmp_active = []; tmp_paused = []
        tmp_ptcl_num = []
        t0 = time.time()
        for i in range(test_param['multi_run']):
            customers, operators, q_backlog, q_thread = initialize(x0, int(op_num), 
                                                   cus_th, cus_num, pause_prob, ot, 
                                                   f_ot, f_rls, show = False)
            if var_op:
                wt_times, outstand, ptcl_fin, ptcl_num, bg_sz, th_sz, ptcl_fin, ptcl_sta, op_sta = sim_run(customers, operators, q_backlog, q_thread,
                                                   cus_lam, cycle, cycle_num, pause_prob, ot,
                                                   f_ot, f_lam, f_rls, 
                                                   poisson, ops_num = ops_num,
                                                   show = False, show_fin = False, count_sta=True)
            else:
                wt_times, outstand, ptcl_fin, ptcl_num, bg_sz, th_sz, ptcl_fin, ptcl_sta, op_sta = sim_run(customers, operators, q_backlog, q_thread,
                                                   cus_lam, cycle, cycle_num, pause_prob, ot,
                                                   f_ot, f_lam, f_rls, 
                                                   poisson, 
                                                   show = False, show_fin = False, count_sta=True)
                
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
        t1 = time.time()
        count += 1
        print(f'##### {count}/{len(test_data)} finished #####')
        print(f'##### time used: {round(t1-t0,2)} sec #####')
    
    return test_data 

##### Auxiliary functions for level-curve tests #####

def roundhalf(x):
    '''round up if decimal part >= 0.5, round down o.w.'''
    n = np.ceil(x)
    if n-x > 0.5: return int(np.floor(x))
    else: return int(n)

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

def plot_multi_QueuesTime(test_data, attr, title = None, buffer_time = 0, plot = True):
    '''plots queue(s) size v. time
    time = test_param['time'],
    queue_data = test_param['qsz']
    '''
    mns = []; data_info = []
    for test_param in test_data:
        if attr == 'job_scale':
            if test_param['fix_lam'] > 0:
                data_info.append([[test_param['time'],test_param['qsz']],'-','{0}={1}'.format(attr,test_param['cus_num']*test_param['cus_lam'])])
            elif test_param['cus_lam'] > 0:
                data_info.append([[test_param['time'],test_param['qsz']],'-','{0}={1}'.format(attr,test_param['cus_num']*test_param['cus_lam'])])
        else:
            data_info.append([[test_param['time'],test_param['qsz']],'-',f'{attr}={test_param[attr]}'])
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
                axis_bounds=None,xscale=None,yscale=None)
    else:
        return mns

def plot_multi_WaitTime(test_data, attr, title = None, buffer_time = 0, plot = True):
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
                data_info.append([[tmp_tm,tmp_wt],'o','{0}={1}'.format(attr,test_param['cus_num']*test_param['cus_lam'])])
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
                axis_bounds=None,xscale=None,yscale=None)
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
        text = title if not(title==None) else 'Wait time v. Time'
        plot_2D(data_info,
                title = text,
                input_label = 'Time(hour)',
                output_label = 'wait time(hour)',
                axis_bounds=None,xscale=None,yscale=None,save=save)

    else:
        return mns