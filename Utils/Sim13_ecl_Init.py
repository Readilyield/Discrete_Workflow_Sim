##### Initialization and single-runs #####
import numpy as np
from Sim13_Classes import customer, operator, protocol, backlog, onthread
from Sim13_ecl_Util import sample_job
from Sim1_Util import count_op, count_ptcl, sort_ptcls
from Sim1_Util import poisson_sim

def initialize(x0 = 0, op_num = 0, cus_num = 1, 
               mean_ot = None, total = None, pause_prob = 0, f_rls = -1,
               jobs_data = None, jobs_prob = None, sample_rm = False,
               show = True):
    '''sample_rm: whether to remove sampled job'''

    '''input: hyperparameters'''
    if (isinstance(jobs_data,dict)):
        keys = list(jobs_data.keys())
        key_prob = [jobs_prob[key] for key in keys]
        assert(len(keys)==len(key_prob))
    '''initial protocl array at t = 0'''
    cus_ind = 0
    init_cus = customer(th = x0, ind = cus_ind)
    init_ptcls = []
    for i in range(x0):
        if not (keys == None):
            f_ot = sample_job(jobs_data, keys, key_prob, sample_rm)
        else:
            f_ot = sample_job(jobs_data, sample_rm)
        init_ptcls.append(protocol(pause = pause_prob, 
                          rls_time = f_rls, force_ot = f_ot, own = init_cus))
    '''initial customers'''
    customers = []
    for i in range(cus_num):
        customers.append(customer(th = total, ind = 233+cus_ind))
        cus_ind += 1
    # customers = [customer(th = total, ind = 233)]

    '''initial queues and operators'''
    q_backlog = backlog()
    for ptcl in init_ptcls:
        q_backlog.add(ptcl)
    q_backlog.sort()
    q_thread = onthread(customers, q_backlog)
    q_thread.len += init_cus.th
    q_thread.fill()
    q_thread.len -= init_cus.th
    operators = []
    for i in range(op_num):
        operators.append(operator())
    if show:
        print('######################')
        print('initilization finished')
        print('num. of customers:',len(customers))
        print('ptcls on thread:',q_thread.curlen)
        print('ptcls backlogged:',q_backlog.curlen)
        c0 = count_op(operators)[0]; c1 = count_op(operators)[1]
        print(f'operators idle: {c0}, busy: {c1}')
        print()
    
    return customers, operators, q_backlog, q_thread

def sim_run(customers, operators, q_backlog, q_thread,
            cycle = 24, cycle_num = 10, 
            pause_prob = 0, f_lam = -1, f_rls = -1, #deterministic param
            jobs_data = None, jobs_prob = None, sample_rm = False,
            poisson = False, ops_num = None,
            show = True, show_fin = True, 
            count_sta = False, wk_ratio = False, debug = False):
    '''hyper parameters'''
    if (isinstance(jobs_data,dict)):
        keys = list(jobs_data.keys())
        key_prob = [jobs_prob[key] for key in keys]
        assert(len(keys)==len(key_prob))
    else:
        keys = None; key_prob = None
    if not (ops_num == None):
        assert(len(ops_num)==cycle_num)

    '''data containers'''
    waiting_times = []
    ptcl_fin = 0
    ptcl_num = 0
    ptcl_come = []
    backlog_sz = [(q_backlog.curlen, -1)]
    thread_sz = [(q_thread.curlen, -1)]
    #add lists that record different status jobs and operators
    op_stacount = []
    ptcl_stacount = []
    ptcl_fincount = []
    outstand_times = []
    if wk_ratio:
        wk_ratios = [0]
    if show:
        print('##### begins simulation #####')
    
    '''begins simulation: t unit = hour'''
    for cyc in range(cycle_num):
        ptcl_fin_old = ptcl_fin
        if show:
            print(f'\n##### begins cycle {cyc} #####')
        if not (ops_num == None):
            '''add/remove operators accordingly'''
            idle = -1
            num_op = ops_num[cyc]
            while len(operators) < num_op:
                operators.append(operator())
            if len(operators) > num_op:
                (idle, busy) = count_op(operators)
                while (len(operators) > num_op) and (idle > 0):
                    for op in operators:
                        if len(operators) == num_op:
                            break
                        if op.sta == 0:
                            operators.remove(op)
                            idle -= 1

            assert((idle == 0) or len(operators) == num_op)

        for t in range(cycle):
            ptcl_fin_tmp = ptcl_fin
            t_r = t+cyc*cycle

            '''customers submit ptcls to backlog for this cycle'''
            if t == 0: #a new cycle
                
                new_ptcls = []
                for cus in customers:
                    ptcls = cus.get_sample_ptcls(jobs_data, keys, key_prob, 
                    sample_rm, poisson, cycle, t_r, pause_prob, f_lam, f_rls)

                    new_ptcls += ptcls
                new_ptcls = sort_ptcls(new_ptcls)
                if wk_ratio:
                    hours_in = 0; hours_out = len(operators)*24
                    for ptcl in new_ptcls:
                        if ptcl.time <= 24+t_r:
                            hours_in += ptcl.ot
                    if len(wk_ratios) > 1:
                        wk_ratios.append((hours_in/hours_out,t_r))
                    else:
                        wk_ratios[0] = (hours_in/hours_out,t_r)
                    
                ptcl_come += new_ptcls
                ptcl_num += len(new_ptcls)
            
            if (t > 0) and (wk_ratio):
                wk_ratios.append(wk_ratios[-1])
            ### record (mean) outstanding time ###
            outstanding_tmp = 0
            for ptcl in q_thread.q:
                outstanding_tmp += ptcl.ot

            outstand_times.append((outstanding_tmp, t_r))
            ### record (mean) outstanding time ###    
            
            while (len(ptcl_come) > 0) and (ptcl_come[0].time <= t_r):
                q_backlog.add(ptcl_come.pop(0))
            q_backlog.sort()
            '''fill available ptcls to onthread'''
            q_thread.fill(t_r)
            '''operators grab ptcls from onthread
               or process onging ptcls'''
            for op in operators:
                if (op.sta == 0) and (q_thread.curlen > 0): #idle operator
                    op.get(q_thread, t_r) #grabs a new ptcl
                    if debug:
                        print(f'%%% grab job {op.jobname} at t= {t_r}')

                if op.sta == 1: #busy operator
                    if t_r > 0: #no progress when t = 0
                        op.job.go(t_r)
                    if op.job.sta == 2: #ptcl finished
                        if debug:
                            print(f'%%% finish job {op.jobname} at t= {t_r}')
                        ptcl_fin += 1
                        waiting_times.append((op.job.wt, t_r))
                        op.drop(q_thread, t_r) #disposes the finished ptcl
                    elif op.job.sta == 3: #ptcl paused
                        if debug:
                            print(f'%%% pause job {op.jobname} at t= {t_r}')
                            print(f'%%% job {op.jobname} will wait at t= {t_r+op.job.rt}')
                        op.drop(q_thread, t_r) #drops the paused ptcl
                        op.get(q_thread, t_r) #grabs a new ptcl
                        if debug:
                            print(f'%%% grab job {op.jobname} at t= {t_r}')
#                         if op.sta == 1: #successfully grabbed a new job
#                             op.job.go(t_r)
                
            q_thread.wait() #remaining ptcls onthread wait time ++
            #if backlog_sz[-1][0] != q_backlog.curlen:
            backlog_sz.append((q_backlog.curlen, t_r))
            #if thread_sz[-1][0] != q_thread.curlen:
            thread_sz.append((q_thread.curlen, t_r))
            #add number of finished/active/paused/waiting jobs at time t
            if count_sta:
                ptcl_fincount.append((ptcl_fin, t_r))
                op_stacount.append((count_op(operators), t_r))
                ptcl_stacount.append((count_ptcl(q_thread), t_r))
                #print([x.sta for x in q_thread.q])

        if show or show_fin:        
            print(f'### finshed {ptcl_fin-ptcl_fin_old} ptcls in cycle {cyc} ###')
            print(f'### remaining {q_thread.curlen} ptcls in cycle {cyc+1} ###')
    if show:
        print('\n######################')
        print('simulation completed!')
        print(f'finshed {ptcl_fin} ptcls in {cycle_num}x{cycle} hours')
        print(f'incoming {ptcl_num} ptcls')
        print(f'thread limit = {q_thread.len}')
    if count_sta:
        if wk_ratio:
            return waiting_times, outstand_times, wk_ratios, ptcl_fin, ptcl_num, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount
        else:
            return waiting_times, outstand_times, ptcl_fin, ptcl_num, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount
    else:
        if wk_ratio:
            return waiting_times, outstand_times, wk_ratios, ptcl_fin, ptcl_num, backlog_sz, thread_sz
        else:
            return waiting_times, outstand_times, ptcl_fin, ptcl_num, backlog_sz, thread_sz

        