##### Initialization and single-runs #####
import numpy as np
from Sim13_Classes import customer, operator, protocol, backlog, onthread
from Sim1_Util import count_op, count_ptcl, sort_ptcls

def initialize(x0 = 0, op_num = 1, cus_th = 10, cus_num = 1, 
               pause_prob = 0, ot = -1, f_ot = -1, f_rls = -1,
               show = True):

    '''input: hyperparameters'''

    '''initial protocl array at t = 0'''
    cus_ind = 0
    init_cus = customer(th = x0, ind = cus_ind)
    init_ptcls = []
    cus_ind += 1
    for i in range(x0):
        init_ptcls.append(protocol(op_time = ot, pause = pause_prob, 
                          rls_time = f_rls, force_ot = f_ot, own = init_cus))
    '''initial customers'''
    customers = []
    for i in range(cus_num):
        customers.append(customer(th = cus_th, ind = cus_ind))
        cus_ind += 1

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
            cus_lam = 5, cycle = 24, cycle_num = 100, 
            pause_prob = 0, ot = -1, 
            f_ot = -1, f_lam = -1, f_rls = -1, #deterministic param
            poisson = False, ops_num = None,
            show = True, show_fin = True, count_sta = False, debug = False):
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
                    new_ptcls += cus.get_ptcls(cus_lam,cycle,t_r,pause_prob,ot, poisson,
                                               f_ot, f_lam, f_rls)
                new_ptcls = sort_ptcls(new_ptcls)
                ptcl_come += new_ptcls
                ptcl_num += len(new_ptcls)
                
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
                    # if op.sta == 1:
                    #     waiting_times.append((op.job.wt, t_r)) #ends waiting
                if op.sta == 1: #busy operator
                    if t_r > 0: #no progress when t = 0
                        op.job.go(t_r)

                    if op.job.sta == 2: #ptcl finished
                        if debug:
                            print(f'%%% finish job {op.jobname} at t= {t_r}')
                            print(f'%%% job waiting time {op.job.wt} hours at t= {t_r}')
                        ptcl_fin += 1
                        waiting_times.append((op.job.wt, t_r))
                        op.drop(q_thread, t_r) #disposes the finished ptcl
                    elif op.job.sta == 3: #ptcl paused
                        if debug:
                            print(f'%%% pause job {op.jobname} at t= {t_r}')
                            print(f'%%% job {op.jobname} will wait at t= {t_r+op.job.rt}')
                            #print([job.sta for job in q_thread.q])
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
        return waiting_times, outstand_times, ptcl_fin, ptcl_num, backlog_sz, thread_sz, ptcl_fincount, ptcl_stacount, op_stacount
    else:
        return waiting_times, outstand_times, ptcl_fin, ptcl_num, backlog_sz, thread_sz

        