##### Util functions #####
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt

def normal_dist(mean, sd, up = 10, low = 1):
    '''returns a normal distributed int value'''
    return max(low, min(round(np.random.normal(mean, sd)), up))

def roundhalf(x):
    '''round up if decimal part >= 0.5, round down o.w.'''
    n = np.ceil(x)
    if n-x > 0.5: return int(np.floor(x))
    else: return int(n)

#number of jobs arrive each week 
# def job_arrival(rate_arrival):
#   #rate_arrival=normal_dist(num_thread, sd)
#   return int(np.random.poisson(lam=rate_arrival))

def poisson_sim(lam, end):
    '''simulates a poisson process of arriving jobs'''
    assert(type(end)==int)
    num = []; time = []
    event = 0; event_time = 0
    while (event_time <= end):
        u = np.random.uniform(0,1) #get a uniform variable
        #interevent_time = np.ceil(-np.log(u) / (lam/end))  #inverse-CDF, inter-event time
        interevent_time = -np.log(u) / (lam/end)
        event += 1
        event_time += interevent_time
        if (event_time <= end):
            num.append(event) #add event 
            time.append(roundhalf(event_time)) #add event occuring time
    assert(len(num) == len(time))
    return (len(num), time)

def sort_ptcls(ptcls):
    '''sorts an array of protocols based on submit time'''
    ptclarr = np.array(ptcls)
    timearr = np.array([ptcl.time for ptcl in ptclarr])
    inds = timearr.argsort()
    sorted_ptcls = ptclarr[inds]
    res = []
    for ptcl in sorted_ptcls:
        res.append(ptcl)
    return res
    
def count_op(operators):
    '''counts num. of idle/busy operators'''
    idle = 0
    for op in operators:
        if op.sta == 0:
            idle += 1
    busy = len(operators) - idle
    assert((idle >= 0) and (busy >= 0))
    return (idle, busy)

def count_ptcl(onthread):
    '''counts num. of active/wait/paused/finished ptcl'''
    p_wait = 0
    p_active = 0
    p_paused = 0
    for ptcl in onthread.q:
        # 0 = waiting, 1 = processing, 2 = finished, 3 = paused
        if ptcl.sta == 0:
            p_wait += 1
        elif ptcl.sta == 1:
            p_active += 1
        elif ptcl.sta == 3:
            p_paused += 1

    return (p_wait, p_active, p_paused)

def plot_2D(data_info,title,input_label,output_label,
            axis_bounds=None,xscale=None,yscale=None,bloc='best',save=(False,'')):
    '''
    NOTES: Plots multiple 2D data on one graph.
    INPUT: 
        data_info = list of lists with structure:
            ith list = ith data information, as list
            ith list[0] = [input, output]
            ith list[1] = desired color for ith data
            ith list[2] = legend label for ith data
        title = string with desired title name
        input_label = string with name of input data
        output_label = string with name of output data
        axis_bounds = list with structure: [xmin, xmax, ymin, ymax]
        xscale = string with x axis scale description
        yscale = string with y axis scale description
    '''
    fig = plt.figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)

    for info_cache in data_info:
        if len(data_info) > 1:
            if 'o' in info_cache[1]:
#                 if len(info_cache[0][1]) > 0:
#                     mksize = max(20/int(len(info_cache[0][1])),7)
#                 else:
#                     mksize = 7
                mksize = 7
                alp = 0.5
            else:
                alp = 0.8
                mksize = 10
            plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=mksize, alpha = alp)
        else:
            plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=10)

    plt.title(title,fontsize=24)
    plt.xlabel(input_label,fontsize=20)
    plt.ylabel(output_label,fontsize=20)
    if axis_bounds is not None:
        plt.axis(axis_bounds)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend(loc=bloc,fontsize=14)
    plt.show()
    if save[0]:
        fig.savefig(save[1]+'_.png')

def plot_hist(data_info,title,input_label,output_label,axis_bounds=None,xscale=None,yscale=None):
    '''
    NOTES: Plots one histogram. (and the average line)
    '''
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)

    plt.hist(data_info[0], color='c', edgecolor='k', alpha=0.65)
    plt.axvline(data_info[1], color='k', linestyle='dashed', linewidth=2)

    plt.title(title,fontsize=24)
    plt.xlabel(input_label,fontsize=20)
    plt.ylabel(output_label,fontsize=20)
    if not axis_bounds == None:
        plt.axis(axis_bounds)
    if not xscale == None:
        plt.xscale(xscale)
    if not yscale == None:
        plt.yscale(yscale)
    plt.legend(loc='best',fontsize=14)
    plt.show()

def plot_2Dy2(data_info,data_info2,title,input_label,output_label,output_label2,
              axis_bounds=None,xscale=None,yscale=None):
    '''
    NOTES: Plots multiple 2D data on one graph.
    INPUT: 
        data_info = list of lists with structure:
            ith list = ith data information, as list
            ith list[0] = [input, output]
            ith list[1] = desired color for ith data
            ith list[2] = legend label for ith data
        data_info2 = data for second plot
        title = string with desired title name
        input_label = string with name of input data
        output_label = string with name of output data
        output_label2 = string with name of second output data
        axis_bounds = list with structure: [xmin, xmax, ymin, ymax]
        xscale = string with x axis scale description
        yscale = string with y axis scale description
    '''
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    style1 = []
    label1 = []
    mksize = max(int(20/len(data_info[0][0][0])),5)
    for info_cache in data_info:
        
        
        plt.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=mksize, alpha = 0.8)
        style1.append(info_cache[1])
        label1.append(info_cache[2])
        dummy = [info_cache[0][0][0],info_cache[0][1][0]]

    plt.title(title,fontsize=24)
    plt.xlabel(input_label,fontsize=20)
    plt.ylabel(output_label,fontsize=20)
    
    plt2 = plt.twinx()  # instantiate a second axes that shares the same x-axis
    
    plt2.set_ylabel(output_label2,fontsize=20)  # set secondary y axis
    for i in range(len(data_info)):
        data_info2.append([dummy,style1[i],label1[i]])
    mksize = max(int(20/len(data_info2[0][0][0])), 5)
    for info_cache in data_info2:
         
        plt2.plot(info_cache[0][0],info_cache[0][1],
                     info_cache[1],label=info_cache[2],
                     linewidth=3,markersize=mksize, alpha = 0.65)
    plt2.tick_params(axis='y')
    
    
    if axis_bounds is not None:
        plt.axis(axis_bounds)
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    plt.legend(loc='best',fontsize=14)
    plt.show()

##### Specific plotting functions #####

def plot_QueuesTime(backlog_data, onthread_data, plot = True):
    '''plots queue(s) size v. time
       backlog_data = an array of (backlog_size, time_point)
       onthread_data = an array of (onthread_size, time_point)'''
    szb = [x for (x,y) in backlog_data]; tmb = [y for (x,y) in backlog_data]
    szt = [x for (x,y) in onthread_data]; tmt = [y for (x,y) in onthread_data]
    print('average onthread queue size:',np.mean(szt))
    if plot:
        data_info = [ [[tmb, szb],'k-', 'backlog'], [[tmt, szt],'c-', 'onthread'] ]
        plot_2D(data_info,
                title = 'Queue size v. Time',
                input_label = 'Time(hour)',
                output_label = 'num. of ptcls',
                axis_bounds=None,xscale=None,yscale=None)

def plot_QueueWait(onthread_data, waiting_times, plot = True, code = ''):
    '''plots onthread size & waiting time v. Time
       onthread_data = an array of (onthread_size, time_point)
       waiting_times = an array of (waiting_time, finish_time)'''
    wt = [x for (x,y) in waiting_times]; wty = [y for (x,y) in waiting_times]
    avg = np.mean(wt)
    if len(code) > 0:
        title = f'Onthread Size v. {code} Time'
        print(f'average {code} time: {avg} hours')
    else:
        title = f'Onthread Size v. Waiting Time'
        print(f'average waiting time: {avg} hours')
    if plot:
        szt = [x for (x,y) in onthread_data]; tmt = [y for (x,y) in onthread_data]
        wt_tm = [ [[wty, wt],'ko','waiting time']]
        th_tm = [ [[tmt, szt],'c-', 'onthread size'] ]
        plot_2Dy2(wt_tm, th_tm,
                  title = title,
                  input_label = 'Time(hour)',
                  output_label = 'hours',
                  output_label2 = 'num. of ptcls')
        
def plot_WaitOutstand(waiting_times, outstand_times, 
                      plot = True, mean = False, onthread_data = None):
    '''plots (waiting - outstand) time v. Time
       waiting_times = an array of (waiting_time, finish_time)
       outstand_times = an array of (waiting_time, start_time)'''
    wt = np.array([x for (x,y) in waiting_times]); wty = [y for (x,y) in waiting_times]
    outst = np.array([x for (x,y) in outstand_times])
    outsty = np.array([y for (x,y) in outstand_times])
    if mean:
        assert(onthread_data is not None)
        szt = [x for (x,y) in onthread_data]
        szt = np.array([x if x > 0 else -1 for x in szt])
        outst = outst/szt[1:]
        outst = np.array([x if x > 0 else 0 for x in outst])
    avg = np.mean(outst)-np.mean(wt)
    print('avg.outstand- avg.wait difference: {} hours'.format(avg))
    if plot:
        wt_tm = [[[wty, wt],'ko','waiting time']]
        if mean:
            text = 'mean Outstanding'
        else:
            text = 'Outstanding'
        ou_tm = [[[outsty, outst],'ro', text]]
        plot_2Dy2(wt_tm, ou_tm,
                  title = f'Waiting Time & {text} Time',
                  input_label = 'Time(hour)',
                  output_label = 'wait hours', output_label2 = 'outstand hours')
    
def plot_WaitDist(waiting_times, plot = True):
    '''plots waiting time
       waiting_times = an array of (waiting_time, time_point)'''
    wt = [x for (x,y) in waiting_times]
    avg = np.mean(wt)
    print('average waiting time: {} hours'.format(avg))
    if plot:
        data_info = [wt, avg]
        plot_hist(data_info,
                  title = 'Waiting Time Distribution',
                  input_label = 'hours',
                  output_label = 'num. of ptcls',)

def plot_Opsta(op_stacount):
    '''plots operator status w.r.t. time'''
    opt_i = [x for ((x,y),t) in op_stacount]; opt_t = [t for ((x,y),t) in op_stacount]
    opt_b = [y for ((x,y),t) in op_stacount]
    data_info = [ [[opt_t, opt_b],'k-', 'busy op'], [[opt_t, opt_i],'c-', 'idle op'] ]
    plot_2D(data_info,
            title = 'Operator Status v. Time',
            input_label = 'Time(hour)',
            output_label = 'num. of op',
            axis_bounds=None,xscale=None,yscale=None)

def plot_Ptclsta(ptcl_fincount, ptcl_stacount, limit = 0, 
                 onthread_data = None, plot = True):
    '''plots ptcl status w.r.t. time'''
    ptcl_f = [x for (x,t) in ptcl_fincount]; ptcl_t = [t for (x,t) in ptcl_fincount]
    
    #check the number of ptcls finished at each time step
    ptcl_tmp_f = np.array(ptcl_f[1:]) - np.array(ptcl_f[:-1])
    ptcl_tmp_f = np.r_[[ptcl_f[0]],ptcl_tmp_f] #finished ptcl at each time
    
    ptcl_w = [x for ((x,y,z),t) in ptcl_stacount]
    ptcl_a = [y for ((x,y,z),t) in ptcl_stacount]
    ptcl_p = [z for ((x,y,z),t) in ptcl_stacount]
    data_info = [ [[ptcl_t, ptcl_w],'r-', 'waiting'], 
                  [[ptcl_t, ptcl_a],'y-', 'active'], [[ptcl_t, ptcl_p],'k-', 'paused']] 
    data_2 = [ [[ptcl_t, ptcl_f],'g-', 'finished'] ]
    if plot:
        plot_2Dy2(data_info, data_2,
                title = 'PTCL Status v. Time',
                input_label = 'Time(hour)',
                output_label = 'num. of ptcls', output_label2 = 'finished ptcls',
                axis_bounds=None,xscale=None,yscale=None)
        if not (onthread_data==None):
            szt = np.array([x for (x,y) in onthread_data[1:]])
            ptcl_sz = np.array(ptcl_w)+np.array(ptcl_a)+np.array(ptcl_p)
            assert(((szt-ptcl_sz)==0).all())
            data_info_x = [ [[ptcl_t, szt],
                             'c-', 'sum'],
                            [[ptcl_t, limit*np.ones(len(ptcl_t))],'k:', 'limit']]
            plot_2D(data_info_x,
                    title = 'Onthread Size v. Queue Limit',
                    input_label = 'Time(hour)',
                    output_label = 'num. of ptcls',
                    axis_bounds=None,xscale=None,yscale=None)
        else:
            print("Error: second plot needs onthread_data")
    