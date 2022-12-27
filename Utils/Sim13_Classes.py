##### Classes #####
import string
import random
import numpy as np
letters = string.ascii_letters
from Sim13_ecl_Util import sample_job
from Sim1_Util import normal_dist, poisson_sim, sort_ptcls

class customer:
    '''generates a random customer for ECL'''
    def __init__(self, mean = 5, sd = 1, th = 0, ind = 0):
        #num. of thread purchased: normal distribution
        #EDITED here
        self.sd = sd
        self.name = 'Cus_'+str(ind)
        if th == 0:
            self.th = normal_dist(mean, sd, up=10) 
        else:
            self.th = th
        self.count = 0 #counts num. of ptcls on Onthread

    def out(self):
        print()
        print('name: ',self.name)
        print('threads: ',self.th)
        print('type: ',self.tp)
    
    def __eq__(self, other = None):
        if other != None:
            return self.name == other.name
        else:
            return 0

    def get_ptcls(self, lam = 1, timespan = 24, curtime = 0, 
                  pause_prob = 0, ot = -1, poisson = False,
                  force_ot = -1, force_lam = -1, force_rls = -1):
        '''returns (ptcls, time) array in <timespan>'''
        if lam == 0:
            lam = normal_dist(self.th/2, self.sd, up=10)
        if poisson:
            if force_lam == -1:
                (num, time) = poisson_sim(lam, timespan)
            elif force_lam >= 0:
                (num, time) = poisson_sim(force_lam, timespan)
        else:
            if force_lam == -1:
                if lam > timespan: #lam exceeds period limit
                    lam = timespan
                interjob_time = int(timespan/lam)
                num = lam
                time = [i*interjob_time for i in range(lam)]
            elif force_lam >= 0: #fixed amount of arrival jobs
                if force_lam > timespan: #lam exceeds period limit
                    force_lam = timespan
                interjob_time = int(timespan/force_lam)
                num = force_lam
                time = [i*interjob_time for i in range(force_lam)]
            assert(time[-1]<timespan)
        self.ptcls = []
        for i in range(num): #get num. of threads to use for each job
            self.ptcls.append(protocol(op_time = ot, pause = pause_prob, 
                                       rls_time = force_rls, force_ot = force_ot,
                                       own = self, time = time[i]+curtime))
        assert(len(self.ptcls) == num)
        if force_lam >= 0 and not poisson:
            assert(len(self.ptcls) == force_lam)
        return self.ptcls

    def get_sample_ptcls(self, jobs_data, keys = None, key_prob = None, 
                  sample_rm = False, poisson = False,
                  timespan = 24, curtime = 0, 
                  pause_prob = 0, force_lam = -1, force_rls = -1):
        '''returns (ptcls, time) array in <timespan> sampled from ecl/random data'''
        assert(force_lam > 0)
        self.ptcls = []
        if poisson:
            (num, time) = poisson_sim(force_lam, timespan)
            for i in range(num): 
                if not (keys == None):
                    f_ot = sample_job(jobs_data, keys, key_prob, sample_rm)
                else:
                    f_ot = sample_job(jobs_data, sample_rm)
                self.ptcls.append(protocol(pause = pause_prob, rls_time = force_rls, 
                                force_ot = f_ot, own = self, time = time[i]+curtime))
        else:
            interjob_time = int(timespan/force_lam)
            
            time = [i*interjob_time for i in range(force_lam)]
            assert(time[-1]<timespan)

            for i in range(force_lam):
                if not (keys == None):
                    f_ot = sample_job(jobs_data, keys, key_prob, sample_rm)
                else:
                    f_ot = sample_job(jobs_data, sample_rm)
                self.ptcls.append(protocol(pause = pause_prob, rls_time = force_rls, 
                                force_ot = f_ot, own = self, time = time[i]+curtime))
            assert(len(self.ptcls) == force_lam)
        
        return self.ptcls


class protocol:
    '''a protocol for ECL'''
    '''EDIT: pause function'''
    def __init__(self, op_time = -1, pause = 0,
                 rls_time = -1, force_ot = -1, own = None, time = 0):
        self.time = time
 
        self.sta = 0 # 0 = waiting, 1 = processing, 2 = finished, 3 = paused
        if own != None:
            self.own = own
            
        if force_ot == -1:
            if op_time == -1:
                self.ot = normal_dist(30, 15, up=48, low=12)
            else:
                self.ot = normal_dist(op_time, op_time/2, up=100, low=1)
        elif force_ot >= 0:
            self.ot = force_ot #operation time-count down
        if rls_time == -1:
            if op_time == -1:
                self.rt = normal_dist(15, 7.5, up=48, low=1)
            else:
                self.rt = normal_dist(op_time/2, op_time/4, up=48, low=1)
        elif rls_time >= 0:
            self.rt = rls_time #release time-count down
        
        if pause > 0:
            assert (pause <= 1)
            tmp = np.random.uniform(0,1)
            if tmp <= pause: 
                self.pause = True #will pause at some time
                #when operating time reduces to stop time -> pauses
                if rls_time >= 0:
                    if force_ot >= 0:
                        self.pt = int(force_ot/2)
                    else:
                        self.pt = int(self.ot/2)
                else:
                    self.pt = self.ot - normal_dist(self.ot/2, self.ot/4, 
                                                up=self.ot-1, low=1)
            else:
                self.pause = False #will not pause
        else:
                self.pause = False #will not pause
        self.wt = 0 #waiting time-count up
        self.begin = 0 #record the beginning time 
        self.complt = 0 #record the ending time
        nm = random.choice(letters)+random.choice(letters)
        self.name = nm+str(random.randint(0, 100000))
    
    def out(self):
        #EDITED here
        print('\nptcl name: ',self.name)
        print('need time: ',self.ot)
        print('pause time: ',self.pt)
        print('status: ',self.sta)
    
    def go(self,t):
        '''process a protocol 1 step'''
        #EDITED here
        if self.sta == 0:
            self.sta = 1
            self.begin = t
        elif self.sta == 1:
            if self.ot > 0:
                if self.pause:
                    if self.ot > self.pt: 
                        self.ot -= 1
                    if self.ot == self.pt: #pauses
                        self.sta = 3
                else:
                    self.ot -= 1
                    
            if self.ot == 0: #finishes
                self.sta = 2
                self.complt = t
                
                
    
class operator:
    '''generates an operator for ECL'''
    def __init__(self):
        self.sta = 0 #0 = idle, 1 = busy
        self.job = None
        self.jobname = None

    def get(self, onthread, t): 
        '''grabs a protocol from the queue and process it'''
        ind = 0
        while (ind < onthread.curlen) and (onthread.q[ind].sta != 0):
            ind += 1
        if ind < onthread.curlen: #suceessfully gets a job
            self.sta = 1
            self.job = onthread.q[ind]
            
            assert(self.job.sta == 0)
            self.jobname = self.job.name
            self.job.go(t)
            
    def drop(self, onthread, t):
        '''releases a protocol (finished or paused)'''
        #EDITED here!!
        assert(self.sta == 1)
        assert(self.job.sta == 2 or self.job.sta == 3)
        if self.job.sta == 2: #finished ptcl
            ind = onthread.q.index(self.job)
            onthread.pop(t,ind)
        self.sta = 0
        self.job = None
        self.jobname = None

class onthread:
    '''generates a queue that counts waiting time'''
    '''self.q: the queue that holds protocols
       self.len: max capacity of onthread
       self.curlen: current length of the queue'''
    '''EDIT: count down paused ptcl release time'''
    def __init__(self, customers, backlog):
        self.len = sum(c.th for c in customers)
        self.bg = backlog
        self.curlen = 0
        self.q = []
    
    def isfull(self, ptcl):
        #decides if the owner of this ptcl has used up its threads
        return ptcl.own.count == ptcl.own.th
            
    def add(self, t):
        #adds one ptcl from the backlog
        if (self.bg.curlen > 0) and (self.curlen < self.len):
            ind = 0
            while (self.isfull(self.bg.q[ind])) and (ind < self.bg.curlen-1):
                if self.bg.q[ind].time > t:
                    break
                ind += 1
            if (ind < self.bg.curlen) and (not self.isfull(self.bg.q[ind])):
                if self.bg.q[ind].time <= t: #ptcl submit time no later than curtime
                    new_ptcl = self.bg.pop(ind)

                    self.q.append(new_ptcl)
                    self.curlen += 1
                    new_ptcl.own.count += 1
        return 0
    
    def fill(self, t = 0):
        #fill onthreads with backlog ptcls
        old_len = self.curlen
        if (self.curlen < self.len) and (self.bg.curlen > 0):
            for i in range(self.len-self.curlen):
                self.add(t)
                if self.curlen == old_len: #no new ptcl is added
                    break
    
    def pop(self, t, ind = 0): 
        #pops the first ptcl from onthread
        if self.curlen > 0:
            if self.q[ind].sta == 2:
                self.curlen -= 1
                ptcl = self.q.pop(ind)
                ptcl.own.count -= 1
                self.add(t)
    
    def wait(self):
        #adds a unit of time to the ptcls in onthread
        #EDITED here
        for ptcl in self.q:
            if ptcl.sta == 0:
                ptcl.wt += 1
            elif ptcl.sta == 3:
                if ptcl.rt > 0:
                    ptcl.rt -= 1
                if ptcl.rt == 0:
                    ptcl.sta = 0 #back on waiting
                    ptcl.pause = False
#                     ptcl.wt += 1
    
    def augment(self, customer):
        #increase the capacity of onthread
        self.len += customer.th
    
class backlog:
    '''generates the backlog that ignores waiting time'''
    '''self.q: the queue that holds protocols'''
    '''self.curlen: current length of the queue'''
    '''self.sort: T if q is sorted, F otherwise'''
    def __init__(self):
        self.q = []
        self.curlen = 0
        self.issort = False
     
    def add(self,ptcl):
        #adds one ptcls to backlog
        #EDITED
        self.q.append(ptcl)
        self.curlen += 1
        self.issort = False
    
    def pop(self,ind):
        #pops the first ptcl from backlog
        if (self.curlen > 0):
            assert(self.issort)
            self.curlen -= 1
            return self.q.pop(ind)
    
    def sort(self):
        #sorts the queue of protocols
        self.q = sort_ptcls(self.q)
        self.issort = True
        