##### case checking script #####
import numpy as np
from Sim13_Init import initialize, sim_run

def check():
    ### test1: customer thread ###
    cus_ths = [10, 7]
    cus_nums = [1,10,50]
    cus_lams = [10,20]
    op_nums = [0,1,7]
    for cth in cus_ths:
        for cnum in cus_nums:
            for clam in cus_lams:
                for onum in op_nums:
                    cus_test1, op_test1, bl_test1, ot_test1 = initialize(
                                x0 = 0, op_num = onum, cus_th = cth, cus_num = cnum, 
                                pause_prob = 0, ot = -1, f_ot = -1, f_rls = -1,
                                show = False)
                    sim_run(cus_test1, op_test1, bl_test1, ot_test1,
                            cus_lam = clam, cycle = 24, cycle_num = 1, 
                            pause_prob = 0, f_ot = 30, 
                            poisson = False, ops_num = None,
                            show = False, show_fin = False)
                    assert(ot_test1.curlen == min(clam*cnum, cth*cnum))
                    assert(bl_test1.curlen == clam*cnum-ot_test1.curlen)

    ### test2: operator rounds ###
    x0s = [1,5,50,100]
    op_nums = [0,7,20]
    for x in x0s:
        for onum in op_nums:
            cus_test2, op_test2, bl_test2, ot_test2 = initialize(
                        x0 = x, op_num = onum, cus_th = 0, cus_num = 0, 
                        pause_prob = 0, ot = -1, f_ot = 23, f_rls = -1,
                        show = False)
            sim_run(cus_test2, op_test2, bl_test2, ot_test2,
                    cus_lam = 0, cycle = 24, cycle_num = 1, 
                    pause_prob = 0, 
                    poisson = False, ops_num = None,
                    show = False, show_fin = False)
            assert(ot_test2.curlen == max(0,x-onum))
            assert(bl_test2.curlen == 0)

    ### test3: operator rounds with pause ###
    x0s = [1,5,50,100]
    op_nums = [0,7,20]
    for x in x0s:
        for onum in op_nums:
            cus_test3, op_test3, bl_test3, ot_test3 = initialize(
                        x0 = x, op_num = onum, cus_th = 0, cus_num = 0, 
                        pause_prob = 1, ot = -1, f_ot = 23, f_rls = 24,
                        show = False)
            sim_run(cus_test3, op_test3, bl_test3, ot_test3,
                    cus_lam = 0, cycle = 24, cycle_num = 2, 
                    pause_prob = 0, 
                    poisson = False, ops_num = None,
                    show = False, show_fin = False)
            assert(ot_test3.curlen == max(0,x-onum))
            assert(bl_test3.curlen == 0)

    ### test4: waiting times ###
    cus_nums = [1]
    op_nums = [1,5,10]
    cus_lams = [10,20]
    for cnum in cus_nums:
        for clam in cus_lams:
            for onum in op_nums:
                cus_test4, op_test4, bl_test4, ot_test4 = initialize(
                            x0 = 0, op_num = onum, cus_th = 100, cus_num = cnum, 
                            pause_prob = 0, ot = -1, f_ot = -1, f_rls = -1,
                            show = False)
                waiting_times, outstand_times, ptcl_fin, ptcl_num, backlog_sz, thread_sz = sim_run(
                        cus_test4, op_test4, bl_test4, ot_test4,
                        cus_lam = clam, cycle = 24, cycle_num = 2, 
                        pause_prob = 0, f_ot = 23,
                        poisson = False, ops_num = None,
                        show = False, show_fin = False)
                assert(bl_test4.curlen == 0)
                if int(24/clam)==2 and onum == 1:
                    assert((0,23) in waiting_times)
                    assert((22,47) in waiting_times)
                elif int(24/clam)==1 and onum == 1:
                    assert((0,23) in waiting_times)
                    assert((23,47) in waiting_times)
                elif int(24/clam)==2 and onum == 5:
                    assert((0,23) in waiting_times)
                    assert((0,25) in waiting_times)
                    assert((14,47) in waiting_times)
                elif int(24/clam)==1 and onum == 5:
                    assert((0,23) in waiting_times)
                    assert((0,24) in waiting_times)
                    assert((19,47) in waiting_times)
                elif int(24/clam)==2 and onum == 10:
                    assert((0,23) in waiting_times)
                    assert((0,41) in waiting_times)
                elif int(24/clam)==1 and onum == 10:
                    assert((0,23) in waiting_times)
                    assert((0,32) in waiting_times)
                    assert((14,47) in waiting_times)

    ### test5: waiting times with pause ###
    cus_nums = [1]
    op_nums = [1,5]
    cus_lams = [10,20]
    for cnum in cus_nums:
        for clam in cus_lams:
            for onum in op_nums:
                cus_test4, op_test4, bl_test4, ot_test4 = initialize(
                            x0 = 0, op_num = onum, cus_th = 100, cus_num = cnum, 
                            pause_prob = 0, ot = -1, f_ot = -1, f_rls = -1,
                            show = False)
                waiting_times, outstand_times, ptcl_fin, ptcl_num, backlog_sz, thread_sz = sim_run(
                        cus_test4, op_test4, bl_test4, ot_test4,
                        cus_lam = clam, cycle = 24, cycle_num = 2, 
                        pause_prob = 1, f_ot = 20, f_rls = 10,
                        poisson = False, ops_num = None,
                        show = False, show_fin = False)
                assert(bl_test4.curlen == 0)
                if int(24/clam)==2 and onum == 1:
                    assert((0,30) in waiting_times)
                    assert((9,41) in waiting_times)
                elif int(24/clam)==1 and onum == 1:
                    assert((0,30) in waiting_times)
                    assert((10,41) in waiting_times)
                elif int(24/clam)==2 and onum == 5:
                    assert((0,30) in waiting_times)
                    assert((1,41) in waiting_times)
                    assert((1,45) in waiting_times)
                elif int(24/clam)==1 and onum == 5:
                    assert((0,30) in waiting_times)
                    assert((6,41) in waiting_times)
                    assert((6,45) in waiting_times)
    print('check passed!')