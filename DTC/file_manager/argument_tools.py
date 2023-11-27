import numpy as np
import sys
import multiprocessing as mp
import getopt

def parse_arguments(sys_argv, subjects):

    """
    argv = sys_argv[1:]
    try:
        opts, args = getopt.getopt(argv, "m:s:hb:e:", ["first=", "last="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -b first -s last')
            sys.exit()
        elif opt in ("-b", "--first"):
            start = arg
        elif opt in ("-e", "--last"):
            end = arg
        if opt == '-m':
            print(f'Max processes is set at {arg}')
            max_processors =
    if 'start' in locals():
        del (start, end)
    if 'start' in locals():
        start = int(start)
        if 'end' in locals():
            subjects = subjects[int(start):int(end) + 1]
        else:
            subjects = subjects[start:]
    if 'start' not in locals():
        if 'end' not in locals():
            subjects = subjects
        else:
            subjects = subjects[0:end]
    """

    if np.size(sys_argv) > 1:
        firstsubj = int(sys_argv[1])-1
        if np.size(sys_argv) > 2:
            lastsubj = int(sys_argv[2])
        else:
            lastsubj = firstsubj+1
    else:
        firstsubj = 0
        lastsubj = np.size(subjects)

    if np.size(sys_argv) > 3:
        max_processors = int(sys_argv[3])
    else:
        max_processors = 1

    if np.size(sys_argv) > 4:
        subject_processes = int(sys_argv[4])
    elif np.size(subjects)>0:
        subject_processes = np.size(subjects)
    else:
        raise Exception('Empty subjects')

    if mp.cpu_count() < max_processors:
        max_processors = mp.cpu_count()
    if max_processors < subject_processes:
        subject_processes = max_processors


    function_processes = int(max_processors / subject_processes)

    return subject_processes, function_processes, firstsubj, lastsubj

def parse_arguments_function(sys_argv):

    """
    argv = sys_argv[1:]
    try:
        opts, args = getopt.getopt(argv, "m:s:hb:e:", ["first=", "last="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -b first -s last')
            sys.exit()
        elif opt in ("-b", "--first"):
            start = arg
        elif opt in ("-e", "--last"):
            end = arg
        if opt == '-m':
            print(f'Max processes is set at {arg}')
            max_processors =
    if 'start' in locals():
        del (start, end)
    if 'start' in locals():
        start = int(start)
        if 'end' in locals():
            subjects = subjects[int(start):int(end) + 1]
        else:
            subjects = subjects[start:]
    if 'start' not in locals():
        if 'end' not in locals():
            subjects = subjects
        else:
            subjects = subjects[0:end]
    """

    if np.size(sys_argv) > 1:
        function_processes = int(sys_argv[1])
    else:
        function_processes = 1

    return function_processes