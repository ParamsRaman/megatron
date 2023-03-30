import numpy as np
import re
import ast
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

def read(filename, model_type="BERT"):

    k1s = ['train', 'val']
    k2s = ['lm', 'sop', 'step', 'lmppl', 'sopppl', 'gradnorm', 'samples']
    res = {k1: {k2: [] for k2 in k2s} for k1 in k1s}
    res['hyperparams'] = {}

    train_mode = "iters"

    with open(filename, 'r') as f:
        for line in f:
            if ' optimizer ' in line:
                res['hyperparams']['optimizer'] = line.strip().split(' ')[2]
            if ' global_batch_size ' in line:
                res['hyperparams']['global_batch_size'] = line.strip().split(' ')[2]
            if ' lr ' in line:
                res['hyperparams']['lr'] = line.strip().split(' ')[2]
            if ' lr_warmup_fraction ' in line:
                res['hyperparams']['warmup_fraction'] = line.strip().split(' ')[2]
            if ' lr_warmup_iters ' in line:
                res['hyperparams']['warmup_iters'] = line.strip().split(' ')[2]
            if ' lr_warmup_samples ' in line:
                res['hyperparams']['warmup_samples'] = line.strip().split(' ')[2]
            if ' tensor_model_parallel_size ' in line:
                res['hyperparams']['tensor_model_parallel_size'] = line.strip().split(' ')[2]

            ## for iter based training
            if ' train_iters ' in line and (line.strip().split(' ')[2] != 'None'):
                res['hyperparams']['train_iters'] = line.strip().split(' ')[2]
                res['hyperparams']['train_samples'] = str(int(res['hyperparams']['global_batch_size']) * int(res['hyperparams']['train_iters']))
            
            ## for samples based training
            if ' train_samples ' in line and (line.strip().split(' ')[2] != 'None'):
                res['hyperparams']['train_samples'] = line.strip().split(' ')[2]
                res['hyperparams']['train_iters'] = str(int(res['hyperparams']['train_samples']) / int(res['hyperparams']['global_batch_size']))
            ## for rampup batch size 
            if ' rampup_batch_size ' in line:
                res['hyperparams']['rampup_batch_size'] = re.sub('\W+',' ', line).strip().split(' ')[1:]
            '''if 'validation loss at iteration' in line:
                words = re.split('[ :|%()\[\]@/*\t\na-df-zA-DF-Z]', line)
                words = [word for word in words if word != '' and word != 'e' and word != '-']
                if float(words[0]) <= 10:
                    continue
                else:
                    res['val']['step'].append(float(words[0]))
                    res['val']['lm'].append(float(words[1]))
                    res['val']['lmppl'].append(float(words[2]))
                    if model_type.lower() == "bert":
                        res['val']['sop'].append(float(words[3]))
                        res['val']['sopppl'].append(float(words[4]))'''
            ## for lookahead optimizer
            if ' la_steps ' in line:
                res['hyperparams']['la_steps'] = line.strip().split(' ')[2]
            if ' la_alpha ' in line:
                res['hyperparams']['la_alpha'] = line.strip().split(' ')[2]
            if ' lookahead ....' in line:
                res['hyperparams']['lookahead'] = line.strip().split(' ')[2]

            if 'consumed samples' in line:
                if 'lm loss:' not in line:
                    continue
                words = re.split('[ :|%()\[\]@/*\t\na-df-zA-DF-Z]', line)
                words = [word for word in words if word != '' and word != 'e' and word != '-']
                if float(words[0]) <= 10:
                    continue
                else:
                    res['train']['step'].append(float(words[0]))
                    res['train']['samples'].append(float(words[2]))
                    res['train']['lm'].append(float(words[6]))
                    if model_type.lower() == "bert":
                        res['train']['gradnorm'].append(float(words[9]))
                    else:
                        res['train']['gradnorm'].append(float(words[8]))
                    if model_type.lower() == "bert":
                        res['train']['sop'].append(float(words[7]))

    for k1 in k1s:
        for k2 in k2s:
            res[k1][k2] = np.array(res[k1][k2])
    print("data before return: ", res)
    return res

def plot_figure(x_values, y_values, legendlabel, fig):
    axarr = fig.add_subplot(1,1,1)
    plt.plot(x_values, y_values, legendlabel)
    return fig

def make_itersbasedplot(plot_title, job_ids, rampup_batchsize=False, grad_norm=False, xlog=False, ylog=False, val_lmloss=False):
    plt.figure()
    for job in job_ids:
        plt_data = data[job]
        print("inside make_itersplot, print dict: {}".format(plt_data['hyperparams']))
        if plt_data['hyperparams']['warmup_fraction'] != 'None':
            warmup = plt_data['hyperparams']['warmup_fraction']
        elif plt_data['hyperparams']['warmup_iters'] != '0':
            warmup = plt_data['hyperparams']['warmup_iters']
        elif plt_data['hyperparams']['warmup_samples'] != '0':
            warmup = str(round((int(plt_data['hyperparams']['warmup_samples']) / int(plt_data['hyperparams']['train_samples'])), 2))
            print("parameshr, debug about to divide, nr: {}, dr: {}, warmup: {}".format(plt_data['hyperparams']['warmup_samples'], plt_data['hyperparams']['global_batch_size'], warmup))
        else:
            warmup = 'None'
        if rampup_batchsize:
            legendlabel = 'rampup' + ' '.join(plt_data['hyperparams']['rampup_batch_size']) + \
                ' ' + plt_data['hyperparams']['optimizer']  + \
                ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                ' lr=' + plt_data['hyperparams']['lr'] + \
                ' warmup=' + warmup + \
                ' iters=' + plt_data['hyperparams']['train_iters']
        else:
            legendlabel = plt_data['hyperparams']['optimizer']  + \
                ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                ' lr=' + plt_data['hyperparams']['lr'] + \
                ' warmup=' + warmup + \
                ' iters=' + plt_data['hyperparams']['train_iters']
        plt.plot(plt_data['train']['step'], plt_data['train']['lm'], label=legendlabel)
    plt.legend(loc="upper left")
    plt.title(plot_title)
    plt.xlabel('steps')
    plt.ylabel('lm train loss')
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.savefig(OUTPUT_DIR + '/' + plot_fname + '_lossvsiters.pdf', bbox_inches='tight')
    plt.close()

    ## Plot validation loss (currently only can be plotted vs iters, not samples)
    if val_lmloss:
        plt.figure()
        for job in job_ids:
            plt_data = data[job]
            if plt_data['hyperparams']['warmup_fraction'] != 'None':
                warmup = plt_data['hyperparams']['warmup_fraction']
            elif plt_data['hyperparams']['warmup_iters'] != '0':
                warmup = plt_data['hyperparams']['warmup_iters']
            elif plt_data['hyperparams']['warmup_samples'] != '0':
                warmup = str(round((int(plt_data['hyperparams']['warmup_samples']) / int(plt_data['hyperparams']['train_samples'])), 2))
                print("parameshr, debug about to divide, nr: {}, dr: {}, warmup: {}".format(plt_data['hyperparams']['warmup_samples'], plt_data['hyperparams']['global_batch_size'], warmup))
            else:
                warmup = 'None'
            if rampup_batchsize:
                legendlabel = 'rampup' + ' '.join(plt_data['hyperparams']['rampup_batch_size']) + \
                    ' ' + plt_data['hyperparams']['optimizer']  + \
                    ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                    ' lr=' + plt_data['hyperparams']['lr'] + \
                    ' warmup=' + warmup + \
                    ' iters=' + plt_data['hyperparams']['train_iters']
            else:
                legendlabel = plt_data['hyperparams']['optimizer']  + \
                    ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                    ' lr=' + plt_data['hyperparams']['lr'] + \
                    ' warmup=' + warmup + \
                    ' iters=' + plt_data['hyperparams']['train_iters']
            plt.plot(plt_data['val']['step'], plt_data['val']['lm'], label=legendlabel)
        plt.legend(loc="upper left")
        plt.title(plot_title)
        plt.xlabel('steps')
        plt.ylabel('lm val loss')
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.savefig(OUTPUT_DIR + '/' + plot_fname + '_vallossvsiters.pdf', bbox_inches='tight')
        plt.close()

def make_samplesbasedplot(plot_title, job_ids, rampup_batchsize=False, grad_norm=False, xlog=False, ylog=False, log_tp_size=False, lookahead=False):
    plt.figure()
    for job in job_ids:
        plt_data = data[job]
        print("inside make_samplesplot, print dict: {}".format(plt_data['hyperparams']))
        if plt_data['hyperparams']['warmup_fraction'] != 'None':
            warmup = plt_data['hyperparams']['warmup_fraction']
        elif plt_data['hyperparams']['warmup_samples'] != '0':
            warmup = plt_data['hyperparams']['warmup_samples']
        elif plt_data['hyperparams']['warmup_iters'] != '0':
            warmup = str(round((int(plt_data['hyperparams']['warmup_iters']) / int(plt_data['hyperparams']['train_iters'])), 2))
        else:
            warmup = 'None'
        if rampup_batchsize:
            legendlabel = 'rampup' + ' '.join(plt_data['hyperparams']['rampup_batch_size']) + \
                ' ' + plt_data['hyperparams']['optimizer']  + \
                ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                ' lr=' + plt_data['hyperparams']['lr'] + \
                ' warmup=' + warmup + \
                ' samples=' + plt_data['hyperparams']['train_samples']
            plt.plot(plt_data['train']['samples'], plt_data['train']['lm'], label=legendlabel)
            #if VAL_LMLOSS:
            #    plt.plot(plt_data['val']['samples'], plt_data['val']['lm'], label=legendlabel)
        elif lookahead and (plt_data['hyperparams']['lookahead'] == "True"):
            legendlabel = 'lookahead' + ' ' + plt_data['hyperparams']['la_steps']  + ' ' + plt_data['hyperparams']['la_alpha'] + \
                ' ' + plt_data['hyperparams']['optimizer']  + \
                ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                ' lr=' + plt_data['hyperparams']['lr'] + \
                ' warmup=' + warmup + \
                ' samples=' + plt_data['hyperparams']['train_samples']
            plt.plot(plt_data['train']['samples'], plt_data['train']['lm'], label=legendlabel)
        else:
            legendlabel = plt_data['hyperparams']['optimizer']  + \
                ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                ' lr=' + plt_data['hyperparams']['lr'] + \
                ' warmup=' + warmup + \
                ' samples=' + plt_data['hyperparams']['train_samples']
            if log_tp_size:
                legendlabel = legendlabel + ' TP=' + plt_data['hyperparams']['tensor_model_parallel_size']
            plt.plot(plt_data['train']['samples'], plt_data['train']['lm'], label=legendlabel)
    if rampup_batchsize:
        plt.axvline(x = 10000000, color = 'r', linestyle='dashed', label = 'rampup cutoff (b=32K reached at 10M samples)')
    #    plt.axvline(x = 623456, color = 'b', linestyle='dotted', label = 'b=2k reached at 623k samples')
    plt.legend(loc="upper left")
    plt.title(plot_title)
    plt.xlabel('samples')
    plt.ylabel('lm train loss')
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.savefig(OUTPUT_DIR + '/' + plot_fname + '_lossvssamples.pdf', bbox_inches='tight')
    plt.close()

    ## Made gradient norm plot
    if not grad_norm:
        return
    plt.figure()
    for job in job_ids:
        plt_data = data[job]
        if plt_data['hyperparams']['warmup_fraction'] != 'None':
            warmup = plt_data['hyperparams']['warmup_fraction']
        elif plt_data['hyperparams']['warmup_samples'] != '0':
            warmup = plt_data['hyperparams']['warmup_samples']
        elif plt_data['hyperparams']['warmup_iters'] != '0':
            warmup = str(round((int(plt_data['hyperparams']['warmup_iters']) / int(plt_data['hyperparams']['train_iters'])), 2))
        else:
            warmup = 'None'
        if rampup_batchsize:
            legendlabel = 'rampup' + ' '.join(plt_data['hyperparams']['rampup_batch_size']) + \
                ' ' + plt_data['hyperparams']['optimizer']  + \
                ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                ' lr=' + plt_data['hyperparams']['lr'] + \
                ' warmup=' + warmup + \
                ' samples=' + plt_data['hyperparams']['train_samples']
            plt.plot(plt_data['train']['samples'], plt_data['train']['gradnorm'], label=legendlabel)
        else:
            legendlabel = plt_data['hyperparams']['optimizer']  + \
                ' b=' + plt_data['hyperparams']['global_batch_size'] + \
                ' lr=' + plt_data['hyperparams']['lr'] + \
                ' warmup=' + warmup + \
                ' samples=' + plt_data['hyperparams']['train_samples']
            plt.plot(plt_data['train']['samples'], plt_data['train']['gradnorm'], label=legendlabel)
    #if rampup_batchsize:
    #    plt.axvline(x = 10000000, color = 'r', linestyle='dashed', label = 'rampup cutoff (b=32K reached at 10M samples)')
    #    plt.axvline(x = 623456, color = 'b', linestyle='dotted', label = 'b=2k reached at 623k samples')
    plt.legend(loc="upper left")
    plt.title(plot_title)
    plt.xlabel('samples')
    plt.ylabel('grad norm')
    if xlog:
        plt.xscale('log')
    ylog = True
    if ylog:
        plt.yscale('log')
    plt.savefig(OUTPUT_DIR + '/' + plot_fname + '_gradnormvssamples.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Some general plot configuration
    PLOT_TITLE = "" # Used for output filename
    YLOG = True
    XLOG = False

    # Select type of plots needed
    TR_LMLOSS_ITERS = True
    TR_LMLOSS_SAMPLES = True
    TR_LMLOSS_TIME = False
    VAL_LMLOSS = False
    VAL_PPL = False
    LOG_TP_SIZE = False

    GRAD_NORM = False
    RAMPUP_BATCHSIZE = False
    LOOKAHEAD = False

    # Get user inputs
    print("Enter model type (BERT, T5): ")
    MODEL_TYPE = sys.stdin.readline().strip()
    print("Enter log file directory: ")
    LOG_DIR = sys.stdin.readline().strip()
    print("Enter output directory for plot files: ")
    OUTPUT_DIR = sys.stdin.readline().strip()

    job_ids = []
    print("Enter job_id in separate lines. Ctrl+D to end the list.")
    while True:
        line = sys.stdin.readline()
        if line:
            tmp = line.strip()
            if tmp != '':
                job_ids.append(tmp)
        else:
            print("Done")
            break

    ## read plot data
    data = {}
    for job in job_ids:
        data[job] = read(LOG_DIR + "/log-" + job, MODEL_TYPE)

    plot_fname = PLOT_TITLE if PLOT_TITLE != '' else time.strftime("%Y%m%d-%H%M%S")
    if TR_LMLOSS_ITERS:
        make_itersbasedplot(plot_fname, job_ids, RAMPUP_BATCHSIZE, GRAD_NORM, XLOG, YLOG, VAL_LMLOSS)
    if TR_LMLOSS_SAMPLES:
        make_samplesbasedplot(plot_fname, job_ids, RAMPUP_BATCHSIZE, GRAD_NORM, XLOG, YLOG, LOG_TP_SIZE, LOOKAHEAD)
