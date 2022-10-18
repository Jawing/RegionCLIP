import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_images(plt_names,filename1,filename2,filename3):

    df1 = pd.read_pickle(f"./vis/{filename1}.pkl")
    df2 = pd.read_pickle(f"./vis/{filename2}.pkl")
    df3 = pd.read_pickle(f"./vis/{filename3}.pkl")
    print(plt_names)
    print(filename1,'\n',df1)
    print(filename2,'\n',df2)
    print(filename3,'\n',df3)
    #makesure xlabel names are the same
    assert df1.name.equals(df2.name)
    assert df2.name.equals(df3.name)
    xtick_names = df1.name.tolist()

    score_t = df3['score_t'][0]
    N = 6
    ind = np.arange(N) 
    width = 0.20


    #plot number of bbox missing by RPN
    xvals = df1.num_nol.tolist()
    bar1 = plt.bar(ind, xvals, width, color = 'blue')
    yvals = df2.num_nol.tolist()
    bar2 = plt.bar(ind+width, yvals, width, color='orange')
    zvals = df3.num_nol.tolist()
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'purple')
    tvals = df3.num_box.tolist()
    # bar4 = plt.bar(ind+width*3, tvals, width, color = 'green')
    plt.xlabel("Classes")
    plt.ylabel("Number of bboxes")
    plt.title(f"Number of bbox missing by RPN out of \n{tvals} Totals")
    min_list = min(xvals+yvals+zvals)
    max_list = max(xvals+yvals+zvals)
    plt.yticks(np.arange(0,max_list+10,10))
    plt.xticks(ind+width,xtick_names)
    plt.legend( (bar1, bar2, bar3), (filename1,filename2,filename3) )
    #plt.show()
    plt.savefig(f"./vis/{plt_names}_fig_num_nol.jpg")
    plt.clf()



    #plot percent of bbox missing by RPN
    width = 0.25
    xvals = df1["nol%"].tolist()
    bar1 = plt.bar(ind, xvals, width, color = 'blue')
    yvals = df2["nol%"].tolist()
    bar2 = plt.bar(ind+width, yvals, width, color='orange')
    zvals = df3["nol%"].tolist()
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'purple')
    plt.legend( (bar1, bar2, bar3), (filename1,filename2,filename3) )
    plt.xlabel("Classes")
    plt.ylabel("Percent of bboxes")
    plt.title("Percent of bbox missing by RPN")

    min_list = min(xvals+yvals+zvals)
    max_list = max(xvals+yvals+zvals)
    plt.xticks(ind+width,xtick_names)
    plt.yticks(np.arange(0,max_list+5,5))
    plt.savefig(f"./vis/{plt_names}_fig_per_nol.jpg")
    plt.clf()


    #plot number of bbox missing by RPN with threshold
    width = 0.20
    xvals = df1.num_nol_t.tolist()
    bar1 = plt.bar(ind, xvals, width, color = 'blue')
    yvals = df2.num_nol_t.tolist()
    bar2 = plt.bar(ind+width, yvals, width, color='orange')
    zvals = df3.num_nol_t.tolist()
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'purple')
    tvals = df3.num_box.tolist()
    # bar4 = plt.bar(ind+width*3, tvals, width, color = 'green')
    plt.xlabel("Classes")
    plt.ylabel("Number of bboxes")
    plt.title(f"Number of bbox missing by RPN with {score_t} Score Threshold \nout of {tvals} Totals")
    min_list = min(xvals+yvals+zvals)
    max_list = max(xvals+yvals+zvals)
    plt.yticks(np.arange(0,max_list+10,10))
    plt.xticks(ind+width,xtick_names)
    plt.legend( (bar1, bar2, bar3), (filename1,filename2,filename3) )
    #plt.show()
    plt.savefig(f"./vis/{plt_names}_fig_num_nol_t.jpg")
    plt.clf()


    #plot percent of bbox missing by RPN
    width = 0.25
    xvals = df1["nol_t%"].tolist()
    bar1 = plt.bar(ind, xvals, width, color = 'blue')
    yvals = df2["nol_t%"].tolist()
    bar2 = plt.bar(ind+width, yvals, width, color='orange')
    zvals = df3["nol_t%"].tolist()
    bar3 = plt.bar(ind+width*2, zvals, width, color = 'purple')
    plt.legend( (bar1, bar2, bar3), (filename1,filename2,filename3) )
    plt.xlabel("Classes")
    plt.ylabel("Percent of bboxes")
    plt.title(f"Percent of bbox missing by RPN with {score_t} Score Threshold")

    min_list = min(xvals+yvals+zvals)
    max_list = max(xvals+yvals+zvals)
    plt.xticks(ind+width,xtick_names)
    plt.yticks(np.arange(0,max_list+5,5))
    plt.savefig(f"./vis/{plt_names}_fig_per_nol_t.jpg")
    plt.clf()

#change names of files to visualize
#dataset = 'custom' 
#dataset = 'hw'
dataset = 'basic'

plt_names = f'{dataset}_0509_6k1k'
filename1 = f'{dataset}_rpn_fp_05_6k_1k'
filename2 = f'{dataset}_rpn_fp_07_6k_1k'
filename3 = f'{dataset}_rpn_fp_09_6k_1k'
generate_images(plt_names,filename1,filename2,filename3)
plt_names = f'{dataset}_0509_12k2k'
filename1 = f'{dataset}_rpn_fp_05_12k_2k'
filename2 = f'{dataset}_rpn_fp_07_12k_2k'
filename3 = f'{dataset}_rpn_fp_09_12k_2k'
generate_images(plt_names,filename1,filename2,filename3)
plt_names = f'{dataset}_0509_24k4k'
filename1 = f'{dataset}_rpn_fp_05_24k_4k'
filename2 = f'{dataset}_rpn_fp_07_24k_4k'
filename3 = f'{dataset}_rpn_fp_09_24k_4k'
generate_images(plt_names,filename1,filename2,filename3)
plt_names = f'{dataset}_05_allk'
filename1 = f'{dataset}_rpn_fp_05_6k_1k'
filename2 = f'{dataset}_rpn_fp_05_12k_2k'
filename3 = f'{dataset}_rpn_fp_05_24k_4k'
generate_images(plt_names,filename1,filename2,filename3)
plt_names = f'{dataset}_07_allk'
filename1 = f'{dataset}_rpn_fp_07_6k_1k'
filename2 = f'{dataset}_rpn_fp_07_12k_2k'
filename3 = f'{dataset}_rpn_fp_07_24k_4k'
generate_images(plt_names,filename1,filename2,filename3)
plt_names = f'{dataset}_09_allk'
filename1 = f'{dataset}_rpn_fp_09_6k_1k'
filename2 = f'{dataset}_rpn_fp_09_12k_2k'
filename3 = f'{dataset}_rpn_fp_09_24k_4k'
generate_images(plt_names,filename1,filename2,filename3)