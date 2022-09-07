import matplotlib.pyplot as plt
import numpy as np



def set_box_color(bp, color, linestyle='solid'):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle)
    plt.setp(bp['caps'], color=color, linestyle=linestyle)
    plt.setp(bp['medians'], color=color, linestyle=linestyle)


def main():
    ###########################################################################
    ### Data Preparation

    data_mel_mlp = [[0.9829268292, 0.9853658536, 0.990243902],          # instr_fml (nsynth)
                    [0.9682926829, 0.9682926829, 0.985280062],          # source
                    [0.6731707317, 0.6756097561, 0.687804878],          # pitch
                    [0.3317073171, 0.3390243902, 0.343902439],          # velocity
                    [0.7121212121, 0.7196969697, 0.727272727],          # instr_fml (rwc)
                    [0.4631578947, 0.4760233918, 0.456140350],          # dynamic
                    [0.3887323943, 0.3690140845, 0.383098591],          # PS (Strings)
                    [0.4724919093, 0.4692556634, 0.482200647],          # PS (Woodwinds)
                    [0.4114583333, 0.4322916666, 0.401041666],]         # PS (Brass)
                    

    data_mel_svm = [[0.97804878048],                                    # instr_fml (nsynth)
                    [0.97804878048],                                    # source
                    [0.72926829268],                                    # pitch
                    [0.35609756097],                                    # velocity
                    [0.77651515151],                                    # instr_fml (rwc)
                    [0.51461988300],                                    # dynamic
                    [0.44225352112],                                    # PS (Strings)
                    [0.51779935275],                                    # PS (Woodwinds)
                    [0.44270833333]]                                    # PS (Brass)
                    
                    
    data_mel_dt  = [[0.6487804878, 0.626829268, 0.6243902439],          # instr_fml (nsynth)
                    [0.7268292683, 0.739024390, 0.7463414634],          # source
                    [0.3341463415, 0.312195122, 0.3268292683],          # pitch
                    [0.1902439024, 0.190243902, 0.1902439024],          # velocity
                    [0.5795454545, 0.583333333, 0.5606060606],          # instr_fml (rwc) 
                    [0.4327485380, 0.426900585, 0.415204678],           # dynamic
                    [0.2901408450, 0.357746478, 0.27605633802],         # PS (Strings)
                    [0.4045307443, 0.349514563, 0.3592233009],          # PS (Woodwinds)
                    [0.2291666666, 0.234375, 0.260416666]]              # PS (Brass)
                    
                    

    data_midi_mlp = [[0.9975609756, 0.9902439024, 0.992682926],         # instr_fml (nsynth) 
                    [0.9926829268, 0.9878048780, 0.985365854],          # source
                    [0.7341463414, 0.7268292682, 0.739024390],          # pitch
                    [0.3146341463, 0.3292682926, 0.365853658],          # velocity
                    [1, 1, 1],                                          # instr_fml (rwc)
                    [0.6865497076, 0.6912280702, 0.684210526],          # dynamic
                    [0.6788732394, 0.7239436619, 0.692957746],          # PS (Strings)
                    [0.7216828479, 0.7087378640, 0.737864077],          # PS (Woodwinds)
                    [0.78125, 0.80208333, 0.802083333]]                 # PS (Brass)
                    

    data_midi_svm = [[0.99268292682],                                   # instr_fml (nsynth)
                    [0.98780487805],                                    # source
                    [0.76097560976],                                    # pitch
                    [0.36097560976],                                    # velocity
                    [0.99574468085],                                    # instr_fml (rwc)
                    [0.70760233918],                                    # dynamic
                    [0.78028169014],                                    # PS (Strings)
                    [0.75404530744],                                    # PS (Woodwinds)
                    [0.8125],]                                          # PS (Brass)
                    
                    
    data_midi_dt  = [[0.6902439024, 0.665853659, 0.6853658537],         # instr_fml (nsynth)
                    [0.7414634146, 0.719512195, 0.7463414634],          # source
                    [0.3219512195, 0.312195122, 0.3463414634],          # pitch
                    [0.2219512195, 0.221951219, 0.2170731707],          # velocity 
                    [0.8978723404, 0.889361702, 0.8680851063],          # instr_fml (rwc)
                    [0.4573099415, 0.480701754, 0.5216374269],          # dynamic
                    [0.3802816901, 0.380281690, 0.35774647887],         # PS (Strings)
                    [0.517799352, 0.482200647, 0.4466019417],           # PS (Woodwinds)
                    [0.473958333, 0.401041667, 0.43229166666]]          # PS (Brass)
                    
                    
    data_mv      = [[0.1707317073],                                     # instr_fml (nsynth)
                    [0.3951219512],                                     # source
                    [0.1658536585],                                     # pitch
                    [0.1975609756],                                     # velocity 
                    [0.3],                                               # instr_fml (rwc)
                    [0.3637426901],                                     # dynamic
                    [0.2112676056],                                     # PS (Strings)
                    [0.3398058252],                                     # PS (Woodwinds)
                    [0.203125]]                                         # PS (Brass)
                    

    data_cr      = [[1/10],                                             # instr_fml (nsynth)
                    [1/3 ],                                             # source
                    [1/10],                                             # pitch
                    [1/5 ],                                             # velocity
                    [1/7 ],                                             # instr_fml (rwc)
                    [1/3 ],                                             # dynamic
                    [1/20],                                             # PS (Strings)
                    [1/7 ],                                             # PS (Woodwinds)
                    [1/18]]                                             # PS (Brass)
                    


    ticks = ['IF(Nsynth)-10', 'Source-3', 'Pitch-10', 'Velocity-5', 'IF(RWC)-7', 'Dynamics-3', 'PS(Strings)-20', 'PS(Woodwinds)-7', 'PS(Brass)-18', ]
    linestyle = ['-', 'dotted', '-', 'dotted', '-',]

    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(1,1,1)

    """
    bpmlp = plt.boxplot(data_midi_mlp, positions=np.array(range(len(data_midi_mlp)))*3.0-0.4, sym='', widths=0.6)
    bpdt  = plt.boxplot(data_midi_dt,  positions=np.array(range(len(data_midi_dt)))*3.0, sym='', widths=0.6)
    bpsvm = plt.boxplot(data_midi_svm, positions=np.array(range(len(data_midi_svm)))*3.0+0.4, sym='', widths=0.6)
    bpmv  = plt.boxplot(data_mv,      positions=np.array(range(len(data_mv)))*3.0-0.4, sym='', widths=0.6)
    bpcr  = plt.boxplot(data_cr,      positions=np.array(range(len(data_cr)))*3.0+0.4, sym='', widths=0.6)


    set_box_color(bpmlp, '#000000', linestyle[0]) # colors are from http://colorbrewer2.org/
    set_box_color(bpsvm, '#000000', linestyle[1])
    set_box_color(bpdt,  '#696969', linestyle[2])
    set_box_color(bpmv,  '#696969', linestyle[3])
    set_box_color(bpcr,  '#A9A9A9', linestyle[4])


    plt.plot([], c='#000000', label='Multilayer Perceptron', linestyle=linestyle[0])
    plt.plot([], c='#000000', label='Support Vector Machine', linestyle=linestyle[1])
    plt.plot([], c='#696969', label='Decision Tree', linestyle=linestyle[2])
    plt.plot([], c='#696969', label='Major Voting', linestyle=linestyle[3])
    plt.plot([], c='#A9A9A9', label='Chance Rate', linestyle=linestyle[4])
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="center", mode="expand", borderaxespad=0, ncol=3)
    """
    
    scdt  = plt.scatter(x=np.array(range(len(data_mel_dt)))*3.0, y=np.mean(np.array(data_mel_dt), axis=1), marker='s', label='Decision Tree')
    scmlp = plt.scatter(x=np.array(range(len(data_mel_mlp)))*3.0-0.1, y=np.mean(np.array(data_mel_mlp), axis=1), marker='X', label='Multilayer Perceptron')
    scsvm = plt.scatter(x=np.array(range(len(data_mel_svm)))*3.0+0.1, y=np.mean(np.array(data_mel_svm), axis=1), marker='P', label='Support Vector Machine')
    sccr  = plt.scatter(x=np.array(range(len(data_cr)))*3.0-0.1, y=np.mean(np.array(data_cr), axis=1), marker='^', label='Chance Rate')
    scmv  = plt.scatter(x=np.array(range(len(data_mv)))*3.0+0.1, y=np.mean(np.array(data_mv), axis=1), marker='v', label='Major Voting')
    

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="center", mode="expand", borderaxespad=0, ncol=3)
    ax.set_ylabel("Micro F1-Score")
    # plt.xticks(range(0, len(ticks) * 3, 3), [3,3,5,10,7,18,20])
    plt.xticks(range(0, len(ticks) * 3, 3), ticks)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.xlim(-2, len(ticks)*3)
    # plt.title("Prediction results using embeddings extracted from \n \"wav-transMIDI-aug-asm\" system", y=1.22)
    plt.tight_layout()
    plt.savefig('probing_boxcompare_mel_shape.pdf')
    






if __name__ == '__main__':
    main()