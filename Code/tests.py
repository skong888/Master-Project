from scipy.stats import kruskal, shapiro, normaltest, anderson, mannwhitneyu
import scikit_posthocs as sp
import numpy as np

lite = False

def openFile(fileName):
    with open(fileName, 'r') as f:
        next(f)  # skip first line
        time = []
        for line in f:
            try:
                t, p = [float(x) for x in line.split()]
            except ValueError:
                t = [float(x) for x in line.split()]
            time.append(t)
        f.close()
    return time

def normTest(timeList):
    alpha = 0.05
    nameList = ['Nano', 'Pi', 'Coral']
    print('------Shapiro-------')
    for i in range(len(timeList)):
        time = timeList[i]
        stat, p = shapiro(time)
        print(nameList[i])
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)\n')
        else:
            print('Sample does not look Gaussian (reject H0)\n')


    print('------Normaltest-------')
    for i in range(len(timeList)):
        time = timeList[i]
        print(nameList[i])
        stat, p = normaltest(time)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)\n')
        else:
            print('Sample does not look Gaussian (reject H0)\n')

    print('------Anderson-------')
    for i in range(len(timeList)):
        time = timeList[i]
        print(nameList[i])
        result = anderson(time.flatten(), dist='norm')

        print('Statistic: %.3f' % result.statistic)
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        print('')


if lite == False:
    print('Separable')
    file_name1 = './Models/Separable/Pruned_vLog_nano.txt'
    file_name2 = './Models/Separable/Pruned_vLog_pi.txt'

    time1 = openFile(file_name1)
    time2 = openFile(file_name2)

    timeList = np.array([time1, time2])
    normTest(timeList)

    print('------Mann-Whitney-------')
    stat, p = mannwhitneyu(time1, time2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')


elif lite == True:
    file_name1 = './coralmodels/separable/prunedlitevLog_nano.txt'
    file_name2 = './coralmodels/separable/prunedlitevLog_pi.txt'
    file_name3 = './coralmodels/separable/prunedlitevLog_coral.txt'

    print('Pruned Separable Lite')

    time1 = openFile(file_name1)
    time2 = openFile(file_name2)
    time3 = openFile(file_name3)
    timeList = np.array([time1, time2, time3])

    normTest(timeList)
    print('Kruskal test')
    kruskal_stat, kp = kruskal(time1, time2, time3)
    print('Statistics=%.3f, p=%.3f' % (kruskal_stat, kp))
    # interpret
    alpha = 0.05
    if kp > alpha:
        print('Same distribution (fail to reject H0)\n')
    else:
        print('Different distribution (reject H0)\n')

    print('Dunn test')
    dunn = sp.posthoc_dunn(timeList, p_adjust='holm')
    print(dunn)
