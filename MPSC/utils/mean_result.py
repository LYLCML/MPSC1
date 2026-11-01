
import numpy as np

metric_name = 'test_macro_f1'
# metric_name = 'test_auroc'
# metric_name = 'test_acc'
result_file = '../Results/cifar80no/asym_40/ONLRC_a0.5b1o0.25lr0.001d80w10e200bs128s0-20241116_092355-bestF1_0.5837/log.txt'
# result_file = '../Results/[\'web-aircraft\', \'web-bird\']/resnet50/ONLRC_un_bird_a0.5b1.0o0.25lr0.005-20241202_140335-bestF1_0.7639/log.txt'
# result_file = '../Results/food101n/resnet50/ONLRC_a0.5b1.0o0.25lr0.01-20241101_022324-bestAcc_87.3030/log.txt'
# ----------------------------------- training-related -----------------------------------
if __name__ == '__main__':
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, 11):
    # for idx in range(121, 131):
        line = lines[-idx].strip()
        epoch, train_loss, train_acc, test_loss, test_acc, test_auroc, test_macro_f1 = line.split(' | ')[:7]
        metric = {'test_acc': test_acc,
                  'test_auroc': test_auroc,
                  'test_macro_f1': test_macro_f1
                  }
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_macro_f1:
            test_acc_list.append(float(metric[metric_name].split(': ')[1]))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), metric[metric_name].split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.4f}, std: {test_acc_list.std():.4f}')
        # return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.4f}, std: {test_acc_list.std():.4f}')
        print(f'mean: {test_acc_list2.mean():.4f}, std: {test_acc_list2.std():.4f}')
        # return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
        #         'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
        #         'valid_epoch': valid_epoch}