import utils.path as path

with open(path.unlabeled_bio, 'r') as f:
    with open(path.number_handled_bio, 'w') as f1:
        listOfLines = f.readlines()
        i = 0
        while i < len(listOfLines) - 2:
            a = listOfLines[i].strip('\n').split('\t')
            b = listOfLines[i + 1].strip('\n').split('\t')
            if len(a) > 1:
                if a[0] in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0') and b[0] in (
                        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
                    f1.writelines(a[0] + '\t' + 'B-INDICATORVALUE' + '\n' + b[0] + '\t' + 'I-INDICATORVALUE' + '\n')
                    i += 1
                # else:
                #     f1.writelines(a[0] + '\t' + a[1] + '\n')
                # print(a[0], a[1])
                else:
                    f1.writelines(a[0] + '\t' + a[1] + '\n')
            else:
                f1.write('\n')
            i += 1
