import json
import utils.path as path
# test = '''
# {"id": 3116, "data": "超 声 所 见 - 经 胸 扫 查 ：   B - M o d e   L V 内 径 ： 4 2   I V S 厚 度 ： 9   L V P W 厚 度 ： 8   L A 内 径 ： 2 7   A O 内 径 ： （ 瓣 环 ： 1 7   窦 部 ： 2 6   根 部 ： 2 0 ）   P A 内 径 ： 1 9   R V 内 径 ： 2 3   R A 内 径 ： 2 8 。 二 尖 瓣 、 三 尖 瓣 、 主 动 脉 瓣 、 肺 动 脉 瓣 轻 度 返 流 ， 左 室 顺 应 性 下 降 。", "label": [[36, 43, "INDICATORNAME"], [46, 49, "INDICATORVALUE"], [52, 61, "INDICATORNAME"], [68, 79, "INDICATORNAME"], [86, 93, "INDICATORNAME"], [96, 99, "INDICATORVALUE"], [102, 109, "INDICATORNAME"], [114, 117, "INDICATORNAME"], [120, 123, "INDICATORVALUE"], [126, 129, "INDICATORNAME"], [132, 135, "INDICATORVALUE"], [138, 141, "INDICATORNAME"], [144, 147, "INDICATORVALUE"], [152, 159, "INDICATORNAME"], [162, 165, "INDICATORVALUE"], [168, 175, "INDICATORNAME"], [178, 181, "INDICATORVALUE"], [184, 191, "INDICATORNAME"], [194, 197, "INDICATORVALUE"], [200, 205, "LOCATION"], [208, 213, "LOCATION"], [216, 223, "LOCATION"], [226, 233, "LOCATION"], [244, 247, "LOCATION"], [248, 257, "SIGN"], [64, 65, "INDICATORVALUE"], [82, 83, "INDICATORVALUE"], [234, 237, "LEVEL"], [237, 241, "SIGN"]]}
# '''

with open(path.jsonl, 'r') as f1:
    with open(path.train_file_path, 'a') as f2:
        with open(path.test_file_path, 'a') as f3:
            with open(path.all_path, 'a') as f4:
                count = 0
                for line in f1.readlines():
                    count += 1
                    print(count)
                    # print(line.strip('\n'))
                    idict = json.loads(line.strip('\n'))
                    structured = []
                    for word in idict['data']:
                        structured.append([word])
                    for label in idict['label']:
                        # print(label[0], label[1], label[2])
                        for index in range(label[0], label[1]):
                            if index == label[0]:
                                structured[index].append('B-' + label[2])
                            else:
                                structured[index].append('I-' + label[2])

                    reverse = False

                    ans_list = []
                    for word in structured:
                        if not reverse:
                            if len(word) == 1:
                                word.append('O')
                            ans_list.append({'chars': word[0], 'tags': word[1]})
                            # print(word)
                        reverse = not reverse

                    for index in range(1, len(ans_list) - 1):
                        p = ans_list[index]
                        pre = ans_list[index - 1]
                        if p['tags'] != 'O':
                            if p['tags'].split('-')[0] == 'I':
                                if pre['tags'] == 'O':
                                    p['tags'] = 'B-' + p['tags'].split('-')[1]
                                elif pre['tags'] != p['tags'] and pre['tags'].split('-')[0] != 'B':
                                    p['tags'] = 'B-' + p['tags'].split('-')[1]
                                elif pre['tags'] != p['tags'] and pre['tags'].split('-')[1] != p['tags'].split('-')[1]:
                                    p['tags'] = 'B-' + p['tags'].split('-')[1]

                    for item in ans_list:
                        f4.write(item['chars'] + '\t' + item['tags'] + '\n')
                        if count <= 800:
                            f2.write(item['chars'] + ' ' + item['tags'] + '\n')
                        else:
                            f3.write(item['chars'] + ' ' + item['tags'] + '\n')

                    f4.write('\n')
                    if count <= 800:
                        f2.write('\n')
                    else:
                        f3.write('\n')
