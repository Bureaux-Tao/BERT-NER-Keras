import utils.path as path


with open(path.all_path, 'r',encoding='utf-8') as f:
    max = 0
    count = 0
    # try:
    for i in f.readlines():
        count += 1
        if i == '\n':
            if count > max:
                max = count
            count = 0
#
# except:
#     print("err:" + str(count))

print(max)
