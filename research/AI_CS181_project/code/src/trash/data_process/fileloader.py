def load_text(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", encoding='utf8') as f:
        for line in f:
            lst.append(line.strip('\n'))
    return lst
def load_label(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", errors='ignore') as f:
        for line in f:
            lst.append(int(line.strip('\n')))
    return lst