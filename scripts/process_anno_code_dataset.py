def process_anno(path):
    regex_split = re.compile(r'[\-|\ |\_ |\.]+')
    regex_sub = re.compile(r'[]')
    f = open(path)
    processed_anno = []
    for line in f:
        clean = regex_split.split(line.strip)
        processed_anno.append(clean)
