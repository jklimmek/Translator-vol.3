def read_txt(path, skip_blank=True):
    lines = []
    with open(path, 'r', encoding="utf-8") as file:
        for line in file:
            if skip_blank and line == '\n':
                continue
            lines.append(line.strip())
    return lines

def save_txt(txt, path, mode="w", skip_blank=True):
    with open(path, mode=mode, encoding='utf-8') as file:
        for line in txt:
            if skip_blank and line == '':
                continue
            file.write(line + '\n')
