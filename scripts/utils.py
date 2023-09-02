def read_txt(path, skip_blank=True):
    """
    Reads a text file and returns a list of lines.

    Parameters:
        path (str): Path to text file.
        skip_blank (bool): Whether to skip blank lines.

    Returns:
        lines (list): List of lines.
    """

    lines = []
    with open(path, 'r', encoding="utf-8") as file:
        for line in file:
            if skip_blank and line == '\n':
                continue
            lines.append(line.strip())
    return lines

def save_txt(txt, path, mode="w", skip_blank=True):
    """
    Saves a list of lines to a text file.

    Parameters:
        txt (list): List of lines.
        path (str): Path to text file.
        mode (str): Mode to open file.
        skip_blank (bool): Whether to skip blank lines.

    Returns:
        None
    """
    
    with open(path, mode=mode, encoding='utf-8') as file:
        for line in txt:
            if skip_blank and line == '':
                continue
            file.write(line + '\n')
