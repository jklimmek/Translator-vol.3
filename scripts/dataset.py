from torch.utils.data import Dataset


class TranslatorDataset(Dataset):
    def __init__(self, src, tgt):
        super().__init__()
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        return self.src[index], self.tgt[index]
    