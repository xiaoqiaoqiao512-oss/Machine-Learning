import torch
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    """
    一个最简单的数据集。
    我们直接随机生成很多 token 序列。
    例如：
        [15, 22, 81, 7, 55]
    输入(input)：
        [15, 22, 81, 7]
    标签(target)：
        [22, 81, 7, 55]
    也就是 GPT 最经典的 Next Token Prediction。
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 1000,
    ):
        super().__init__()
        self.data = []
        for _ in range(num_samples):
            # 生成长度为 seq_len + 1 的随机序列
            tokens = torch.randint(
                low=0,
                high=vocab_size,
                size=(seq_len + 1,),
                dtype=torch.long,
            )
            self.data.append(tokens)
    def __len__(self):
        """
        Dataset 一共有多少条数据。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        返回一条训练样本。
        input:
            前 seq_len 个 token
        target:
            后 seq_len 个 token
        """
        tokens = self.data[index]
        x = tokens[:-1]
        y = tokens[1:]
        return x, y