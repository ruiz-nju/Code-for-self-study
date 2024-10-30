import torch
import torch.nn as nn
import torch.optim as optim

char_list = ["h", "e", "l", "o"]
num_embedding = len(char_list)  # size of the dictionary of embeddings

char_to_idx = {ch: idx for idx, ch in enumerate(char_list)}
idx_to_char = {idx: ch for idx, ch in enumerate(char_list)}
x = [char_to_idx[ch] for ch in "hello"]


class charRNN(nn.Module):
    def __init__(self, num_embedding, embedding_dim, hidden_size, device) -> None:
        super().__init__()
        self.device = device
        self.num_embedding = num_embedding  # 词典大小
        self.embedding_dim = embedding_dim  # 词向量维度
        self.embed = nn.Embedding(self.num_embedding, self.embedding_dim).to(
            self.device
        )  # 将单词转换为词嵌入
        self.hidden_size = hidden_size  # 隐层大小
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_size, batch_first=True).to(
            self.device
        )
        self.fc = nn.Linear(self.hidden_size, self.num_embedding).to(self.device)

    def forward(self, x, hidden):
        x = x.to(self.device)
        hidden = hidden.to(self.device)

        x = self.embed(x)  # (1, sequence_length, embedding_dim)
        output, hidden = self.rnn(x, hidden)  # (1, sequence_length, embedding_dim)
        output = output[:, -1, :]  # (1, embedding_dim), 只需要最后一个时间步的预测
        output = self.fc(output)  # (1, num_embedding)
        return output, hidden


# 超参数设置
embedding_dim = 128
hidden_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 50
lr = 1e-2

# 初始化模型
model = charRNN(num_embedding, embedding_dim, hidden_size, device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(num_epochs):
    # (batch_size, sequence_length, hidden_size)
    hidden = torch.randn(1, 1, hidden_size)
    loss = 0
    for i in range(len(x) - 1):
        input_sequence = torch.LongTensor([x[i]]).unsqueeze(0)  # (1,sequence_length)
        target_sequence = torch.LongTensor([x[i + 1]]).to(device)  # (1,)
        output, hidden = model(input_sequence, hidden)
        loss += criterion(output, target_sequence)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    hidden = torch.randn(1, 1, hidden_size)
    input = torch.LongTensor([x[0]]).unsqueeze(0)
    predict = []

    for _ in range(len(x) - 1):
        output, hidden = model(input, hidden)  # (1, num_embedding)
        output = torch.softmax(output, dim=1)
        label = torch.argmax(output, dim=1)
        predict.append(label.item())
        input = label.unsqueeze(0)

print("Predicted sequence:", "".join([idx_to_char[idx] for idx in predict]))
