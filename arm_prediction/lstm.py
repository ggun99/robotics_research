import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('device : ', device)
# 하이퍼 파라미터

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset

train_dataset = torchvision.datasets.MNIST(root = '../../data', 
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root = '../../data', 
                                           train=False,
                                           transform=transforms.ToTensor())

# 데이터 로더

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size = batch_size,
                                           shuffle=False)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__() # 상속한 nn.Module에서 RNN에 해당하는 init 실행
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x): 
        # input x : (BATCH, LENGTH, INPUT_SIZE) 입니다 (다양한 length를 다룰 수 있습니다.).
        # 최초의 hidden state와 cell state를 초기화시켜주어야 합니다.
        # 배치 사이즈는 가변적이므로 클래스 내에선 표현하지 않습니다.
        # 만약 Bi-directional LSTM이라면 아래의 hidden and cell states의 첫번째 차원은 2*self.num_layers 입니다. 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # (BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # hidden state와 동일

        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0)) # output : (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE) tensors. (hn, cn)은 필요 없으므로 받지 않고 _로 처리합니다. 

        # 마지막 time step(sequence length)의 hidden state를 사용해 Class들의 logit을 반환합니다(hidden_size -> num_classes). 
        out = self.fc(out[:, -1, :])
        return out
    
    # 모델 할당 후 학습
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device) # 

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss() # 분류
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습

total_step = len(train_loader) # 배치 개수
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
       images = images.reshape(-1, sequence_length, input_size).to(device) # (BATCH(100), 1, 28, 28) -> (BATCH(100), 28, 28)
       labels = labels.to(device) # Size : (100)

       # 순전파
       outputs = model(images)
       loss = criterion(outputs, labels)

       # 역전파 & 최적화
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if (i+1) % 100 == 0: 
         print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
             epoch+1, num_epochs, i+1, total_step, loss.item()))
         
 # 모델 평가

model.eval() # Dropout, Batchnorm 등 실행 x
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1) # logit(확률)이 가장 큰 class index 반환
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))