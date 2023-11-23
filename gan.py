import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import scipy.stats as st
import torch.optim as optim
import torch
torch.set_default_tensor_type(torch.DoubleTensor)




#真实数据
# 这是生成真实的数据, 被用于模仿


data = pd.read_csv(
    r'D:\PycharmProjects\pythonProject2\sci2\数据\zaomindata.csv', header=None)
# print(data)
y_label = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y_label)
labeldata = np.array(le.transform(y_label)).reshape(-1, 1)
columnstestdata = data.shape[1]-1
testdata = pd.concat([data.iloc[:, 0:columnstestdata], pd.DataFrame(labeldata)], axis=1)
testdata.columns = [i for i in range(0, columnstestdata + 1)]

'-----获取某一类数据-----'
# mindata=pd.DataFrame(testdata.loc[testdata.iloc[:,-1]==1,[1]])
mindata=pd.DataFrame(testdata.loc[testdata.iloc[:,-1]==0,0:columnstestdata-1])
# print(mindata)
# majdata=pd.DataFrame(testdata.loc[testdata.iloc[:,-1]==0,2])
real_data=torch.Tensor(mindata.values)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

'---创建生成器与判别器-----'
'判别器'
class Discriminator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Discriminator, self).__init__()
        self.disc=nn.Sequential(nn.Linear(input_size,hidden_size),
                                nn.Linear(hidden_size, 512),
                                nn.LeakyReLU(0.1),
                                nn.Linear(512, hidden_size),
                                nn.LeakyReLU(0.1),
                                nn.Linear(hidden_size,output_size),
                                nn.Sigmoid()
                                )

    def forward(self,disc_data):
        dic_output=self.disc(disc_data)
        return dic_output
'生成器'
class Generator(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        '''input_size 是指输入到生成器数据的维度，可以自定义，
        output_size是指输出到判别器的维度必须和源数据的维度相同，因为此时判别器需要判断是真数据还是假数据'''
        super(Generator, self).__init__()
        self.gen=nn.Sequential(nn.Linear(input_size,hidden_size),
                               nn.Linear(hidden_size,128),
                               nn.Linear(128, hidden_size),
                               nn.LeakyReLU(0.1),
                               nn.Linear(hidden_size,output_size),
                               nn.Tanh()  #数据归一化处理
                               )

    def forward(self,gen_data):
        gen_data_output=self.gen(gen_data)
        return gen_data_output


'----概率密度图---'
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf

#测试生成器与判别器
# tensor1=torch.ones(10,2).view(-1)
# print(tensor1)
# print(tensor1.shape)
# model1=Generator(64,1)
# print(model1(tensor1))

'---规定参数----'
'----'
learning_rate=0.01
G_input_size=30
G_hidden_size=64
G_output_size=columnstestdata
gen=Generator(input_size=G_input_size,hidden_size=G_hidden_size,output_size=G_output_size).to(device)
'-------'
D_input_size=columnstestdata
D_hidden_size=128
D_output_size=1
disc=Discriminator(input_size=D_input_size,hidden_size=D_hidden_size,output_size=D_output_size).to(device)

##测试
# torch.randn((mindata.shape[0],G_input_size))
# print(torch.randn(30,1))
# print(torch.randn(30,1).view(-1,30))
# print(gen(torch.randn((mindata.shape[0],G_input_size))))


'---定义优化算法---'
optim_gen=optim.Adam(gen.parameters(),lr=0.001,betas=(0.5,0.9))
optim_disc=optim.Adam(disc.parameters(),lr=0.0001,betas=(0.5,0.9))

'---定义损失函数---'
criterion=nn.BCELoss()   #采样上述损失函数

'''----参数迭代----'''
epochs=500
batch=mindata.shape[0]
# print(batch)
batches=len(mindata)//batch
# print(batches)
loss_G=[]
loss_D=[]
loss_D1=[]
G_mean=[]
G_std=[]
loss_Real=[]
loss_Fake=[]
for epoch in range(epochs):
    '''对数据进行切分，每一次得到batch个数据'''
    '''训练分类器'''
    for i in range(20):
        #train on generator
        # stat=i*batch
        # end=stat+batch
        # '''判别器的损失'''
        # x_real_data=real_data[stat:end]
        optim_disc.zero_grad()
        disc_real_data=disc(real_data)
        errorD_real_data=criterion(disc_real_data,torch.ones_like(disc_real_data))
        loss_Real.append(errorD_real_data.data.numpy())
        errorD_real_data.backward()

        #train on fake
        noise=torch.randn((batch,G_input_size))
        # noise=torch.normal(0.5,0.005,size=(batch,G_input_size))
        # noise = torch.normal(mean=0.54,std=0.13,size=(batch, G_input_size))
        gen_data1=gen(noise)
        gen_data2=disc(gen_data1.detach())
        errorD_fake_data=criterion(gen_data2, torch.zeros_like(gen_data2))
        error_sum= (errorD_real_data + errorD_fake_data)
        loss_D1.append(error_sum.data.numpy())
        loss_Fake.append(errorD_fake_data.data.numpy())
        errorD_fake_data.backward()
        optim_disc.step()
    loss_D.append(loss_D1[-1])


    '''生成器的损失'''
    ##生成器的反向传播
    optim_gen.zero_grad()
    noise = torch.randn((batch, G_input_size))
    gen_data4 = gen(noise)
    gen_data3=disc(gen_data4)
    # gen_data3 = disc(gen_data4)
    errorG_real_data=criterion(gen_data3,torch.ones_like(gen_data3))
    loss_G.append(errorG_real_data.data.numpy())
    errorG_real_data.backward()
    optim_gen.step()

    with torch.no_grad():
        G_mean.append(np.mean(gen_data4.data.numpy()))
        G_std.append(np.std(gen_data4.data.numpy()))

    if epoch % 10==0:
        print("Epoch: {}, loss_D:{} ,loss_G:{} ,mean:{},std:{}"
              .format(epoch,loss_D[-1],loss_G[-1],G_mean[-1],G_std[-1]))

print(disc.state_dict().keys())  # 输出模型参数名称
#保存模型参数到路径"./data/model_parameter.pkl"
torch.save(disc.state_dict(), "D:\PycharmProjects\pythonProject2\sci3\程序/model_parameter.pth")


'''loss函数画图'''
plt.plot(loss_G,c='green',label='loss G')
plt.plot(loss_D,c='red',label='loss D')
plt.title('Loss Function')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
# plt.savefig('D:/PycharmProjects/pythonProject2/生成对抗网络论文/画图程序/loss.eps',
#             format='eps',dpi=1000,bbox_inches='tight')
plt.show()



