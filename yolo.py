import torch
import torchvision as tv
import torch.nn.functional as F
import numpy as np
import time

batch_size = 64 #calcular batch segun memoria de gpu


"""
trn_load = torch.utils.data.DataLoader( trn_data, batch_size=B,
 shuffle=True, num_workers=2)

tst_load = torch.utils.data.DataLoader( tst_data, batch_size=B,
 shuffle=False, num_workers=2)

C = len(classes)
# idata = iter( tst_load)
# image, label = next( idata)
"""
class YoloBase(torch.nn.Module):
    
    def __init__( _ ): 
        
         super().__init__()

         lrelu =  torch.nn.LeakyReLU(0.1, inplace = True)
         conv = torch.nn.Conv2d
         maxpool = torch.nn.MaxPool2d(2, stride = 2)

         _.model = torch.nn.Sequential(

            #primer bloque
            conv(3, 64, 7, stride = 2, padding = 3),
            lrelu,
            maxpool,

            #segundo bloque
            conv(64, 192, 3),
            lrelu,
            maxpool,

            #tercer bloque
            conv(192, 128, 1, padding = 1),
            lrelu,
            conv(128, 256, 3, padding = 1),
            lrelu,
            conv(256, 256, 1, padding = 1),
            lrelu,
            conv(256, 512, 3, padding = 1),
            lrelu,
            maxpool,

            #cuarto bloque
            conv(512, 256, 1, padding = 1),
            lrelu,
            conv(256, 512, 3, padding = 1),
            lrelu,
            conv(512, 256, 1, padding = 1),
            lrelu,
            conv(256, 512, 3, padding = 1),
            lrelu,
            conv(512, 256, 1, padding = 1),
            lrelu,
            conv(256, 512, 3, padding = 1),
            lrelu,
            conv(512, 256, 1, padding = 1),
            lrelu,
            conv(256, 512, 3, padding = 1),
            lrelu,
            conv(512, 512, 1, padding = 1),
            lrelu,
            conv(512, 1024, 3, padding = 1),
            lrelu,
            maxpool,

            #quinto bloque
            conv(1024, 512, 1, padding = 1),
            lrelu,
            conv(512, 1024, 3, padding = 1),
            lrelu,
            conv(1024, 512, 1, padding = 1),
            lrelu,
            conv(512, 1024, 3, padding = 1),
            lrelu
        )
  
    def setForPretraining( _, output_classes):
        _.pretraining = True
        _.fc = torch.nn.Linear(1024, output_classes)


    def setForTraining( _ ):
        _.pretraining = False

    def forward(_, x):    

        y = _.model(x)
        if _.pretraining:
            y = F.avg_pool2d(y, (y.size(2), y.size(3)))
            y = y.squeeze()
            y = _.fc(y)
            #y = F.softmax(_.fc(y), dim = 0)

        return y

def timeit(f, arg):
    b = time.time()
    f(arg)
    e = time.time()
    return (e-b)


device = torch.device('cuda')
T = 5

X = torch.rand(512, 3, 448, 448)
Y_c = torch.randint(10, (512, 1))
Y = []
for y_c in Y_c:
    l = np.zeros((10, 1))
    l[y_c] = 1
    Y.append(l)

Y = torch.tensor(Y)
Y = Y.view(512, 10)

model_cuda = YoloBase()
model_cuda.setForPretraining(10)
model_cuda.cuda()


class Mem:
    def __call__(_):
        return torch.cuda.memory_allocated() / (1024*1024)

mem = Mem()

images = X[0:64].to(device)
y = Y[0:64].to(device)

optim = torch.optim.Adam( model_cuda.parameters())
costf = torch.nn.CrossEntropyLoss() 


