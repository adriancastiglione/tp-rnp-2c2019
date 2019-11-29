import torch
import torchvision as tv


B = 100 #calcular batch segun memoria de gpu



trn_load = torch.utils.data.DataLoader( trn_data, batch_size=B,
 shuffle=True, num_workers=2)

tst_load = torch.utils.data.DataLoader( tst_data, batch_size=B,
 shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
 'dog', ' frog', 'horse', 'ship', 'truck')  #mirar dataset imagenet

C = len(classes)
# idata = iter( tst_load)
# image, label = next( idata)
class ConvNet( torch.nn.Module):
    
    def __init__( _, num_classes): # O = | (I + 2p - k)/s + 1 |
        
         super().__init__() # Tambien pueden ser tuplas.
            
        #primer bloque
        _.conv1 = torch.nn.Conv2d( 3, 64, kernel_size=7, stride=2, padding=0)
        _.mpool1 = torch.nn.MaxPool2d( kernel_size=2, stride=2)
        
        #segundo bloque
        _.conv2 = torch.nn.Conv2d( 64, 192, kernel_size=3)
        _.mpool2 = torch.nn.MaxPool2d( kernel_size=2, stride=2)
        
        #tercer bloque
        _.conv3 = torch.nn.Conv2d(192, 128, kernel_size=1)
        _.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3)
        _.conv5 = torch.nn.Conv2d(256, 256, kernel_size=1)
        _.conv6 = torch.nn.Conv2d(256, 512, kernel_size=3)
        _.mpool3 = torch.nn.MaxPool2d( kernel_size=2, stride=2)
        
        #cuarto bloque
        _.conv7 = torch.nn.Conv2d(512, 256, kernel_size = 1)
        _.conv8 = torch.nn.Conv2d(256, 512, kernel_size = 3)
        _.conv9 = torch.nn.Conv2d(512, 256, kernel_size = 1)
        _.conv10 = torch.nn.Conv2d(256, 512, kernel_size = 3)
        _.conv11 = torch.nn.Conv2d(512, 256, kernel_size = 1)
        _.conv12 = torch.nn.Conv2d(256, 512, kernel_size = 3)
        _.conv13 = torch.nn.Conv2d(512, 256, kernel_size = 1)
        _.conv14= torch.nn.Conv2d(256, 512, kernel_size = 3)
        _.conv15 = torch.nn.Conv2d(512, 512, kernel_size = 1)
        _.conv16 = torch.nn.Conv2d(512, 1024, kernel_size = 3)
        _.mpool4 = torch.nn.MaxPool2d( kernel_size=2, stride=2)
        
        #quinto bloque
        
        _.conv17 = torch.nn.Conv2d(1024, 512, kernel_size = 1)
        _.conv18 = torch.nn.Conv2d(512, 1024, kernel_size = 3)
        _.conv19 = torch.nn.Conv2d(1024, 512, kernel_size = 1)
        _.conv20 = torch.nn.Conv2d(512, 1024, kernel_size = 3)
        _.conv21 = torch.nn.Conv2d(1024, 1024, kernel_size = 3)
        _.conv22 = torch.nn.Conv2d(1024, 1024, kernel_size = 3, stride = 2)
        
        #sexto bloque
        
        _.conv23 = torch.nn.Conv2d(1024, 1024, kernel_size = 3)
        _.conv24 = torch.nn.Conv2d(1024, 1024, kernel_size = 3)
        
        #septimo bloque
        _.linear1 = torch.nn.Linear(1024*7*7, 4096)
        _.linear2 = torch.nn.Linear(4096, 7*7*30)
        
        
def forward(_, x):
    
    #primer bloque
    h1 = _.conv1(x).LeakyReLU()
    m1 = _.mpool1(h1).LeakyReLU()
    
    #segundo bloque
    h2 = _.conv2(m1).LeakyReLU()
    m2 = _.mpool2(h2).LeakyReLU()
    
    #tercer bloque
    h3 = _.conv3(m2).LeakyReLU()
    h4 = _.conv4(h3).LeakyReLU()
    h5 = _.conv5(h4).LeakyReLU()
    h6 = _.conv6(h5).LeakyReLU()
    m3 = _.mpool3(h6).LeakyReLU()
    
    #cuarto bloque
    h7 = _.conv7(m3).LeakyReLU()
    h8 = _.conv8(h7).LeakyReLU()
    h9 = _.conv9(h8).LeakyReLU()
    h10 = _.conv10(h9).LeakyReLU()
    h11 = _.conv11(h10).LeakyReLU()
    h12 = _.conv12(h11).LeakyReLU()
    h13 = _.conv13(h12).LeakyReLU()
    h14 = _.conv14(h13).LeakyReLU()
    h15 = _.conv15(h14).LeakyReLU()
    h16 = _.conv16(h15).LeakyReLU()
    m4 = _.mpool4(h16).LeakyReLU()
    
    
    #quinto bloque
    h17 = _.conv17(m4).LeakyReLU()
    h18 = _.conv18(h17).LeakyReLU()
    h19 = _.conv19(h18).LeakyReLU()
    h20 = _.conv20(h19).LeakyReLU()
    h21 = _.conv21(h20).LeakyReLU()
    h22 = _.conv22(h21).LeakyReLU()
    
    #sexto bloque
    h23 = _.conv23(h22).LeakyReLU()
    h24 = _.conv24(h23).LeakyReLU()
    
    #septimo bloque
    h24 = h24.view(-1, 7*7*1024)
    l1 = _.linear1(h24).LeakyReLU()
    l2 = _.linear2(l1)
    
    y = l2.view(7, 30, 7)
    
    return y

