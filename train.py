import torch as t
from torch.nn import functional as F
from torch.autograd import Variable
from tools.draw_and_save import curve_draw
from tools.dloss import dloss
from tools.ListToTensor import ListToTensor
from tools.dataenhance import *
from ResNet import *
from tools.dataloader import *
from train_once import *
from tools.evaluate import *
from hyper_parameters import *
from scripe.datadivision import data_divide
#from ResNet import resnet18, resnet34, resnet50, resnet101, resnet152

def train(epochs=120, 
          init_lr=0.001, 
          weight_decay=1e-8, 
          model_num=1, 
          batch_size=64, 
          train_dir=train_dir, 
          test_dir=test_dir, 
          log_dir=log_dir, 
          ):
    #divide data
    #data_divide(data_dir=data_dir, train_dir=train_dir, test_dir=test_dir)
    #load data
    print("data loading......\n")
    transform = enhance_transform()
    transform_std = transform_standard()
    trainset = DataClassify(train_dir, transforms=transform)
    testset = DataClassify(test_dir, transforms=transform_std)
    train_length = len(trainset)
    test_length = len(testset)
    data_loader_train = t.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    data_loader_test = t.utils.data.DataLoader(testset, batch_size, shuffle=False)
    print("loading complete\n")

    if model_num==0:
        exit(0)
    elif model_num==1:
        net = resnet18()
    elif model_num==2:
        net = resnet34()
    elif model_num==3:
        net = resnet50()
    elif model_num==4:
        net = resnet101()
    elif model_num==5:
        net = resnet152()

    cost = t.nn.CrossEntropyLoss()
    train_loss_list = []
    train_accurate_list = []
    test_loss_list = []
    test_accurate_list = []

    for epoch in range(epochs):
        print("epoch " + str(epoch+1) + " start training...\n")
        net.train()
        learning_rate = dloss(train_loss_list, init_lr, lr_coefficient, init_lr)
        optimizer = t.optim.Adam(list(net.parameters()), lr=learning_rate, weight_decay=weight_decay)

        run_loss, corr = train_once(data_loader_train, net, optimizer, cost)
        train_loss_list.append(run_loss/train_length)
        train_accurate_list.append(corr/train_length)

        print('epoch %d, training loss %.6f, training accuracy %.4f ------\n' %(epoch+1, run_loss/train_length, corr/train_length))
        print("epoch " + str(epoch+1) + " finish training\n")
        print("-----------------------------------------------\n")
        print("epoch " + str(epoch+1) + " start testing...\n")

        net.eval()
        test_corr = evaluate(net, data_loader_test)
        test_accurate_list.append(test_corr/test_length)
        print('epoch %d, testing accuracy %.4f ------\n' %(epoch+1, test_corr/test_length))
        print("epoch " + str(epoch+1) + " finish testing\n")
        print("-----------------------------------------------\n")

    curve_draw(train_loss_list, train_accurate_list, test_accurate_list, log_dir)


if __name__ == '__main__':
    train(epochs=epochs,
           model_num=model_num,
           train_dir=train_dir, 
           test_dir=test_dir,
           log_dir=log_dir)