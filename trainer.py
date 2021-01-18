import torch
import time
from utils import log_gaussian_loss

class RegressionTrainer(object) :
    def __init__(self,name) :
        self.name = name

    def train_model(self,model,criterion,optimizer,scheduler,num_epochs,dataloaders,is_cuda=False, verbose=True) :
        since = time.time()
        best_model_wts = model.state_dict()
        best_loss = 1e8
        for epoch in range(num_epochs) :
            if epoch % 100 == 0 and verbose == True:
                vbs = True
            else : 
                vbs = False
            if  vbs == True :
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
            train_loss = self.fit('train',epoch,model,criterion,optimizer,dataloaders['train'],is_cuda,vbs)
            valid_loss = self.fit('valid',epoch,model,criterion,optimizer,dataloaders['valid'],is_cuda,vbs)
            if valid_loss < best_loss :
                best_mode_wts = model.state_dict()
                best_loss = valid_loss
            if vbs == True :
                print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:4f}'.format(best_loss))

        model.load_state_dict(best_model_wts)
        return model

    def fit(self,phase,epoch,model,criterion,optimizer,data_loader,is_cuda,verbose=True) :
        if phase == 'train':
            model.train()
        if phase == 'validation':
            model.eval()
        running_loss = 0.0
        for batch_idx , sample in enumerate(data_loader):
            x = sample['input']
            y = sample['output']
            if is_cuda:
                x,y = x.cuda(), y.cuda()
            if phase == 'train':
                optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred,y)
            
            running_loss += loss.data

            if phase == 'train':
                loss.backward()
                optimizer.step()
        
        epoch_loss = running_loss.item()/len(data_loader.dataset)
        if verbose == True :
            print('{} loss is {:.4f}'.format(phase, epoch_loss)) 
        return epoch_loss


class RegDensityTrainer(RegressionTrainer) :
    def __init__(self,name) :
        super().__init__(name)

    def fit(self,phase,epoch,model,criterion,optimizer,data_loader,is_cuda,verbose=True) :
        if phase == 'train':
            model.train()
        if phase == 'validation':
            model.eval()
        running_loss = 0.0
        for batch_idx , sample in enumerate(data_loader):
            x = sample['input']
            y = sample['output']
            if is_cuda:
                x,y = x.cuda(), y.cuda()
            if phase == 'train':
                optimizer.zero_grad()

            y_pred, y_sigma = model(x)
            loss = log_gaussian_loss(y_pred,y_sigma,y,dim=1)
            
            running_loss += loss.data

            if phase == 'train':
                loss.backward()
                optimizer.step()
        
        epoch_loss = running_loss.item()/len(data_loader.dataset)
        if verbose == True :
            print('{} loss is {:.4f}'.format(phase, epoch_loss)) 
        return epoch_loss





