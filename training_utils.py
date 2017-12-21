from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from sklearn import svm
from copy import deepcopy
from sklearn.metrics import roc_auc_score

plt.ion()   # interactive mode


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        pred_labels = np.array([])
        true_labels = np.array([])

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data['image'], data['class1']

                inputs, labels = Variable(inputs), Variable(labels)
                labels = labels.long()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                #print(outputs), print(labels)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                pred_labels=np.concatenate((pred_labels,preds.numpy() ))
                true_labels=np.concatenate((true_labels,labels.data.numpy() ))
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_auc = roc_auc_score(true_labels, pred_labels)

            print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_auc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return



def test_model(model, criterion,dataloaders,dataset_sizes):
    since = time.time()
    correct = 0.0
    total = 0.0
    loss_final =0.0

    phase = 'test'
    pred_labels = np.array([])
    true_labels = np.array([])
    # Iterate over data.
    for data in dataloaders['test']:
        # get the inputs
        inputs, labels = data['image'], data['class1']

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        #print(outputs), print(labels)
        loss = criterion(outputs, labels)


        # statistics
        loss_final += loss.data[0]
        correct += torch.sum(preds == labels.data)
        pred_labels=np.concatenate((pred_labels,preds.numpy() ))
        true_labels=np.concatenate((true_labels,labels.data.numpy() ))

    loss_final = loss_final / dataset_sizes[phase]
    acc = correct / dataset_sizes[phase]
    epoch_auc = roc_auc_score(true_labels, pred_labels)

    print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
                phase, loss_final, acc, epoch_auc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
def train_meta_model(model1, model2, dataloaders, dataset_sizes, model_dir, num_models):
    since = time.time()
    
    ii, jj, kk = num_models
    feat_list=np.zeros((dataset_sizes['val'],len(ii)*jj*kk, 2))
    labels_list=np.zeros(dataset_sizes['val'])
    
    phase='val'
   
    # get all the features extracted from the models
    for i,data in enumerate(dataloaders['val']):
        # get the inputs
        inputs, labels = data['image'], data['class1']
        batch_size = data['image'].shape[0]
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        # forward
        count = 0
        for i in ii:
            if i==0:
                model = deepcopy(model1)
            else:
                model = deepcopy(model2)
            for j in range(jj):
                for k in range(kk):
                    model.load_state_dict(torch.load(model_dir+str(i+1)+str(j+1)+str(k)+'.pt'))
                    outputs = model(inputs)
                    feat_list[i*batch_size:(i+1)*batch_size, count,:]=outputs.data.numpy()
                    count +=1
   
        labels_list[i*batch_size:(i+1)*batch_size]=labels.data.numpy()
 
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    XX = np.concatenate((feat_list[:,:,0],feat_list[:,:,1]),axis=1 )
    clf.fit(XX, labels_list)  

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    return clf


def test_meta_model(model1, model2, dataloaders, dataset_sizes, model_dir, num_models, clf):
    since = time.time()
    
    ii, jj, kk = num_models
    correct = 0
    phase = 'test'
    pred_labels = np.array([])
    true_labels = np.array([])

    for data in dataloaders[phase]:
        inputs, labels = data['image'], data['class1']
        batch_size = data['image'].shape[0]
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        feat_list=np.zeros((batch_size,len(ii)*jj*kk,2))
        count = 0
        for i in ii:
            if i==0:
                model = deepcopy(model1)
            else:
                model = deepcopy(model2)
            for j in range(jj):
                for k in range(kk):
                    model.load_state_dict(torch.load(model_dir+str(i+1)+str(j+1)+str(k)+'.pt'))
                    outputs = model(inputs)
                    feat_list[:, count,:]=outputs.data.numpy()
                    count +=1
        XX = np.concatenate((feat_list[:,:,0],feat_list[:,:,1]),axis=1 )
        preds = clf.predict(XX)

        correct += np.sum(preds == labels.data.numpy())
        pred_labels=np.concatenate((pred_labels,preds ))
        true_labels=np.concatenate((true_labels,labels.data.numpy() ))

    acc = correct / dataset_sizes[phase]
    epoch_auc = roc_auc_score(true_labels, pred_labels)

    print('{}  Acc: {:.4f} AUC: {:.4f}'.format(
        phase, acc, epoch_auc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return acc

def train_model_epochs(model, model_num, model_dir, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    
    model_list = []
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data['image'], data['class1']

                inputs, labels = Variable(inputs), Variable(labels)
                labels = labels.long()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                #print(outputs), print(labels)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            best_model_wts = model.state_dict()
            model_list.append(best_model_wts)
            torch.save(best_model_wts, model_dir+str(model_num)+str(1)+str(epoch)+'.pt')
            
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return model_list

def test_ensamble_model(model1, model2 ,dataloaders, dataset_sizes, model_dir, num_model): 
    # num_models = (i, j, k)
    since = time.time()
    
    ii, jj, kk = num_model
    
    pred_labels = np.array([])
    true_labels = np.array([])    
    correct = 0
    phase = 'test'
    for data in dataloaders[phase]:
        inputs, labels = data['image'], data['class1']
        batch_size = data['image'].shape[0]
        inputs, labels = Variable(inputs), Variable(labels)
        labels = labels.long()
        feat_list=np.zeros((batch_size,len(ii)*jj*kk,2))
        count = 0
        for i in ii:
            if i==0:
                model = deepcopy(model1)
            else:
                model = deepcopy(model2)
            for j in range(jj):
                for k in range(5,kk):
                    model.load_state_dict(torch.load(model_dir+str(i+1)+str(j+1)+str(k)+'.pt'))
                    outputs = model(inputs)
                    feat_list[:, count,:]=outputs.data.numpy()
                    count +=1
        feat_list = np.sum(feat_list, axis=1)/feat_list.shape[1]     
        preds = np.argmax(feat_list, axis = 1)
        correct += np.sum(preds == labels.data.numpy())
        pred_labels=np.concatenate((pred_labels,preds ))
        true_labels=np.concatenate((true_labels,labels.data.numpy() ))

    acc = correct / dataset_sizes[phase]
    epoch_auc = roc_auc_score(true_labels, pred_labels)

    print('{}  Acc: {:.4f} AUC: {:.4f}'.format(
        phase, acc, epoch_auc))

    print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return acc


