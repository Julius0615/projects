"""Train the model"""

import argparse
import logging
import os
import csv

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import utils
from models import metrics,loss_fn
import models.data_loader as data_loader
from evaluate import evaluate
import  models
import  torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument('--data_csv', default='Label.csv', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/test', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument("--model", default="VGG")


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    mse = nn.MSELoss()
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):
            train_batch, labels_batch,target_landmark,landmark_image = batch['image'],batch['labels'],batch['landmark'],batch['landmarkimage']
            train_batch = torch.Tensor(train_batch)
            labels_batch = torch.LongTensor(labels_batch)
            target_landmark = torch.FloatTensor(target_landmark)
            # move to GPU if available
            if params.cuda:
                # train_batch, labels_batch = train_batch.cuda(async = True), labels_batch.cuda(async = True)
                # syntx err ???
                # cuda(device=None, non_blocking=False) -> Tensor
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
                target_landmark = target_landmark.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            target_landmark = Variable(target_landmark)
            # compute model output and loss
            output_batch,landmark = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)+mse(landmark.view(-1),target_landmark.view(-1))

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    mse = nn.MSELoss()
    #best_f1_score =0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        val_acc = val_metrics['accuracy']
        #val_f1_score = val_metrics['f1_score']
        
        is_best = val_acc>=best_val_acc
        #is_best = val_f1_score > best_f1_score
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            #best_f1_score= val_f1_score
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            
            labels_print=[]
            outputs_print=[]
            
            #save best lables and outputs in order to perform error analysis
            for batch in val_dataloader:
                data_batch, labels_batch =batch['image'], batch['labels']
                # move to GPU if available
                if params.cuda:
                    # data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
                    data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()                    
                # fetch the next evaluation batch
                data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
                # compute model output
                output_batch,landmark_batch = model(data_batch)
        
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                
                #labels
                labels_print.append(labels_batch)
                labels_p=np.concatenate(labels_print)
                #outputs
                outputs_b = np.argmax(output_batch, axis=1)
                outputs_print.append(outputs_b)
                outputs_p=np.concatenate(outputs_print)
            
            csvfile1=os.path.join(model_dir,'labels.csv')
            with open(csvfile1, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                for val in labels_p:
                    writer.writerow([val])
                    
            csvfile2=os.path.join(model_dir,'outputs.csv')
            with open(csvfile2, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                for val in outputs_p:
                    writer.writerow([val])
            
            

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    datas = {"image":[],"label":[]}
    with open(args.data_csv) as reader:
        for i in reader:
            data = i.strip().split(",")
            datas['image'].append(data[1])
            datas['label'].append(data[2])
    print(set(datas['label']))
    X_train,X_test,y_train,y_test = train_test_split(datas["image"],datas["label"],test_size=0.2,random_state=1)
    datas["train"]=(X_train,y_train)
    datas['val'] = (X_test,y_test)
    # Define the model and optimizer


    model_class = getattr(models, args.model)
    if args.model == "resnext50_32x4d":
        model = model_class().cuda() if params.cuda else model_class()
    else:
        model = model_class(params).cuda() if params.cuda else model_class(params)
    print(model)
    params.resize = model.input_size
    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', "val"], datas, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")


    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate,weight_decay=0.01)

    # fetch loss function and metrics
    loss_fn = nn.CrossEntropyLoss()
    metrics = metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)