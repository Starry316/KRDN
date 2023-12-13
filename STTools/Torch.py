import collections
import os

import torch
from STTools.Logger import STLogger
from STTools.Utils import getNetworkName, getClazzName

def loadModel(path, network):
    print(path)
    model = torch.load(path)
    if 'networkState' in model.keys():
        network.load_state_dict(model['networkState'])
    else:
        network.load_state_dict(model)

def saveModel(path, network, optimizer, scheduler, epoch=0):
    torch.save({
        'epoch': epoch,
        'networkState': network.state_dict(),
        'optimizerState': optimizer.state_dict(),
        # 'schedulerState': scheduler.state_dict(),
    }, path)

def restoreModel(path, network, optimizer, scheduler):
    model = torch.load(path)
    epoch = 1
    if 'epoch' in model.keys():
        epoch = model['epoch']
        STLogger.info(f'Restore model of epoch: {epoch}')
    else:
        STLogger.warning('Loaded weight dosent contain a epoch information!')

    if 'networkState' in model.keys():
        network.load_state_dict(model['networkState'])
        STLogger.info(f'Network State restored')
    else:
        STLogger.warning('Loaded weight dosent contain a networkState!')

    if 'optimizerState' in model.keys():
        optimizer.load_state_dict(model['optimizerState'])
        STLogger.info(f'Optimizer State restored')
    else:
        STLogger.warning('Loaded weight dosent contain a optimizerState!')

    if 'schedulerState' in model.keys():
        scheduler.load_state_dict(model['schedulerState'])
        STLogger.info(f'Scheduler State restored')
    else:
        STLogger.warning('Loaded weight dosent contain a schedulerState!')
    return epoch

def saveTrainingConfig(path, args, network, optimizer):
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write(f'{arg} = {attr}\n')
        file.write(f'network = {getNetworkName(network)}\n')
        file.write(f'optimizer = {getClazzName(optimizer)}\n')
