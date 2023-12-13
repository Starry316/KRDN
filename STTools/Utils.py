import os

from STTools.Logger import STLogger


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
def getClazzName(obj):
    return str(type(obj)).split('\'')[1]

def getNetworkName(obj):
    return getClazzName(obj).split('.')[1]

def createDirs(saveDir, saveName):
    STLogger.info(f'Creating directories for {saveDir} {saveName}')
    os.makedirs(os.path.join('logs', saveDir, saveName), exist_ok=True)
    os.makedirs(os.path.join('results', saveDir, saveName), exist_ok=True)
    os.makedirs(os.path.join('outputs', saveDir, saveName), exist_ok=True)