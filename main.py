import os
from numpy import loadtxt
import keras
from keras.models import load_model
from atten import *
from tensorflow.keras.utils import CustomObjectScope
# from params import parseargs
from datamanager import *
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.pyplot import cm
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

def parseargs():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument(
    '--num_classes',
    type=int,
    default=0,
    help='Number of classes (families).'
    )
    parser.add_argument(
    '--max_seq_len',
    type=int,
    default=0,
    help='Length of input sequences.'
    )
    parser.add_argument(
    '--max_smi_len',
    type=int,
    default=0,
    help='Length of input sequences.'
    )
    parser.add_argument(
    '--binary_th',
    type=float,
    default=0.0,
    help='Threshold to split data into binary classes'
    )
    parser.add_argument(
    '--checkpoint_path',
    type=str,
    default='',
    help='Path to write checkpoint file.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

def prepare_interaction_pairs(XD, XT):
    drugs = []
    targets = []

    drug = XD[0]
    drugs.append(drug)

    target=XT[0]
    targets.append(target)

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)
    return drug_data,target_data

def predict(smileStr, proteinSeq):
    if(smileStr == None):
      return('Simile input is null')
    if(proteinSeq == None):
      return('Protein Seq input is null')
    FLAGS = parseargs()
    dependencies = {
      'cindex_score': cindex_score,
      'AttentionAugmentation2D': AttentionAugmentation2D
    }
    model = load_model('data/model.h5', custom_objects=dependencies)
    FLAGS.max_seq_len  = 1000
    FLAGS.max_smi_len = 100
    dataset = DataSet(
                  seqlen = FLAGS.max_seq_len,
                  smilen = FLAGS.max_smi_len,
                  need_shuffle = False )
    
    XD, XT = dataset.parse_data(FLAGS, smileStr, proteinSeq)

    val_drugs, val_prots = prepare_interaction_pairs(XD, XT)
    # val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)

    predicted_labels = model.predict([np.array(val_drugs), np.array(val_prots)])[0][0] 
    return (predicted_labels)
    # print(model.summary())
  
#the following is test value for smile and pretein seq. You can change the values or send externally.
proteinSeq = 'MTVKTEAAKGTLTYSRMRGMVAILIAFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINLGPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMSERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAEIASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLAPEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSARHLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHFDPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL'
smileStr = 'CC1CC=CC(=O)CCCCCC2=CC(=CC(=C2C(=O)O1)O)OC'
response = predict(smileStr, proteinSeq)
print(response)
