#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# In[2]:


class ParserModel(nn.Module):


    def __init__(self, config, word_embeddings=None, pos_embeddings=None,
                 dep_embeddings=None):
        super(ParserModel, self).__init__()

        self.config = config
        
        # These are the hyper-parameters for choosing how many embeddings to
        # encode in the input layer.  See the last paragraph of 3.1
        n_w = config.word_features_types # 18
        n_p = config.pos_features_types # 18
        n_d = config.dep_features_types # 12


        self.word_embeddings = word_embeddings
        self.pos_embeddings = pos_embeddings # TODO
        self.dep_embeddings = dep_embeddings
        
   
        self.layer1 = nn.Linear((n_w + n_p + n_d) * self.config.embedding_dim, self.config.l1_hidden_size)
        
        #add another layer
        self.layer2 = nn.Linear(self.config.l1_hidden_size,self.config.l2_hidden_size)
     
        self.dropout = nn.Dropout(self.config.keep_prob)
                
 
        self.outputlayer = nn.Linear(self.config.l2_hidden_size, self.config.num_classes)


        self.init_weights()
        
    def init_weights(self):

        initrange = 0.1
        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer2.weight.data.uniform_(-initrange, initrange)
        self.outputlayer.weight.data.uniform_(-initrange, initrange)
        self.layer1.bias.data.fill_(0)
        self.outputlayer.bias.data.fill_(0)
        self.layer2.bias.data.fill_(0)

        
    def lookup_embeddings(self, word_indices, pos_indices, dep_indices, keep_pos = 1):
        

        w_embeddings = self.word_embeddings(word_indices)
        p_embeddings = self.pos_embeddings(pos_indices)
        d_embeddings = self.dep_embeddings(dep_indices)
        
        return w_embeddings, p_embeddings, d_embeddings

    def forward(self, word_indices, pos_indices, dep_indices):

        w_embeddings, p_embeddings, d_embeddings = self.lookup_embeddings(word_indices, pos_indices, dep_indices)

        w_embeddings = w_embeddings.view(w_embeddings.shape[0],-1)
        
        p_embeddings = p_embeddings.view(p_embeddings.shape[0],-1)

        d_embeddings = d_embeddings.view(d_embeddings.shape[0],-1)

        

        input_embeddings = torch.cat((w_embeddings, p_embeddings, d_embeddings),1)
        
 
        input_layer = self.layer1(input_embeddings)
        relu_activation1 = F.relu(input_layer)
        hidden2layer = self.layer2(relu_activation1)
        relu_activation2 = F.relu(hidden2layer)
        
        
  
        drop1 = self.dropout(relu_activation2)


        output = self.outputlayer(drop1)

        return output    

        


# In[ ]:




