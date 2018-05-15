import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

class MultiAttrEncoder(nn.Module):
    def __init__(self, inp_size, out_size, hidden_size, cuda, rnn_cell='gru'):
      super(MultiAttrEncoder, self).__init__()      
      self.hidden_size = hidden_size
      self.inp_size = inp_size
      self.out_size = out_size
      self.cuda_flag = cuda
      if rnn_cell.lower() == 'lstm':
        self.rnn_cell = nn.LSTMCell
      elif rnn_cell.lower() == 'gru':
        self.rnn_cell = nn.GRUCell
      else:
        raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
      
      self.rnn = nn.GRUCell(inp_size, hidden_size)
      # self.rnn = self.rnn_cell(inp_size, hidden_size)
      self.output_layer = nn.Linear(hidden_size,out_size*2)

    def forward(self, input_var):
      """
      Applies a multi-layer RNN to an input sequence.

      Args:
          input_var (batch, seq_len, inp_size): tensor containing the features of the input sequence.          

      Returns: output
          - **output** (batch, seq_len, output_size, 2): output_size pair of logits
      """
      bs, seq_len, _ = input_var.size()
      

      inp = input_var.transpose(0,1).float()
      h = Variable(torch.zeros(bs,self.hidden_size, dtype=inp.dtype))
      outputs = []
      if self.cuda_flag:
        h = h.cuda()

      for i, tok in enumerate(inp):        
        h = self.rnn(tok,h)
        outputs.append(self.output_layer(h).view(bs,1,self.out_size, 2))

      return torch.cat(outputs,dim=1)
