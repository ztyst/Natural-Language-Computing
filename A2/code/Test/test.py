# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


# t = torch.Tensor([[1, 2, 3],[2,3,4]])
# t=t.type(torch.LongTensor)
# print(t.size())
# # temp=(t == 2).nonzero()
# # print ((t == 2).nonzero())
# # a = torch.Tensor([[[1, 2, 3],[2,3,4]], [[1, 2, 3],[2,3,4]],[[1, 2, 3],[2,3,4]]])
# # for i in range(len(temp)):
# #     for j in range(len(temp[i])):
# #         a[i,j,:] = 100
# # print(a)
# torch.manual_seed(1)
# word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(5, 5)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.tensor([word_to_ix["hello"], word_to_ix["world"]], dtype=torch.long)
# # hello_embed = embeds(lookup_tensor)
# hello_embed = embeds(t)
# print(hello_embed.size())

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# seqs = ['gigantic_string','tiny_str','medium_str', '']
seqs = ['gig', 'a']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(set(''.join(seqs)))

# make model
embed = nn.Embedding(len(vocab), 10,padding_idx=0)
lstm = nn.LSTM(10, 5,bidirectional=True)

vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]

# get the length of each seq in your batch
seq_lengths = torch.LongTensor([len(seq) for seq in vectorized_seqs])

print(vectorized_seqs)
print(seq_lengths)

# dump padding everywhere, and place seqs on the left.
# NOTE: you only need a tensor as big as your longest sequence
seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
	seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

# SORT YOUR TENSORS BY LENGTH!
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]

# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)
print(seq_tensor)
# print((seq_tensor == 0.).nonzero())

# print("====================")
# pad_id_index = (seq_tensor == 0).nonzero()
# print(pad_id_index)

# embed your sequences
seq_tensor = embed(seq_tensor)
print(seq_tensor.size())

# print("==========zero===========")
# print((seq_tensor == 0.).nonzero())

# pack them up nicely
packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
print(packed_input)

print("check 1===================")
# throw them through your LSTM (remember to give batch_first=True here if you packed with it)
packed_output, (ht, ct) = lstm(packed_input)

print(packed_output)
print("============")
# unpack your output if required
output, _ = pad_packed_sequence(packed_output, padding_value = 100)
print (output.size())
print(output)
forward = output[:,:,:5]
backward = output[:,:,5:]
print(forward)
print(backward)
print(forward.size())


x = torch.zeros(5)

for i in x:
    i = 1
print(x.size())
x = x.unsqueeze(0)
print(x.size())

def test(a,b):
    return a+1, b+1, a+b

c,_= test(1,2)
print(c)
print(test(c[1], c[2]))
# x_cat = torch.cat((x, x, x), 1)
# print(x_cat.size())
# print((output==100).nonzero())
# for i in range(len(seq_lengths)):
#     tmp = seq_lengths[i].item()
#     print(output[tmp])
exit()

# Or if you just want the final hidden state?
print (ht[-1])

# REMEMBER: Your outputs are sorted. If you want the original ordering
# back (to compare to some gt labels) unsort them
_, unperm_idx = perm_idx.sort(0)
output = output[unperm_idx]
print (output)