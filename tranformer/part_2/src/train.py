#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import pdb,math
import matplotlib.pyplot as plt
class char_tokenizer:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        # TODO: calculate the vocab size and create a dictionary that maps each character to a unique integer
        self.char_to_idx={}; self.idx_to_char={}
        for char in corpus:
            self.idx_to_char[len(self.char_to_idx)]=char
            self.char_to_idx[char]=len(self.char_to_idx)
        self.n_vocab=len(self.char_to_idx)
        # End of your code

    def encode(self, string: str):
        # TODO: convert a string into a list of integers and return, using the dictionary you created above
        return [self.char_to_idx[char] for char in string]
        # End of your code
 
    def decode(self, codes: List[int]):
        # TODO: convert a list of integers into a string and return, using the dictionary you created above
        outstring=''
        for idx in codes:
            outstring+=self.idx_to_char[idx]
        return outstring
        # End of your code

class PositionalEncoder(nn.Module):
    def __init__(self,max_seq_len=1000):
        #n_embd为嵌入维度
        super(PositionalEncoder, self).__init__()
        position=torch.zeros([max_seq_len,n_embd])
        #行为pos
        for pos in range(max_seq_len):
            #列为i
            for i in range(n_embd,2):
                #公式
                position[pos,i]=torch.sin(pos/(10000**(2*i/n_embd)))
                position[pos,i+1]=torch.cos(pos/(10000**((2*i+1)/n_embd)))
        self.position=position.to(device)

    def forward(self,x):
        #对嵌入向量进行放缩，原因在前面的理论部分已讲
        x=x*math.sqrt(n_embd)
        seq_len=x.shape[1]
        #调整大小
        x=x+self.position[:seq_len,:]
        return x
    
class Head(nn.Module):
    """single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # TODO: create three linear layers, Key, Query, and Value, each of which maps from n_embd to head_size
        #       and assign them to self.Key, self.Query, and self.Value, respectively
        self.Key=nn.Linear(n_embd,head_size).to(device)
        self.Query=nn.Linear(n_embd,head_size).to(device)
        self.Value=nn.Linear(n_embd,head_size).to(device)
        # End of your code
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))    #假设自己也会对自己产生注意力
        self.tril=self.tril.to(device)
        self.mask=(self.tril==0).to(device)


    def forward(self, inputs):
        # TODO: implement the forward function of the head
        #       the input is a tensor of shape (batch, time, n_embd)
        #       the output should be a tensor of shape (batch, time, head_size)
        #       you may use the tril buffer defined above to mask out the upper triangular part of the affinity matrix
        K=self.Key(inputs)          #(batch,time,head_size)
        Q=self.Query(inputs)        #(batch,time,head_size)
        V=self.Value(inputs)        #(batch,time,head_size)
        Score=torch.matmul(K,torch.transpose(Q,1,2))    #(batch,time,time)
        # # mask_S=self.tril*Score                        #(time,time)
        # attention=(Score/math.sqrt(K.shape[-1]))
        # attention[:,self.mask]=-1e9
        attention=torch.masked_fill(input=Score/math.sqrt(K.shape[-1]),mask=(self.tril==0),value=-1e9)
        attention=attention.softmax(dim=-1)     #(batch,time,time)
        out=torch.matmul(attention,V)   #(batch,time,head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        #TODO: implement heads and projection
        self.n_heads=n_heads
        self.head_list=[]
        for i in range(n_heads):
            self.head_list.append(Head(head_size))
        self.projection=nn.Linear(n_heads*head_size,n_embd)

        # End of your code
    def forward(self, inputs):
        #TODO: implement the forward function of the multi-head attention
        self.out_list=[]   
        for i in range(self.n_heads):
            self.out_list.append(self.head_list[i](inputs))
        out = torch.cat(self.out_list,dim=-1)   #(batch,time,n_heads*head_size)
        return self.projection(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        #TODO: implement the feed-forward network
        self.hidden=forward_hidden
        self.net = nn.Sequential(
            nn.Linear(n_embd,self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden,n_embd)
        )

    def forward(self, inputs):
        return self.net(inputs)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # TODO: implement the block of transformer using the MultiHeadAttention and 
        # FeedForward modules, along with the layer normalization layers
        self.MultiHeadAttention=MultiHeadAttention(n_heads,head_size)
        self.LayerNorm=nn.LayerNorm(n_embd)
        self.FeedForward=FeedForward(n_embd)
        self.dropout=nn.Dropout(dropout)
        # End of your code
    def forward(self, inputs):
        #TODO: implement the forward function of the block, you may refer to the docs of this experiment
        out=self.MultiHeadAttention(inputs)
        mid=self.LayerNorm(inputs+self.dropout(out))      #add+Norm
        out=self.FeedForward(mid)
        out=self.LayerNorm(mid+self.dropout(out))             #add+Norm
        # End of your code
        return out


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: create the embedding table, the stack of blocks, the layer normalization layer, 
        # and the linear layers.
        self.char_embeddings=nn.Embedding(n_vocab,n_embd)
        self.net=nn.Sequential(
            Block(n_embd,n_heads),
            Block(n_embd,n_heads),
            Block(n_embd,n_heads),
            Block(n_embd,n_heads),
            Block(n_embd,n_heads),
            Block(n_embd,n_heads)
        )
        self.cls=nn.Sequential(
            # nn.Linear(n_embd,n_vocab)
            nn.Linear(n_embd,forward_hidden),
            nn.ReLU(),
            nn.Linear(forward_hidden,n_vocab)
        )
        self.pos_encode=PositionalEncoder()
        # End of your code

    def forward(self, inputs, labels=None):
        inputs=self.char_embeddings(inputs)
        inputs=self.pos_encode(inputs)
        out=self.net(inputs)
        logits=self.cls(out)
        
        if labels is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)
            labels = labels.view(batch * time )
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # TODO: generatenew tokens from the transformer, using the inputs as the context,
        #  and return the generated tokens with length of max_new_tokens
        batch,len_input=inputs.shape
        embeds=self.char_embeddings(inputs).to(device) #(batch,time,embd)
        seq=torch.zeros(batch,len_input+max_new_tokens,n_embd).to(device)
        seq[:,:len_input,:]=embeds
        seq=self.pos_encode(seq)       #位置嵌入
        for ite in range(len_input,len_input+max_new_tokens):
            # generates new tokens by iteratively sampling from the model's predicted probability distribution, 
            # concatenating the sampled tokens to the input sequence, and returning the updated sequence.
            if(ite<=block_size):
                seq[:,ite,:]+=self.net(seq[:,:block_size,:])[:,ite-1,:]
            else:
                seq[:,ite,:]+=self.net(seq[:,ite-block_size:ite,:])[:,-1,:] #(batch,time,n_embd)
        seq=self.cls(seq[:,len_input:len_input+max_new_tokens,:])
        # print(f'shape1={inputs.shape},shape2={seq.shape}')
        # pdb.set_trace()
        outputs=torch.cat([inputs,torch.argmax(seq,dim=-1)],dim=1)        #(batch,time)
        # print(outputs.shape)
        # End of your code
        return outputs

#修改了y为类别向量
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

@torch.no_grad()
def generate(model):
    model.eval()
    context=torch.tensor(encode(text[:start_tokens]),dtype=torch.long).unsqueeze(0).to(device) #(batch,block_size)
    print(decode(model.generate(context, max_new_tokens)[0].tolist()))


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        
        if iter % eval_interval == eval_interval-1:
            losses = estimate_loss(model)
            print(
                f"step {iter+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            train_loss.append(losses['train'].item()); val_loss.append(losses['val'].item()); epoch_list.append(iter+1)
            generate(model)
        model.train()
        inputs, labels = get_batch("train")
        inputs =inputs.to(device)   
        labels =labels.to(device)
        logits, loss = model(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    

# define the hyperparameters
batch_size = 16
block_size = 256
max_iters = 20000 # set the number of training iterations as you like
eval_interval = 200
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 256
n_heads = 8
n_layers = 6
head_size=256            #注意力头大小
forward_hidden=1024      #FeedForward和最后的分类器中的隐藏层位数
dropout=0.2             #dropout大小
start_tokens=256        #生成文本的起始片段长度
max_new_tokens=300      #生成文本的长度

# read the dataset
with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))


# initialize the vocabulary
tokenizer = char_tokenizer(chars)
encode = tokenizer.encode
decode = tokenizer.decode
n_vocab = tokenizer.n_vocab

# separate the dataset into train and validation
train_data = torch.tensor(encode(text[: -len(text) // 10]), dtype=torch.long)
val_data = torch.tensor(encode(text[-len(text) // 10 :]), dtype=torch.long)

train_loss=[]
val_loss=[] 
epoch_list=[]  

# define the model
model = Transformer().to(device)
train(model)
generate(model)

plt.plot(epoch_list,train_loss,'g',label='train')
plt.plot(epoch_list,val_loss,'b',label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss of transformer')
plt.legend()
plt.savefig('transformer_loss.png')