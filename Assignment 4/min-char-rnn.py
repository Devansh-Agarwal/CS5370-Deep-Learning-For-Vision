"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import pickle 

# data I/O
# data = open('input.txt', 'r').read() # should be simple plain text file
# chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# print 'data has %d characters, %d unique.' % (data_size, vocab_size)
# char_to_ix = { ch:i for i,ch in enumerate(chars) }
# ix_to_char = { i:ch for i,ch in enumerate(chars) }
with open("char-rnn-snapshot.pkl", "rb") as f:
    snapshot = pickle.load(f, encoding="latin1")

Wxh = snapshot["Wxh"]
Whh = snapshot["Whh"]
Why = snapshot["Why"]
bh = snapshot["bh"]
by = snapshot["by"]
mWxh = snapshot["mWxh"] 
mWhh = snapshot["mWhh"] 
mWhy = snapshot["mWhy"]
mbh  = snapshot["mbh"]
mby = snapshot["mby"]
chars = snapshot["chars"].tolist() 
data_size = snapshot["data_size"].tolist() 
vocab_size = snapshot["vocab_size"].tolist() 
char_to_ix = snapshot["char_to_ix"].tolist()
ix_to_char =   snapshot["ix_to_char"].tolist() 
print(by.shape)
# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
# Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
# Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
# Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
# bh = np.zeros((hidden_size, 1)) # hidden bias
# by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh[:, 0]) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by[:, 0] # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    # loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh[:, 0]), np.zeros_like(by[:, 0])
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(np.array([dy]).T, np.array([hs[t]]))
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(np.array([dhraw]).T,np.array([xs[t]]))
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n, alpha = 1):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh[:, 0])
    y = alpha * (np.dot(Why, h) + by[:, 0])
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def train():
  n, p = 0, 0
  mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
  smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
  while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0: 
      hprev = np.zeros((hidden_size,1)) # reset RNN memory
      p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
      sample_ix = sample(hprev, inputs[0], 200)
      txt = ''.join(ix_to_char[ix] for ix in sample_ix)
      print ('----\n %s \n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
    
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                  [dWxh, dWhh, dWhy, dbh, dby], 
                                  [mWxh, mWhh, mWhy, mbh, mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter 

print("part 1")

for temperature in [0.1,0.3,0.5,0.75,1,1.5,2,5]:
    print("Sample temperature = ", temperature)
    indexs = sample(np.zeros(250,), 3, 100, temperature)
    sentence = [ix_to_char[i] for i in indexs]
    print("".join(sentence))
    print("\n\n")
    
def completeString(originalString, outputSize):
  h_t = np.zeros(250, )
  x_t = np.zeros((vocab_size))
  for inp in originalString:
    x_t = np.zeros((vocab_size))
    x_t[char_to_ix[inp]] = 1
    h_t = np.tanh(np.dot(Wxh, x_t) + np.dot(Whh, h_t) + bh[:, 0])

  remainingString = []
  remainingStringSize = outputSize - len(originalString)
  for i in range(remainingStringSize):
    index = sample(h_t, np.argmax(x_t), 1, 1)[0]
    x_t = np.zeros((vocab_size))
    x_t[index] = 1
    remainingString.append(ix_to_char[index])
    h_t = np.tanh(np.dot(Wxh, x_t) + np.dot(Whh, h_t) + bh[:, 0])

  return originalString + "".join(remainingString)  

def getTopWeight(w):
  tempShape = w.shape
  tempW = np.reshape(w, (-1))
  tempW2= []
  for idx, weig in enumerate(tempW):
    tempW2.append(((idx // tempShape[1],idx % tempShape[1]), weig)) 
  tempW2 = sorted(tempW2, key=lambda x: -x[1])
  return tempW2

print("part 2")

print("\nString to be completed- The:")
for i in range(5):
  print("string number ", i, "\n")
  print(completeString("The", 100), "\n")


print("part 3")
print('Weights for “ ” after “:”')
print("Specific Weights in Why")
loss, dWxh, dWhh, dWhy, dbh, dby, hsss = lossFun([char_to_ix[":"]], [char_to_ix[" "]], np.zeros(250))
for i in getTopWeight(dWhy)[:5]:
  print("location", i[0], "gradient", i[1], "WeightValue", Why[i[0][0], i[0][1]])

print("Specific Weights in Wxh")
for i in getTopWeight(dWxh)[:5]:
  print("location", i[0], "gradient", i[1], "WeightValue", Wxh[i[0][0], i[0][1]])

print('Weights for newline after “:”')
loss, dWxh, dWhh, dWhy, dbh, dby, hsss = lossFun([char_to_ix[":"]], [char_to_ix["\n"]], np.zeros(250))
for i in getTopWeight(dWhy)[:5]:
  print("location", i[0], "gradient", i[1], "WeightValue", Why[i[0][0], i[0][1]])

print("Specific Weights in Wxh")
for i in getTopWeight(dWxh)[:5]:
  print("location", i[0], "gradient", i[1], "WeightValue", Wxh[i[0][0], i[0][1]])

print("part 4, e gives an e after it")

loss, dWxh, dWhh, dWhy, dbh, dby, hsss = lossFun([char_to_ix["e"]], [char_to_ix["e"]], np.zeros(250))

print("Specific Weights in Why")
for i in getTopWeight(dWhy)[:5]:
  print("location", i[0], "gradient", i[1], "WeightValue", Why[i[0][0], i[0][1]])

print("Specific Weights in Wxh")
for i in getTopWeight(dWxh)[:5]:
  print("location", i[0], "gradient", i[1], "WeightValue", Wxh[i[0][0], i[0][1]])

print(ix_to_char)