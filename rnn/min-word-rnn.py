"""
Minimal word-level Vanilla RNN model. Written by zhi Liu refer Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import re
import pickle as pkl

# data I/O
data = open('input1.txt', 'r').read() # should be simple plain text file
words = [word.strip() for word in re.split('\s+|[,.!?;"]', data)]
uniquewords = list(set(words))
data_size, vocab_size = len(data), len(uniquewords)
print('data has %d words, %d unique.' % (data_size, vocab_size))
word_to_ix = { ch:i for i,ch in enumerate(uniquewords) }
ix_to_word = { i:ch for i,ch in enumerate(uniquewords) }

with open('char_ind', 'wb') as writerhandle:
	pkl.dump(word_to_ix, writerhandle)
	pkl.dump(ix_to_word, writerhandle)

sentenceNum = []
sentences = [line.strip() for line in open('input1.txt', 'r').readlines()]
wordlists = []
for line in sentences:
	wordlist = [word.strip() for word in re.split('\s+|[,.!?;"]', line) if line.strip()]
	wordnum = [word_to_ix[word] for word in wordlist]
	wordnum.append(word_to_ix['#'])
	sentenceNum.append(wordnum)

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
# we have the entire dataset whose length greater than 25, so we chunck it each time 25 length
seq_length = 50 # number of steps to unroll the RNN for 25
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def svaeParams(savefile):
	with open(savefile, 'wb') as writerhandle:
		pkl.dump(Wxh, writerhandle)
		pkl.dump(Whh, writerhandle)
		pkl.dump(Why, writerhandle)
		pkl.dump(bh, writerhandle)
		pkl.dump(by, writerhandle)

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
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def predict(inputs):
	questnum = [word_to_ix[word.strip()] for word in re.split('\s+|[,.!?;"]', qestinputs) if word.strip()]
	questlen = len(questnum)
	h = np.zeros((hidden_size,1))
	ixes = []
	x = np.zeros((vocab_size,1))
	x[questnum[0]] = 1

	c = 1
	for i in range(50):
		h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
		y = np.dot(Why, h) + by
		p = np.exp(y) + np.sum(np.exp())
		ix = np.random.choice(range(vocab_size), p=p.ravel())
		# ix = np.argmax(p)

		if ix == word_to_ix['#']:
			break
		x = np.zeros((vocab_size,1))
		if c < questlen:
			# if current cth is still in question, then set the currnet question word
			x[questnum[c]] = 1
		else:
			# if current cth is larger than question' length, it means that the current word is an anwser word
			x[ix] = 1
			ixes.append(ix)
		c += 1
	predTxt = ' '.join([ix_to_word[ix] for ix in ixes])
	return predTxt

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    if ix == word_to_ix['#']:
    	break
    # ix = np.argmax(p)
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

sentLen = np.size(sentenceNum)
sentId = 0;

while smooth_loss > 0.005:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  hprev = np.zeros((hidden_size,1)) # reset RNN memory

  wordlen = len(sentenceNum[sentId])
  inputs = [wordnums for wordnums in sentenceNum[sentId][0:wordlen-1]]
  targets = [wordnums for wordnums in sentenceNum[sentId][1:wordlen]]
  
  sentId += 1
  if sentId > (sentLen-2):
  	sentId = 0

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 35)
    txt = ' '.join(ix_to_word[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  seq_length = wordlen
  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 

while True:
	qestinputs = input('Question:')
	print('Question: ', qestinputs)
	questnum = [word_to_ix[word.strip()] for word in re.split('\s+|[,.!?;"]', qestinputs) if word.strip()]
	questlen = len(questnum)
	if questlen == 0: continue

	h = np.zeros((hidden_size,1))
	ixes = []
	x = np.zeros((vocab_size,1))
	x[questnum[0]] = 1

	c = 1
	for i in range(50):
		h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
		y = np.dot(Why, h) + by
		p = np.exp(y) + np.sum(np.exp())
		ix = np.random.choice(range(vocab_size), p=p.ravel())
		# ix = np.argmax(p)

		if ix == word_to_ix['#']:
			break
		x = np.zeros((vocab_size,1))
		if c < questlen:
			# if current cth is still in question, then set the currnet question word
			x[questnum[c]] = 1
		else:
			# if current cth is larger than question' length, it means that the current word is an anwser word
			x[ix] = 1
			ixes.append(ix)
		c += 1

predTxt = ' '.join([ix_to_word[ix] for ix in ixes])
print('Anwser: ', predTxt)


