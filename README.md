s
# Simple Neural Network: This is me getting my feet wet with Neural Networks

## Meta Data
Hyposhisis: Learning the MNIST data set to recognice handwritten numbers
Input Layer: 784 (28x28 Pixel)
Hidden Layer: N neurons
Output Layer: 10 neurons (0-9)
batch_size: s

## Flow
```
for batch in batchs(batch_size):
  inputs = batch                  | matrix of (batch_size, 784)
  targets;                        | matrix of (batch_size, 1)

  // feed-forward
  hidden = weights_IH X inputs + bias_IH.T
  a_hidden = activation(hidden)

  output = weights_HO X a_hidden + bias_HO
  a_output = activation(output)

  // backpropagation
  outErrs = targets - a_output

  gradientOutput = lr * (a'(a_output) X outErrs)
  weights_HO = weights_HO + (gradientOutput * a_hidden.T)
  bias_HO = bias_HO + gradientOutput

  hiddenErrs = weights_HO.T X outErrs
  gradientHidden = lr * (a'(a_hidden) X hiddenErrs)
```
# test-neural-network
# test-neural-network
# test-neural-network
