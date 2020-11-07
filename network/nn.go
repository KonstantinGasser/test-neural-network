package network

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork interface {
	Dense(inputs, neurons int, activation func(i, j int, v float64) float64)
	Train(inputData, targetData [][]float64)
	Predict(inputData []float64) mat.Matrix
	Print()
}

type dense struct {
	inputsN    int
	neuronsN   int
	weights    *mat.Dense
	bias       *mat.Dense
	activation func(i, j int, v float64) float64
}

func NewNeuralNetwork(batchSize int, learningRate float64) NeuralNetwork {
	return &network{
		batchSize: batchSize,
		lr:        learningRate,
		layers:    []*dense{},
	}
}

type network struct {
	batchSize int
	lr        float64
	layers    []*dense
}

func (nn *network) Train(batchX, batchY [][]float64) {
	X := from2DArr(batchX).(*mat.Dense) // inputs batch X
	Y := from2DArr(batchY).(*mat.Dense) // target batch Y

	xr, _ := X.Dims()
	for i := 0; i < xr; i++ {
		_, _ = nn.backpropagation(X.RowView(i), Y.RowView(i))
	}
}

func (nn *network) backpropagation(x, y mat.Matrix) (mat.Matrix, mat.Matrix) {
	// feedward for each layer in network
	// fmt.Println("=================== FEED-FORWARD ===================")
	var activations = make([]mat.Matrix, len(nn.layers))
	var layerinputs = make([]mat.Matrix, len(nn.layers))
	activation := x
	for i, layer := range nn.layers {
		layerinputs[i] = activation

		a := apply(layer.activation, add(mul(layer.weights, activation), layer.bias))
		activations[i] = a
		activation = a
		// print(a)
	}
	// fmt.Println("====================================================")

	finalErrs := sub(y, activations[len(activations)-1])
	layerErrs := finalErrs
	// fmt.Println("================= BACKPROPPAGATION =================")
	for i := len(nn.layers) - 1; i >= 0; i-- {
		layerouts := activations[i]
		dz := apply(dsigmoid, layerouts)

		gradients := scale(nn.lr, dot(dz, layerErrs))
		deltas := mul(gradients, layerinputs[i].T())

		// nn.layers[i].weights = add(nn.layers[i].weights, deltas).(*mat.Dense)
		// nn.layers[i].bias = add(nn.layers[i].bias, gradients).(*mat.Dense)

		layerErrs = mul(nn.layers[i].weights.T(), layerErrs)
	}
	// fmt.Println("============================================")

	return nil, nil
}

func (nn *network) Predict(inputData []float64) mat.Matrix {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	activation := inputs
	for _, layer := range nn.layers {
		a := apply(layer.activation, add(mul(layer.weights, activation), layer.bias))
		activation = a.(*mat.Dense)
		// print(a)
	}
	return activation

}

// Dense adds a new layer to the network with N inputs, M neurons and a activation function
func (nn *network) Dense(inputs, neurons int, activation func(i, j int, v float64) float64) {
	layer := &dense{
		inputsN:    inputs,
		neuronsN:   neurons,
		weights:    mat.NewDense(neurons, inputs, randMat(neurons*inputs, neurons)),
		bias:       randBias(neurons, 1), // mat.NewDense(neurons, inputs, randMat(neurons*inputs, neurons)), //
		activation: activation,
	}
	nn.layers = append(nn.layers, layer)
}

func (nn *network) Print() {
	fmt.Printf("Batch-Size: %d\nLayers: %d\n", nn.batchSize, len(nn.layers))
	for _, l := range nn.layers {
		fmt.Println("=======================")
		print(l.weights)
		print(l.bias)
		fmt.Println("=======================")

	}
}
func shape(s string, m interface{}) {
	var r, c int
	switch v := m.(type) {
	case *dense:
		_ = v
		r, c = m.(*dense).weights.Dims()
	case mat.Transpose:
		r, c = m.(mat.Transpose).Dims()
	default:
		r, c = m.(*mat.Dense).Dims()
	}
	fmt.Printf("%s: (%d,%d)\n", s, r, c)
}

func print(X mat.Matrix) {
	defer fmt.Println("------")
	fmt.Println("------")
	fmt.Println(X.Dims())
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
