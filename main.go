package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/KonstantinGasser/neural-network/network"
	"gonum.org/v1/gonum/mat"
)

func main() {
	epochs := 100000
	INPUTS := [][]float64{ // mini batch of 4 data samples each sample with 2 features
		{1, 0},
		{0, 1},
		{0, 0},
		{1, 1},
	}

	TARGETS := [][]float64{ // mini batch of 4 data samples each sample with 2 features
		{1},
		{1},
		{0},
		{0},
	}

	nn := network.NewNeuralNetwork(4, 0.1)
	// add hidden layer
	nn.Dense(2, 2, network.Sigmoid)
	// nn.Dense(2, 10, network.Sigmoid)
	// add output later
	nn.Dense(2, 1, network.Sigmoid)

	start := time.Now()
	for i := 0; i < epochs; i++ {
		rand.Seed(time.Now().UnixNano())
		a := rand.Intn(4)
		b := rand.Intn(4)
		c := rand.Intn(4)
		d := rand.Intn(4)

		in := [][]float64{INPUTS[a], INPUTS[b], INPUTS[c], INPUTS[d]}
		ou := [][]float64{TARGETS[a], TARGETS[b], TARGETS[c], TARGETS[d]}
		nn.Train(in, ou)

	}
	ellpased := time.Since(start)
	fmt.Printf("Training took: %v seconds\n", ellpased.Seconds())
	fmt.Printf("Target: 1")
	print(nn.Predict([]float64{1, 0}))
	fmt.Printf("Target: 1")
	print(nn.Predict([]float64{0, 1}))
	fmt.Printf("Target: 0")
	print(nn.Predict([]float64{0, 0}))
	fmt.Printf("Target: 0")
	print(nn.Predict([]float64{1, 1}))
}

func print(X mat.Matrix) {
	defer fmt.Println("------")
	fmt.Println("------")
	fmt.Println(X.Dims())
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}
