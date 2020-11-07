package network

import "math"

func Sigmoid(i, j int, v float64) float64 {
	return 1.0 / (1 + math.Exp(-1*v))
}

func dsigmoid(i, j int, v float64) float64 {
	return v * (1 - v)
}

func Relu(i, j int, v float64) float64 {
	return math.Max(0.0, v)
}

func drelu(i, j, int, v float64) float64 {
	if v > 0 {
		return 1.0
	}
	return 0.0
}
