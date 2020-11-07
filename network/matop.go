package network

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func add(a, b mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	if br, bc := b.Dims(); r != br && c != bc {
		panic("matrix.Add: shape of A must match shape of B")
	}
	res := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res.Set(i, j, (a.At(i, j) + b.At(i, j)))
		}
	}
	return res
}

func sub(a, b mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	if br, bc := b.Dims(); r != br && c != bc {
		panic("matrix.Sub: shape of A must match shape of B")
	}
	res := mat.NewDense(r, c, nil)
	res.Sub(a, b)
	return res
}

func dot(a, b mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	// _, c := b.Dims()

	res := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res.Set(i, j, (a.At(i, j) * b.At(i, j)))
		}
	}
	return res
}

func mul(a, b mat.Matrix) mat.Matrix {
	ar, ac := a.Dims()
	_, bc := b.Dims()
	// print(a)
	// print(b)

	res := mat.NewDense(ar, bc, nil)

	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			var sum float64
			for k := 0; k < ac; k++ {
				sum += a.At(i, k) * b.At(k, j)
			}
			res.Set(i, j, sum)
		}
	}

	return res
}

func scale(scaler float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Scale(scaler, m)
	return res
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	res := mat.NewDense(r, c, nil)
	res.Apply(fn, m)
	return res
}

func randMat(flatSize int, v int) []float64 {

	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(float64(v)),
		Max: 1 / math.Sqrt(float64(v)),
	}

	data := make([]float64, flatSize)
	for i := 0; i < flatSize; i++ {
		data[i] = dist.Rand()
	}
	return data
}

func randBias(r, c int) *mat.Dense {

	data := randMat(c, r*c)

	res := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		res.SetRow(i, data)
	}
	return res
}

func from2DArr(arr [][]float64) mat.Matrix {
	res := mat.NewDense(len(arr), len(arr[0]), nil)
	for i := 0; i < len(arr); i++ {
		res.SetRow(i, arr[i])
	}
	return res
}
