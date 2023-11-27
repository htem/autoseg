package algo

type Config struct {
	AdjBiasRange [2]float64
	LrBiasRange  [2]float64
	NumSteps     int
}

type Score struct {
	ABias   float64
	LBias   float64
	Fitness float64
}
