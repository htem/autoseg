package genetic

type Config struct {
	AdjBiasRange     [2]float64
	LrBiasRange      [2]float64
	PopulationSize   int
	NumGenerations   int
	MutationRate     float64
	MutationStrength float64
}

// Individual struct represents an individual in the population
type Individual struct {
	AdjBias float64
	LrBias  float64
}

// Score struct represents the fitness of an individual
type Score struct {
	AdjBias float64
	LrBias  float64
	Fitness float64
}
