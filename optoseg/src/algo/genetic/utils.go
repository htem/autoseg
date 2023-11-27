package genetic

import (
	"math"
	"math/rand"
	"sort"
)

func evaluateWeightBiases(adjBias, lrBias float64) float64 {
	// TODO: remove placeholder logic
	return math.Sin(adjBias) + math.Cos(lrBias)
}

// crossover performs crossover by blending the weight biases of the parents
func crossover(parent1, parent2 Individual) Individual {
	alpha := rand.Float64() // blend factor

	adjBiasChild := alpha*parent1.AdjBias + (1-alpha)*parent2.AdjBias
	lrBiasChild := alpha*parent1.LrBias + (1-alpha)*parent2.LrBias

	return Individual{AdjBias: adjBiasChild, LrBias: lrBiasChild}
}

// mutate performs mutation by adding random noise to the weight biases
func mutate(individual Individual, mutationRate, mutationStrength float64) Individual {
	if rand.Float64() < mutationRate {
		// Add random noise to the weight biases
		individual.AdjBias += rand.Float64()*(2*mutationStrength) - mutationStrength
		individual.LrBias += rand.Float64()*(2*mutationStrength) - mutationStrength
	}

	return individual
}

func sortFitnessValues(values []Score) {
	sortSlice := func(i, j int) bool {
		return values[i].Fitness > values[j].Fitness
	}
	sort.Slice(values, sortSlice)
}
