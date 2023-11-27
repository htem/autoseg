package genetic

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

func GeneticOptimization(config Config, scoresChan chan<- Score, wg *sync.WaitGroup) {
	defer wg.Done()

	rand.Seed(time.Now().UnixNano())

	// Initialize the population
	population := make([]Individual, config.PopulationSize)
	for i := range population {
		population[i] = Individual{
			AdjBias: rand.Float64()*(config.AdjBiasRange[1]-config.AdjBiasRange[0]) + config.AdjBiasRange[0],
			LrBias:  rand.Float64()*(config.LrBiasRange[1]-config.LrBiasRange[0]) + config.LrBiasRange[0],
		}
	}

	// Evolution loop
	for generation := 0; generation < config.NumGenerations; generation++ {
		fmt.Println("Generation:", generation)

		// Evaluate the fitness of each individual in the population
		fitnessValues := make([]Score, len(population))
		for i, ind := range population {
			fmt.Println("BIASES:", ind.AdjBias, ind.LrBias)
			fitness := evaluateWeightBiases(ind.AdjBias, ind.LrBias)
			fitnessValues[i] = Score{
				AdjBias: ind.AdjBias,
				LrBias:  ind.LrBias,
				Fitness: fitness,
			}
		}

		// Sort individuals by fitness (descending order)
		sortFitnessValues(fitnessValues)

		// Send the scores to the channel
		for _, score := range fitnessValues {
			scoresChan <- score
		}

		// Select parents for the next generation
		parents := fitnessValues[:config.PopulationSize/2]
		parents = parents[:len(parents):len(parents)] // Ensure capacity is the same as the length

		// Create the next generation through crossover and mutation
		offspring := make([]Individual, config.PopulationSize-len(parents))
		for i := range offspring {
			parent1 := parents[rand.Intn(len(parents))]
			parent2 := parents[rand.Intn(len(parents))]
			child := crossover(parent1, parent2)
			child = mutate(child, config.MutationRate, config.MutationStrength)
			offspring[i] = child
		}

		// Combine parents and offspring to form the new population
		population = append(parents, offspring...)
	}

	close(scoresChan)
}
