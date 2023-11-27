package main

import (
	"fmt"
	a "optoseg/src/algo/grid"
	"sync"
)

func main() {
	config := a.Config{
		AdjBiasRange: [2]float64{-1.0, 1.0},
		LrBiasRange:  [2]float64{-1.0, 1.0},
		NumSteps:     10,
	}

	// Create a channel to receive scores from goroutines
	scoresChan := make(chan a.Score, config.NumSteps)

	// Use a WaitGroup to wait for all goroutines to finish
	var wg sync.WaitGroup

	// Launch goroutines
	for i := 0; i < config.NumSteps; i++ {
		wg.Add(1)
		go a.GridSearch(config, "rand_voi", scoresChan, &wg)
	}

	// Close the channel when all goroutines are done
	go func() {
		wg.Wait()
		close(scoresChan)
	}()

	// Collect results from the channel
	var results []a.Score
	for score := range scoresChan {
		results = append(results, score)
	}

	for _, result := range results {
		fmt.Printf("ABias: %.2f, LBias: %.2f, Fitness: %.4f\n", result.ABias, result.LBias, result.Fitness)
	}
}
