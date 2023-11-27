package algo

import (
	"math"
	"sync"
)

func GridSearch(config Config, evalMethod string, scoresChan chan<- Score, wg *sync.WaitGroup) {
	defer wg.Done()

	var scores []Score

	for aBias := config.AdjBiasRange[0]; aBias <= config.AdjBiasRange[1]; aBias += 0.1 {
		for lBias := config.LrBiasRange[0]; lBias <= config.LrBiasRange[1]; lBias += 0.1 {
			// TODO: use opto logic, dummy logic for now
			fitness := math.Sin(aBias) + math.Cos(lBias)

			scores = append(scores, Score{ABias: aBias, LBias: lBias, Fitness: fitness})
		}
	}

	for _, score := range scores {
		scoresChan <- score
	}
}
