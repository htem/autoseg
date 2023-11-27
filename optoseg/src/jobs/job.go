package jobs

import (
    "fmt"
    "os/exec"
    "sync"
)

func TrainModel(modelType string, scriptPath string, args ...string) {
    cmd := exec.Command(scriptPath, args...)
    output, err := cmd.CombinedOutput()
    if err != nil {
        fmt.Printf("Error running %s: %v\nOutput:\n%s\n", scriptPath, err, output)
        return
    }
    fmt.Printf("Output from %s:\n%s\n", scriptPath, output)
}

func RunTrainingInParallel(trainingParams []struct {
    modelType string
    scriptPath string
    args []string
}) {
    var wg sync.WaitGroup

    for _, params := range trainingParams {
        wg.Add(1)
        go func(p struct {
            modelType string
            scriptPath string
            args []string
        }) {
            defer wg.Done()
            TrainModel(p.modelType, p.scriptPath, p.args...)
        }(params)
    }

    wg.Wait()
}
