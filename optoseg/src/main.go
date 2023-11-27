package main

import "autoseg"

func main() {
    trainingParams := []struct {
        modelType   string
        scriptPath  string
        args        []string
    }{ // TODO: add model type arg
        {"MTLSD", "../batch_run.py", []string{"--iterations", "100000", "--raw_file", "path/to/zarr/or/n5", "--voxel_size", "33"}},
        {"ACLSD", "../batch_run.py", []string{"--iterations", "100000", "--raw_file", "path/to/zarr/or/n5", "--warmup", "100000"}},
        {"STELARR", "../batch_run.py", []string{"--iterations", "100000", "--raw_file", "path/to/zarr/or/n5", "--warmup", "100000"}},
    }

    autoseg.RunTrainingInParallel(trainingParams)
}