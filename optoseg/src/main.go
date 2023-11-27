package main

import "autoseg"

func main() {
    trainingParams := []struct {
        modelType   string
        scriptPath  string
        args        []string
    }{
        {"MTLSD", "./scripts/mtlsd_train.sh", []string{"--iterations", "100000", "--raw_file", "path/to/zarr/or/n5", "--voxel_size", "33"}},
        {"ACLSD", "./scripts/aclsd_train.sh", []string{"--iterations", "100000", "--raw_file", "path/to/zarr/or/n5", "--warmup", "100000"}},
        {"STELARR", "./scripts/stelarr_train.sh", []string{"--iterations", "100000", "--raw_file", "path/to/zarr/or/n5", "--warmup", "100000"}},
        // Add more parameter sets as needed
    }

    autoseg.RunTrainingInParallel(trainingParams)
}