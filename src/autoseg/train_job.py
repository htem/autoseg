from .train.MTLSDTrain import mtlsd_train

if __name__ == "__main__":
    mtlsd_train(iterations=10000,
                data_store="../../../xray-challenge-entry/data/xpress-challenge.zarr")