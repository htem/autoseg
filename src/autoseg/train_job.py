from .train import mtlsd_train, aclsd_train, stelarr_train

    
def train_model(model_type:str="MTLSD",
                iterations:int=10000,
                data_store:str="path/to/zarr/or/n5") -> None:
    match model_type.lower():
        case "mtlsd":
            mtlsd_train(iterations=iterations,
                        data_store=data_store)
        case "aclsd":
            raise NotImplementedError
        case "stelarr":
            raise NotImplementedError
