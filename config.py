class Parameters:
    # CTGAN hyperparameters
    CTGAN_EPOCHS = 1
    CTGAN_SAMPLES = 1000
    ORIGINAL_SAMPLES = 10000
    SYN_RATIO = 0.2
    SAMPLED_DATA_FILE = f"{SYN_RATIO}_sampled.csv"

    # DCGAN settings
    dcgan_train = True  # Set to True to train DCGAN model
    # Initialize DCGAN model
    latent_dim = 100
    img_channels = 3
    img_size = 64
    dcgan_epochs = 10
    dcgan_batch_size = 16
    dcgan_workers = 1

    # GridSearchCv hyperparameters
    CV = 2
    cross_validate = True

    # CNN hyperparameters
    cnn_batch_size = 16
    cnn_epochs = 10