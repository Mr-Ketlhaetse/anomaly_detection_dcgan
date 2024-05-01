class Parameters:
    # CTGAN hyperparameters
    CTGAN_EPOCHS = 3
    CTGAN_SAMPLES = 1000
    ORIGINAL_SAMPLES = 10000
    SYN_RATIO = 1.0
    SAMPLED_DATA_FILE = f"{SYN_RATIO}_sampled.csv"

    # DCGAN settings
    dcgan_train = True  # Set to True to train DCGAN model
    # Initialize DCGAN model
    latent_dim = 100
    img_channels = 3
    img_size = 64
    dcgan_epochs = 5
    dcgan_batch_size = 64
    # GridSearchCv hyperparameters
    CV = 2