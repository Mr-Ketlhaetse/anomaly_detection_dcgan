class Parameters:
    # CTGAN hyperparameters
    CTGAN_EPOCHS = 3
    CTGAN_SAMPLES = 10
    ORIGINAL_SAMPLES = 1000
    SYN_RATIO = 1.0
    SAMPLED_DATA_FILE = f"{SYN_RATIO}_sampled.csv"

    # DCGAN settings
    dcgan_train = True  # Set to True to train DCGAN model
    # Initialize DCGAN model
    latent_dim = 100
    img_channels = 3
    img_size = 64
    dcgan_epochs = 5
    dcgan_batch_size = 16
    dcgan_workers = 1

    # GridSearchCv hyperparameters
    CV = 2
    cross_validate = False

    # CNN hyperparameters
    cnn_batch_size = 16
    cnn_epochs = 10