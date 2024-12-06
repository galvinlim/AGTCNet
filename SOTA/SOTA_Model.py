from . import models

def SOTA_Models(model_name):
    # Select the model
    if(model_name == 'ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.ATCNet( 
            # Dataset parameters
            n_classes = 4, 
            in_chans = 22, 
            in_samples = 1125, 
            # Sliding window (SW) parameter
            n_windows = 5, 
            # Attention (AT) block parameter
            attention = 'mha', # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1 = 16,
            eegn_D = 2, 
            eegn_kernelSize = 64,
            eegn_poolSize = 7,
            eegn_dropout = 0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 32,
            tcn_dropout = 0.3, 
            tcn_activation='elu'
            )     
    elif(model_name == 'TCNet-Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes = 4)      
    elif(model_name == 'EEG-TCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes = 4)          
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes = 4) 
    elif(model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps = 1125 , n_features = 22, n_outputs = 4)
    elif(model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    elif(model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes = 4 , Chans = 22, Samples = 1125)
    elif (model_name == 'DB-ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DB_ATCNet(
            # Dataset parameters
            n_classes=4,
            in_chans=22,
            in_samples=1125,

            # Attention Dual-branch Convolution block (ADBC) parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            drop1=0.35,
            depth1=2,
            depth2=4,

            # Sliding window (SW) parameter
            n_windows=5,

            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'

            # Temporal convolutional Fusion Network block (TCFN) parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            drop2=0.1,
            drop3=0.15,
            drop4=0.15,

            tcn_activation='elu',
        )
    elif (model_name == 'DB-ATCNet-EEGMMIDB-4Class'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DB_ATCNet(
            # Dataset parameters
            n_classes=4,
            in_chans=64,
            in_samples=640,

            # Attention Dual-branch Convolution block (ADBC) parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            drop1=0.35,
            depth1=2,
            depth2=4,

            # Sliding window (SW) parameter
            n_windows=5,

            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'

            # Temporal convolutional Fusion Network block (TCFN) parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            drop2=0.1,
            drop3=0.15,
            drop4=0.15,

            tcn_activation='elu',
        )
    elif (model_name == 'DB-ATCNet-EEGMMIDB-2Class'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DB_ATCNet(
            # Dataset parameters
            n_classes=2,
            in_chans=64,
            in_samples=640,

            # Attention Dual-branch Convolution block (ADBC) parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            drop1=0.35,
            depth1=2,
            depth2=4,

            # Sliding window (SW) parameter
            n_windows=5,

            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'

            # Temporal convolutional Fusion Network block (TCFN) parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            drop2=0.1,
            drop3=0.15,
            drop4=0.15,

            tcn_activation='elu',
        )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model