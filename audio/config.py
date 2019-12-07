args = dict()


#project structure
args["CODE_DIRECTORY"] = "/home/SharedData/Smeet/DDP/Projects/deep_avsr/audio"
args["DATA_DIRECTORY"] = "/home/SharedData/Smeet/LRS2_Dataset"
args["PRETRAINED_MODEL_FILE"] = None #"/final/models/pretrain_005w-step_0010-wer_1.000.pt"
args["TRAINED_MODEL_FILE"] = None #"/final/models/train-step_0010-wer_1.000.pt"
args["TRAINED_LM_FILE"] = "/home/SharedData/Smeet/DDP/Projects/deep_avsr/pretrained/lrs2_language_model.pt"


#data
args["PRETRAIN_VAL_SPLIT"] = 0.05
args["NUM_WORKERS"] = 8
args["PRETRAIN_NUM_WORDS"] = 1
args["CHAR_TO_INDEX"] = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
                         "A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18, 
                         "L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14, 
                         "X":26, "Z":28}
args["INDEX_TO_CHAR"] = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8", 
                         5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
                         11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y", 
                         26:"X", 28:"Z"}


#features
args["STFT_WINDOW"] = "hamming"
args["STFT_WIN_LENGTH"] = 0.040
args["STFT_OVERLAP"] = 0.030


#training
args["SEED"] = 10
args["BATCH_SIZE"] = 32
args["STEP_SIZE"] = 16000
args["NUM_STEPS"] = 6000
args["SAVE_FREQUENCY"] = 10
args["EMPTY_CACHE"] = False


#optimizer and scheduler
args["INIT_LR"] = 1e-4
args["FINAL_LR"] = 1e-6
args["LR_SCHEDULER_FACTOR"] = 0.5
args["LR_SCHEDULER_WAIT"] = 10
args["LR_SCHEDULER_THRESH"] = 0.001
args["MOMENTUM1"] = 0.9     
args["MOMENTUM2"] = 0.999


#model
args["AUDIO_FEATURE_SIZE"] = 321
args["NUM_CLASSES"] = 39


#transformer architecture
args["PE_MAX_LENGTH"] = 2500
args["TX_NUM_FEATURES"] = 512
args["TX_ATTENTION_HEADS"] = 8
args["TX_NUM_LAYERS"] = 6
args["TX_FEEDFORWARD_DIM"] = 2048
args["TX_DROPOUT"] = 0.1


#beam search
args["BEAM_WIDTH"] = 100
args["LM_WEIGHT_ALPHA"] = 0.5
args["LENGTH_PENALTY_BETA"] = 0.1


#testing
args["TEST_DEMO_DECODING"] = "greedy"


if __name__ == '__main__':
    
    for key,value in args.items():
        print(str(key) + " : " + str(value))
