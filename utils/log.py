import json

def dict_pad(input_dict, pad_value=None, pad_pos='leading'):

    max_length = max(len(arr) for arr in input_dict.values())

    padded_dict = {}
    for key, array in input_dict.items():
        if pad_pos == 'leading':
            padded_dict[key] = [pad_value] * (max_length - len(array)) + array
        elif pad_pos == 'trailing':
            padded_dict[key] = array + [pad_value] * (max_length - len(array))
        else:
            raise ValueError('Invalid Padding')    
    return padded_dict

class Logging:
    def __init__(self, log_path):
        self.log_path = log_path
    
    def write(self, info):
        with open(self.log_path, 'a') as log_write:
            print(info)
            log_write.write(info)

def save_checkpoint(last_subject, last_subject_train_num,
                    subject_summary=False, overall_summary=False,
                    file_path='checkpoint.json'):
    checkpoint_data = {'last_subject': last_subject,
                       'last_subject_train_num': last_subject_train_num,
                       'subject_summary': subject_summary,
                       'overall_summary': overall_summary}
    with open(file_path, 'w') as f:
        json.dump(checkpoint_data, f)

def load_checkpoint(file_path='checkpoint.json'):
    try:
        with open(file_path, 'r') as f:
            checkpoint_data = json.load(f)
            last_subject = checkpoint_data.get('last_subject')
            last_subject_train_num = checkpoint_data.get('last_subject_train_num')
            subject_summary = checkpoint_data.get('subject_summary')
            overall_summary = checkpoint_data.get('overall_summary')
            return last_subject, last_subject_train_num, subject_summary, overall_summary
    except FileNotFoundError:
        print("Checkpoint file not found.")