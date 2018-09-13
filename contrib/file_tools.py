import os

tfm_root = '/home/asema/work/'
directory = tfm_root+'PUBLICATION/score_results/'


def writef(preds_batch, output):
    mode = 'a' if os.path.isfile(output) else 'w'
    with open(output, mode) as f:
        for it_preds in preds_batch:
            for it, class_score in enumerate(it_preds):            
                line = str(class_score)
                if it < len(it_preds)-1:
                    line += ","
                f.write(line)
            f.write("\n")


def compose_file_name(suffix, net, it):
    filename = "{}_net_{}_it_{}.txt".format(suffix,net,it)
    return "{d}{f}".format(d=directory, f=filename)


def read_predictions_batch(filename):
    mode = 'r'

    with open(filename, mode) as f:   
        return [map(float, line.split(",")) for line in f]


def read_ground_truth_batch(filename):
    mode = 'r'

    with open(filename, mode) as f:   
        return [map(int, list(line)) for line in f][0]
