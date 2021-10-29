import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import stats
import config.constants as consts
import numpy as np
import physionet_metrics


# need to change these after re-running experiments to get separate baseline and unlearning results!!
filenames = ['results/shaoxing_baseline_predictions.npz', 'results/shaoxing_unlearning_predictions.npz', 'results/baseline_test_cpsc_xl_predictions.npz',
             'results/unlearning_test_cpsc_xl_predictions.npz', 'results/baseline_test_georgia_predictions.npz', 'results/unlearning_test_georgia_predictions.npz',
             'results/baseline_test_ptb_xl_predictions.npz', 'results/unlearning_test_ptb_xl_predictions.npz']

models = ['baseline', 'unlearning'] * 4
datasets = ['shaoxing'] * 2 + ['cpsc'] * 2 + ['georgia'] * 2 + ['ptb'] * 2
columns = ['model', 'dataset', 'class', 'accuracy', 'f1', 'tp', 'tn', 'fp', 'fn', 'cm']

for i in range(len(filenames)):
    data = np.load(filenames[i])
    y_true = data['y_true']
    y_pred = data['y_pred']
    tp, tn, fp, fn = stats.per_class_counts(y_true, y_pred)
    total_examples = len(y_true)
    acc = stats.per_class_accuracy(tp, tn, total_examples)
    f1 = stats.per_class_f1_score(tp, fp, fn)
    cm = physionet_metrics.challenge_metric(y_true, y_pred)
    model = [models[i]] * len(acc)
    dataset = [datasets[i]] * len(acc)
    d = {columns[0]: model, columns[1]: dataset, columns[2]: consts.classes_24, columns[3]: acc, columns[4]: f1,
         columns[5]: tp, columns[6]: tn, columns[7]: fp, columns[8]: fn, columns[9]: cm}
    if i == 0:
        df = pd.DataFrame(data=d)
    else:
        df_temp = pd.DataFrame(data=d)
        df = df.append(df_temp, ignore_index=True)

# view of per-class performance broken down by model and dataset
#sns.catplot(x='model', y='fp', hue='dataset', data=df, col='class', col_wrap=6, kind='point', markers=['.', '^', '*', 'D'])
#plt.show()
# sns.set_style("whitegrid")
# sns.catplot(x='model', y='cm', hue='dataset', data=df, kind='point', markers=['.', '^', '*', 'D'])
