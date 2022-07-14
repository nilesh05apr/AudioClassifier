import argparse
from ensurepip import bootstrap
from data import init_data
from train import train_ann,train,train_all
from metrics import metrics, plot_fit_performance, plot_nofit_performance, performance
import warnings
warnings.filterwarnings('ignore')


def main(args):
    X_train,X_test,y_train,y_test = init_data()

    if args.model == 'ann':
        ann_acc = train_ann(X_train,X_test,y_train,y_test)
        print('-'*10+"ANN Performance"+'-'*10)
        print("ANN Accuracy: {}".format(ann_acc))
    elif args.model == 'all':
        final,results = train_all(X_train,X_test,y_train,y_test)
        print("-"*10+"Fit time and Analysis"+"-"*10)
        print(final)
        for k,v in results.items():
            print("Model: {} classification report: {}".format(k,v))
        if args.metric == True:
            r_long_fit,r_long_unfit,bs_df = metrics(final)
            plot_fit_performance(r_long_fit)
            plot_nofit_performance(r_long_unfit)
            bs_fit_df = performance(r_long_fit,bs_df)
            bs_unfit_df = performance(r_long_unfit,bs_df)
            print("-"*10+"Bootstrap performace"+"-"*10)
            print("Booststrap With Fit Performance")
            print(bs_fit_df)
            print("Booststrap with UnFit Performace")
            print(bs_unfit_df)
    else:
        results = train(args.model,X_train,X_test,y_train,y_test)
        print("-"*10+"Fit time and Analysis"+"-"*10)
        print(results)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='ann'
    )
    parser.add_argument(
        '--metric',
        type=bool,
        default=False
    )
    args = parser.parse_args()
    main(args)