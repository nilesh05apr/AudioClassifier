import argparse
from data import init_data
from train import train_ann,train,train_all
import warnings
warnings.filterwarnings('ignore')


def main(args):
    X_train,X_test,y_train,y_test = init_data()

    if args.model == 'ann':
        train_ann(X_train,X_test,y_train,y_test)
    elif args.model == 'all':
        final,results = train_all(X_train,X_test,y_train,y_test)
        print("-"*10+"Fit time and Analysis"+"-"*10)
        print(final)
        for k,v in results.items():
            print("Model:{} classification report: {}".format(k,v))
    else:
        results = train(args.model,X_train,X_test,y_train,y_test)
        print(results)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='ann'
    )
    args = parser.parse_args()
    main(args)