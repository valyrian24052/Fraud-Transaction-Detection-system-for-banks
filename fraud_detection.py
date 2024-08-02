
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


class FraudDetection:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.transfer_dict = {}
        self.cash_out_dict = {}

    def clean_data(self):
        unique_amounts_with_count_1 = self.df[self.df['isFraud'] == 1]['amount'].value_counts()
        amounts_to_drop = unique_amounts_with_count_1[unique_amounts_with_count_1 == 1].index
        self.df = self.df[~((self.df['isFraud'] == 1) & (self.df['amount'].isin(amounts_to_drop)))]

    def create_transfer_cashout_dicts(self):
        transfer_transactions = self.df[self.df['type'] == 'TRANSFER']
        cash_out_transactions = self.df[self.df['type'] == 'CASH_OUT']

        self.transfer_dict = transfer_transactions.groupby(['step', 'amount']).apply(lambda x: x.index.tolist()).to_dict()
        self.cash_out_dict = cash_out_transactions.groupby(['step', 'amount']).apply(lambda x: x.index.tolist()).to_dict()

    def identify_pairs(self):
        self.df['pair'] = 0
        transfer_transactions = self.df[self.df['type'] == 'TRANSFER']

        for _, transfer_row in transfer_transactions.iterrows():
            transfer_step = transfer_row['step']
            transfer_amount = transfer_row['amount']
            if (transfer_step, transfer_amount) in self.cash_out_dict:
                cash_out_indices = self.cash_out_dict[(transfer_step, transfer_amount)]
                self.df.loc[transfer_row.name, 'pair'] = 1
                self.df.loc[cash_out_indices, 'pair'] = 1

        self.df.loc[(self.df['type'] == 'CASH_OUT') & (self.df['pair'] == 0), 'pair'] = 1
        self.df['pair'] = self.df['pair'].fillna(0).astype(int)

    def add_features(self):
        self.df['istype_co_tf'] = (self.df['type'].isin(['CASH_OUT', 'TRANSFER'])).astype(int)
        self.df['amount_gt_10000000'] = (self.df['amount'] > 10000000).astype(int)

    def train_model(self):
        X = self.df[['newbalanceOrig', 'pair', 'istype_co_tf', 'amount_gt_10000000']]
        y = self.df['isFraud']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

        model = XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_output = classification_report(y_test, y_pred)

        print(f'Accuracy: {accuracy}')
        print(classification_report_output)

        result_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        result_df = pd.concat([result_df, X_test], axis=1)

        return result_df

    def exploratory_data_analysis(self):
        print(self.df['type'].value_counts())
        print(self.df[self.df['isFlaggedFraud'] == 1]["type"].value_counts())
        print(self.df[self.df['isFraud'] == 1]["type"].value_counts())
        print(self.df[self.df['isFraud'] == 1]["newbalanceOrig"].value_counts())
        print(self.df[self.df['isFraud'] == 1]["newbalanceOrig"].nunique())
        print(self.df[self.df['amount'] == 10000000.00])
        print(self.df[self.df['isFraud'] == 1]["amount"].value_counts().tail(44))
        print(self.df[self.df['isFraud'] == 1]["amount"].value_counts().head(10))
        print(self.df[self.df['isFraud'] == 1].count())

        amount_value_counts = self.df[self.df['isFraud'] == 1]['amount'].value_counts()
        new_df = amount_value_counts.reset_index()
        new_df.columns = ['amount', 'count']

        print(self.df[self.df['newbalanceOrig'] == 0])
        print(new_df['count'].value_counts())
        print(self.df[(self.df['isFraud'] == 1) & (self.df['amount'] == 0)])
        print(self.df['amount'].value_counts())
        print(self.df[(self.df['type'] == 'CASH_OUT') | (self.df['type'] == 'TRANSFER')])
        print(self.df[self.df['isFraud'] == 1])
        print(self.df[self.df['nameOrig'] == "C553264065"])

