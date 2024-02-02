import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, accuracy_score

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.precision', 6)

# Layer 1 Security Variables - "Too Many Transactions in a row"/"Above Certain Amount in a period"/"Previous Chargeback"
#   Set safety time threshold
consecutive_transaction_threshold = 10000  # 5 minutes
#   Set the threshold for consecutive transactions by the same user (in seconds)
max_consecutive_transactions = 3  # Not in Use
#   Set the threshold for sum of transactions by the same user within time threshold
transactions_sum_threshold = 1000  # Not in Use

# Creating flask app
app = Flask(__name__)


@app.route('/fraud_check', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = {"transaction_id": float(data["transaction_id"]),
                      "merchant_id": float(data["merchant_id"]),
                      "user_id": float(data["user_id"]),
                      "card_number": str(data["card_number"]),
                      "transaction_date": str(data["transaction_date"]),
                      "transaction_amount": float(data["transaction_amount"]),
                      "device_id": float(data["device_id"])}

        input_df = pd.DataFrame([input_data])
        input_df["has_cbk"] = None

        prediction_df = preprocess(input_df)
        prediction = prediction_df.iloc[0, prediction_df.columns.get_loc("anomaly_prediction")]

        if prediction == 1:
            recommendation = "deny"
        else:
            recommendation = "approve"

        response = {"transaction_id": data["transaction_id"],
                    "recommendation": recommendation}

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

def preprocess(input_df):
    # Load dataset
    file_path = r"C:\Users\OFF\Desktop\CloudWalk Case\transactional-sample.csv"
    df_1 = pd.read_csv(file_path)
    df = pd.concat([input_df, df_1], ignore_index=True)

    # Data preprocessing ---------------------------------------------------------------------------------------------------
    df['card_number'] = df['card_number'].replace('\*', '', regex=True)
    df['transaction_date_unix'] = pd.to_datetime(df['transaction_date'], format='%Y-%m-%dT%H:%M:%S.%f').astype('int64') // 10 ** 9
    df = df.drop(columns=["transaction_date"])
    df.fillna(0, inplace=True)

    # L1s Optimization -----------------------------------------------------------------------------------------------------
    #   Add elapsed time since last user_id transaction in UNIX
    df.sort_values(by=['user_id', 'transaction_date_unix'], inplace=True)
    df['time_difference_unix'] = df.groupby('user_id')['transaction_date_unix'].diff().fillna(0)

    #   Not needed to use these repeated loops, but in order to make it easier to understand the logic, I separated them
    #   Prioritizing sharing the logic than latency
    #   Loop through DF to verify many transactions in a row
    df['consecutive_count'] = 0
    for row_index in range(len(df)):
        # Define Columns Index
        user_id_index = df.columns.get_loc("user_id")
        time_difference_index = df.columns.get_loc("time_difference_unix")
        consecutive_count_index = df.columns.get_loc("consecutive_count")
        # Retrieve current row values
        user_id = df.iloc[row_index, user_id_index]
        time_difference = df.iloc[row_index, time_difference_index]
        consecutive_count = df.iloc[row_index, consecutive_count_index]
        # Retrieve previous row values
        prev_user_id = df.iloc[row_index - 1, user_id_index]
        prev_time_difference = df.iloc[row_index - 1, time_difference_index]
        prev_consecutive_count = df.iloc[row_index - 1, consecutive_count_index]
        # Count Consecutive Transactions within Interval threshold
        if time_difference < consecutive_transaction_threshold and user_id == prev_user_id:
            df.iloc[row_index, consecutive_count_index] = prev_consecutive_count + 1
        else:
            df.iloc[row_index, consecutive_count_index] = 0

    #   Loop through DF to verify sum of transactions in a given period
    df['amount_in_period'] = 0
    for row_index in range(len(df)):
        # Define Columns Index
        user_id_index = df.columns.get_loc("user_id")
        time_difference_index = df.columns.get_loc("time_difference_unix")
        amount_period_index = df.columns.get_loc("amount_in_period")
        amount_index = df.columns.get_loc("transaction_amount")
        # Retrieve current row values
        user_id = df.iloc[row_index, user_id_index]
        time_difference = df.iloc[row_index, time_difference_index]
        amount_in_period = df.iloc[row_index, amount_period_index]
        amount = df.iloc[row_index, amount_index]
        # Retrieve previous row values
        prev_user_id = df.iloc[row_index - 1, user_id_index]
        prev_time_difference = df.iloc[row_index - 1, time_difference_index]
        prev_amount_in_period = df.iloc[row_index - 1, amount_period_index]
        prev_amount = df.iloc[row_index - 1, amount_index]
        # Sum of Transactions within Interval threshold
        if time_difference < consecutive_transaction_threshold and user_id == prev_user_id:
            # If it is the first transaction within the threshold, it must retrieve prev_amount to sum, else maintain the sum
            if prev_time_difference == 0:
                df.iloc[row_index, amount_period_index] = amount + prev_amount
            else:
                df.iloc[row_index, amount_period_index] = amount + prev_amount_in_period
        else:
            df.iloc[row_index, amount_period_index] = 0

    #   Loop through DF to check if there is a previous chargeback
    df['cbk_lock'] = False
    for row_index in range(len(df)):
        # Define Columns Index
        user_id_index = df.columns.get_loc("user_id")
        has_cbk_index = df.columns.get_loc("has_cbk")
        cbk_lock_index = df.columns.get_loc("cbk_lock")

        # Retrieve current row values
        user_id = df.iloc[row_index, user_id_index]
        has_cbk = df.iloc[row_index, has_cbk_index]

        # Retrieve previous row values
        prev_user_id = df.iloc[row_index - 1, user_id_index]
        prev_has_cbk = df.iloc[row_index - 1, has_cbk_index]

        # Sum of Transactions within Interval threshold
        if user_id == prev_user_id:
            # If it is the first transaction within the threshold, it must retrieve prev_amount to sum, else maintain the sum
            if prev_has_cbk == True:
                df.iloc[row_index, cbk_lock_index] = True
        else:
            df.iloc[row_index, cbk_lock_index] = False
    # L1s Optimization END--------------------------------------------------------------------------------------------------
    # Data preprocessing END------------------------------------------------------------------------------------------------

    # L2 ML MODEL ----------------------------------------------------------------------------------------------------------
    # Separate the data for analysis
    legit = df[df.has_cbk == False]
    fraud = df[df.has_cbk == True]

    # Model Training using Isolation Forest
    isolation_forest = IsolationForest(contamination=0.15)
    isolation_forest.fit(df.drop("has_cbk", axis=1))

    # Predict anomalies
    predictions = isolation_forest.predict(df.drop("has_cbk", axis=1))
    # Convert anomaly labels (-1) to standard binary labels (0 for normal, 1 for anomaly)
    df["anomaly_prediction"] = (predictions == -1).astype(int)
    # Convert "has_cbk" column to binary labels (False -> 0, True -> 1)
    df["has_cbk"] = df["has_cbk"].astype(int)

    # Accuracy Analysis
    print("\nHISTORICAL TRANSACTION DATA -------------------------------------------------------------------------------------------------------")
    print("Legit Transactions: ", legit.shape)
    print("Legit Transactions Statistics")
    print(legit.transaction_amount.describe())
    print("\nFraud Transactions: ", fraud.shape)
    print("Fraud Transactions Statistics")
    print(fraud.transaction_amount.describe())
    y_df = df[df["transaction_id"] != input_df.iloc[0, input_df.columns.get_loc("transaction_id")]]
    y_true = y_df["has_cbk"]
    y_pred = y_df["anomaly_prediction"]
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    auc_score = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(precision_score(y_true, y_pred))
    print("Accuracy Score: ", acc)
    print("AUC Score: ", auc_score)
    print("HISTORICAL TRANSACTION DATA -------------------------------------------------------------------------------------------------------")

    #   Retrieve
    input_transaction_id = input_df.iloc[0, input_df.columns.get_loc("transaction_id")]
    print("\nCURRENT TRANSACTION DATA --------------------------------------------------------------------------------------------------------")
    print("REQUEST TRANSACTION ID: ", input_transaction_id)
    print(df[df["transaction_id"] == input_transaction_id])
    print("CURRENT TRANSACTION DATA ----------------------------------------------------------------------------------------------------------")
    return df[df["transaction_id"] == input_transaction_id]

if __name__ == "__main__":
    app.run(debug=True)



