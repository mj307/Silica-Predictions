import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Streamlit app title with custom font and color
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            color: #d44e2f; /* Iron Ore Red */
            font-family: 'Verdana', sans-serif;
            font-weight: bold;
        }
        .subtitle {
            font-size: 28px;
            color: #e09d31; /* Mining Gold */
            font-family: 'Arial', sans-serif;
            margin-top: 20px;
        }
        .advice-box {
            background-color: #3b0a45; /* Deep purple */
            color: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 18px;
            line-height: 1.6;
            font-family: 'Arial', sans-serif;
            max-width: 800px;
        }
        .warning {
            color: #f6e200; /* Yellow for warning */
            font-weight: bold;
        }
        .good {
            color: #4CAF50; /* Green for good results */
        }
        .data-table {
            background-color: #292b2c;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title">Silica Concentration Prediction and Advice</div>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    data_0 = pd.read_csv(uploaded_file)


    st.write("Data Preview:")
    st.dataframe(data_0.head(), use_container_width=True)


    df = data_0.sample(n=50000, random_state=1)


    cols_to_change = df.columns.tolist()[1:]
    for col in cols_to_change:
        df.loc[:, col] = df[col].astype(str).str.replace(',', '.', regex=True)
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric and handle errors


    X = df.drop(["% Silica Concentrate", "date"], axis=1)
    y = df["% Silica Concentrate"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predictions
    y_preds = model.predict(X_test)


    results = pd.DataFrame({"Preds": y_preds, "Actual": y_test})


    threshold = 3.25


    advice_list = []
    for i, pred in enumerate(y_preds[:20]):  
        if pred > threshold:
            advice = f"**Warning**: Silica concentration is {pred:.2f}% and is above the {threshold}% threshold."
            advice += "\n**Advice**: Consider adjusting air flow, water flow, or pulp quality to reduce silica."
            advice_class = "warning"
        else:
            advice = f"Silica concentration is {pred:.2f}%, within the acceptable range. No action needed."
            advice_class = "good"
        

        advice_list.append(f'<div class="advice-box {advice_class}">{advice}</div>')

   
    st.write("First 20 Rows of Data (Including Predictions and Advice):")
    

    first_20_results = results.head(20)
    first_20_data = df.head(20)
    

    combined_data = pd.concat([first_20_data, first_20_results["Preds"]], axis=1)


    st.dataframe(combined_data, use_container_width=True)

   
    st.markdown("<h2 class='subtitle'>Advice for Each Prediction</h2>", unsafe_allow_html=True)
    st.markdown(''.join(advice_list), unsafe_allow_html=True)


    st.subheader("Actual vs Predicted Silica Concentration")
    fig, ax = plt.subplots()
    ax.scatter(y_test[:20], y_preds[:20], color='orange', label="Predictions")
    ax.plot([min(y_test[:20]), max(y_test[:20])], [min(y_test[:20]), max(y_test[:20])], color='red', linestyle='--', label="Ideal")
    ax.set_xlabel("Actual Silica Concentration")
    ax.set_ylabel("Predicted Silica Concentration")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to proceed.")
