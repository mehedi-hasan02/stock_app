import pandas as pd
import pickle
import gradio as gr

with open("stock_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)


def predect_stock(stock1, stock2, stock3, stock4):

    input_df = pd.DataFrame([[
       stock1, stock2, stock3, stock4 
    ]],
        columns= ['Stock_1', 'Stock_2', 'Stock_3', 'Stock_4']
    )

    prediction = model.predict(input_df)[0]

    return f"Predicted Stock: {prediction: .3f}"


inputs = [
    gr.Number(label="Stock1", value=95),
    gr.Number(label="Stock2", value=95),
    gr.Number(label="Stock3", value=95),
    gr.Number(label="Stock4", value=95)
]

app = gr.Interface(
    fn=predect_stock,
    inputs=inputs,
    outputs="text",
    title="Stock5 Prediction"
)

app.launch(share=True)



