#Import the libs
import gradio as gr
import joblib as jb
import pandas as pd
import numpy as np

#Import the necessary functions from the module functions.py
from functions import custom_encode, cria_feature

#Import the necessary files
top_50_mais_quebrados = pd.read_csv('top_50_mais_quebrados.csv')
products_id = pd.read_csv('Unique Product ID.csv')

#Import forecasting and data processing models
model_1 = jb.load('modelo 1.pkl')
model_2 = jb.load('modelo 2.pkl')
model_3 = jb.load('modelo 3.pkl')
model_4 = jb.load('modelo 4.pkl')

stack = jb.load('Stack.pkl')
pipeline = jb.load('pipeline.pkl')

#define the predict function responsible from processing the data and calculating probabilities
def predict(product_id,air_temperature,process_temperature,rotational_speed,torque,tool_wear,TWF,HDF,PWF,OSF,RNF):

    #Since each product_id has only 1 type, we need to extract the type from the product database
    type = products_id.loc[products_id['Product ID'] == product_id]['Type'].values[0]

    #Creates the mapping from each feature
    base = {'Product ID':[product_id],
            'Type':[type],
            'Air temperature [K]':[int(air_temperature)],
            'Process temperature [K]':[int(process_temperature)],
            'Rotational speed [rpm]':[int(rotational_speed)],
            'Torque [Nm]':[int(torque)],
            'Tool wear [min]':[int(tool_wear)],
            'TWF':[int(TWF)],
            'HDF':[int(HDF)],
            'PWF':[int(PWF)],
            'OSF':[int(OSF)],
            'RNF':[int(RNF)]}
    
    input = pd.DataFrame(data=base)

    #Execute the processing functions
    input['Product ID'] = custom_encode(top_50_mais_quebrados,input['Product ID'])
    input = cria_feature(input)
    input = pipeline.transform(input)

    #Execute the predict on each forecast model and averages their probabilities
    preds = np.zeros((1,4))

    preds[:, 0] = model_1.predict_proba(input)[:,1]
    preds[:, 1] = model_2.predict_proba(input)[:,1]
    preds[:, 2] = model_3.predict_proba(input)[:,1]
    preds[:, 3] = model_4.predict_proba(input)[:,1]

    pred_return = preds.mean(axis=1)

    #Returns the probabilities for each class
    return {"Will Not Fail": float(1-pred_return[0]), "Will Fail": float(pred_return[0])}

demo = gr.Interface(fn = predict,                    
                        inputs=[gr.Dropdown(choices=products_id['Product ID'].to_list(),type='value',label='Product ID'),
                                gr.Number(label='Air temperature [K]'),
                                gr.Number(label='Process temperature [K]'),
                                gr.Number(label='Rotational speed [RPM]'),
                                gr.Number(label='Torque [Nm]'),
                                gr.Number(label='Tool wear [min]'),
                                gr.Dropdown(choices=['1','0'],type='value',label='TWF'),
                                gr.Dropdown(choices=['1','0'],type='value',label='HDF'),
                                gr.Dropdown(choices=['1','0'],type='value',label='PWF'),
                                gr.Dropdown(choices=['1','0'],type='value',label='OSF'),
                                gr.Dropdown(choices=['1','0'],type='value',label='RNF')
                                ],
                                outputs='label',
                                    examples=[
                                                ['L52498',330,330,1345,70,196,'0','0','0','0','0'],
                                                ['H34252',330,330,1345,150,240,'0','0','0','0','0'],
                                                ['M20199',330,330,1345,20,55,'0','1','0','0','0'],
                                            ]
                                
) 

demo.launch()