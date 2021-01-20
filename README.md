# Regression-to-the-mean-in-predictive-process-monitoring

The following is a predictive process monitoring program which aims to
predict the 

label = remaining time left on trace(remainingTraceTime)
features/attributes
    1. current number of events in trace
    2. presence of event classifications (0 or 1)
    3. loan amount
    

<string key="concept:name" value="173784"/> this is the id of a trace, not important for model


python version 3.7.9

first use xml-reader.py to extract relevant data into data.csv
then use data-processor.py to transform event data into numeric form
then use model-builder.py to train your model from processedData.csv