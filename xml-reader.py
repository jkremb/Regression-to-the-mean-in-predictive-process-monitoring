import datetime
import os
import csv
import pandas as pd
from xml.etree.ElementTree import ElementTree

### This creates a csv file with header: traceTimeInMin,loanAmount,events
### events is a list of events containing raw info, for example
## [['COMPLETE', 'A_SUBMITTED', '2011-10-01T00:38:44.546+02:00'], ['COMPLETE', 'A_PARTLYSUBMITTED', '2011-10-01T00:38:44.880+02:00']]


#################### parsing the XML file ####################
file_name = 'BPI_Challenge_2012_modified.xml'
full_file = os.path.join('data', file_name)

tree = ElementTree()

tree.parse(full_file)

################## create corresponding csv object #############


def calcTotalTraceTime(trace):

    traceStartTime = trace[0].attrib['value']
    traceStartTime = traceStartTime[:-6]
    lastEvent = trace[len(trace) - 1]

    lastEventTimestamp = 0
    for datapoint in lastEvent:
        if datapoint.attrib['key']=='time:timestamp':
            lastEventTimestamp = datapoint.attrib['value']
    lastEventTimestamp = lastEventTimestamp[:-6]

    dateformat = "%Y-%m-%dT%H:%M:%S.%f"

    # TODO: fix the UTC offset issue laster
    # example = "2011-10-01T00:38:44.880+02:00"
    # d1 = datetime.datetime.strptime(example, dateformat)
    
    d1 = datetime.datetime.strptime(traceStartTime, dateformat)
    d2 = datetime.datetime.strptime(lastEventTimestamp, dateformat)
    difference = d2-d1
    differenceInMin = int(difference.total_seconds())/60
    print(difference)
    print(differenceInMin)
    return differenceInMin

csvdata = open('data.csv', 'w', newline='',encoding='utf-8')
csvwriter = csv.writer(csvdata);

col_names = ['traceTimeInMin',  'loanAmount', 'numberOfEvents','events']
csvwriter.writerow(col_names)

for trace in tree.findall('trace'):
    traceData = []

    numberOfEvents = 0;
    traceTime = calcTotalTraceTime(trace)
    traceData.append(traceTime)

    for traceElement in trace:
        # print element.attrib['key']
        # print(traceElement)
        # print(traceElement.tag)
        if 'key' in traceElement.attrib:
            if traceElement.attrib['key']=='AMOUNT_REQ':
                loanAmount = traceElement.attrib['value']
                traceData.append(loanAmount)

    # processing event number
    for event in trace.findall('event'):
        numberOfEvents = numberOfEvents + 1;
    traceData.append(numberOfEvents)

    # processing events
    eventList = []
    for event in trace.findall('event'):

        eventDetails = []
        for datapoint in event:

            if datapoint.attrib['key']=='lifecycle:transition':
                eventValue = datapoint.attrib['value']
                eventDetails.append(eventValue)

            if datapoint.attrib['key']=='concept:name':
                eventType = datapoint.attrib['value']
                eventDetails.append(eventType)
            
            if datapoint.attrib['key']=='time:timestamp':
                eventTimestamp = datapoint.attrib['value']
                eventDetails.append(eventTimestamp)

        eventList.append(eventDetails)

    traceData.append(eventList)

    csvwriter.writerow(traceData)

csvdata.close()

# read the csv file
dataframe = pd.read_csv('data.csv')
print(dataframe.shape)



# confirmed that all traces 13087 traces under events are printed
# element = 1
# for trace in traces:
#     print(element)
#     events = trace.findall('event')
#     for event in events:
#         print(event.tag)
#     element = element + 1
