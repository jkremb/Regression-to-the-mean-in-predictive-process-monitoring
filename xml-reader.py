import datetime
import os
import csv
import pandas as pd
from xml.etree.ElementTree import ElementTree

############ some functions to help with the timestamps ###############
def calcTimeBetweenTwoEvents(firstEventTimestamp, secondEventTimestamp):
    """
    eventFirstTimestamp is chronologically first in format of 2011-10-01T00:38:44.880+02:00

    returns the difference in minutes
    """
    # event1TimestampAdjusted = firstEventTimestamp[:-6]
    # event2TimestampAdjusted = secondEventTimestamp[:-6]

    dateformat = "%Y-%m-%dT%H:%M:%S.%f"

    # TODO: fix the UTC offset issue laster
    # example = "2011-10-01T00:38:44.880+02:00"
    # d1 = datetime.datetime.strptime(example, dateformat)

    d1 = datetime.datetime.strptime(firstEventTimestamp, dateformat)
    d2 = datetime.datetime.strptime(secondEventTimestamp, dateformat)
    difference = d2-d1
    differenceInMin = int(difference.total_seconds())/60

    # print('+++++++++++++inside function, differenceinMin is')
    # print(differenceInMin)
    return differenceInMin

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
    return differenceInMin


#################### parsing the XML file ####################
file_name = 'BPI_Challenge_2012_shortened.xml'
full_file = os.path.join('data', file_name)

tree = ElementTree()

tree.parse(full_file)

################## create corresponding csv object #############

csvdata = open('data_withFasterParser.csv', 'w', newline='',encoding='utf-8')
csvwriter = csv.writer(csvdata);

col_names = ['remainingTraceTime', 'loanAmount','firstTraceEventTimestamp','lastTraceEventTimestamp','totalPrefixLength','currentPrefixLength',' W_Completeren aanvraag-COMPLETE','W_Completeren aanvraag-START','W_Nabellen offertes-COMPLETE','W_Nabellen offertes-START','A_SUBMITTED-COMPLETE','A_PARTLYSUBMITTED-COMPLETE','W_Nabellen incomplete dossiers-COMPLETE','W_Nabellen incomplete dossiers-START','W_Valideren aanvraag-COMPLETE','W_Valideren aanvraag-START','A_DECLINED-COMPLETE','W_Completeren aanvraag-SCHEDULE','A_PREACCEPTED-COMPLETE','O_SELECTED-COMPLETE','O_CREATED-COMPLETE','O_SENT-COMPLETE','W_Nabellen offertes-SCHEDULE','W_Afhandelen leads-COMPLETE','W_Afhandelen leads-START','A_ACCEPTED-COMPLETE','W_Valideren aanvraag-SCHEDULE','A_FINALIZED-COMPLETE','W_Afhandelen leads-SCHEDULE','O_CANCELLED-COMPLETE','O_SENT_BACK-COMPLETE','A_CANCELLED-COMPLETE','W_Nabellen incomplete dossiers-SCHEDULE','A_REGISTERED-COMPLETE','A_APPROVED-COMPLETE','A_ACTIVATED-COMPLETE','O_ACCEPTED-COMPLETE','O_DECLINED-COMPLETE','W_Beoordelen fraude-START','W_Beoordelen fraude-COMPLETE','W_Beoordelen fraude-SCHEDULE','W_Wijzigen contractgegevens-SCHEDULE']
csvwriter.writerow(col_names)

count = 1
for trace in tree.findall('trace'):
    traceData = []
    print(len(trace.findall('event')))
    W_Completeren_aanvraag_COMPLETE = 0
    W_Completeren_aanvraag_START = 0
    W_Nabellen_offertes_COMPLETE = 0
    W_Nabellen_offertes_START = 0
    A_SUBMITTED_COMPLETE = 0
    A_PARTLYSUBMITTED_COMPLETE = 0
    W_Nabellen_incomplete_dossiers_COMPLETE = 0
    W_Nabellen_incomplete_dossiers_START = 0
    W_Valideren_aanvraag_COMPLETE = 0
    W_Valideren_aanvraag_START = 0
    A_DECLINED_COMPLETE = 0
    W_Completeren_aanvraag_SCHEDULE = 0
    A_PREACCEPTED_COMPLETE = 0
    O_SELECTED_COMPLETE = 0
    O_CREATED_COMPLETE = 0
    O_SENT_COMPLETE = 0
    W_Nabellen_offertes_SCHEDULE = 0
    W_Afhandelen_leads_COMPLETE = 0
    W_Afhandelen_leads_START = 0
    A_ACCEPTED_COMPLETE = 0
    W_Valideren_aanvraag_SCHEDULE = 0
    A_FINALIZED_COMPLETE = 0
    W_Afhandelen_leads_SCHEDULE = 0
    O_CANCELLED_COMPLETE = 0
    O_SENT_BACK_COMPLETE = 0
    A_CANCELLED_COMPLETE = 0
    W_Nabellen_incomplete_dossiers_SCHEDULE = 0
    A_REGISTERED_COMPLETE = 0
    A_APPROVED_COMPLETE = 0
    A_ACTIVATED_COMPLETE = 0
    O_ACCEPTED_COMPLETE = 0
    O_DECLINED_COMPLETE = 0
    W_Beoordelen_fraude_START = 0
    W_Beoordelen_fraude_COMPLETE = 0
    W_Beoordelen_fraude_SCHEDULE = 0
    W_Wijzigen_contractgegevens_SCHEDULE = 0

    loanAmount = 0
    for traceElement in trace:
        # print element.attrib['key']
        # print(traceElement)
        # print(traceElement.tag)
        if 'key' in traceElement.attrib:
            if traceElement.attrib['key']=='AMOUNT_REQ':
                loanAmount = traceElement.attrib['value']

    # finding firstTraceEventTimestamp and lastTraceEventTimestamp
    firstEvent = trace.find('event')
    firstEventTimestamp = 0
    for datapoint in firstEvent:
        if datapoint.attrib['key']=='time:timestamp':
            firstEventTimestamp = datapoint.attrib['value']
            firstEventTimestamp = firstEventTimestamp[:-6]
            
    lastEvent = trace[len(trace) - 1]
    lastEventTimestamp = 0
    for datapoint in lastEvent:
        if datapoint.attrib['key']=='time:timestamp':
            lastEventTimestamp = datapoint.attrib['value']
            lastEventTimestamp = lastEventTimestamp[:-6]

    # finding total prefix lengths
    totalPrefixLength = len(trace.findall('event'))

    currentPrefixLength = 0
    # processing events
    for event in trace.findall('event'):
        currentPrefixLength = currentPrefixLength + 1

        datapointList = []
        datapointName = ""
        eventType = ""
        eventValue = ""
        for datapoint in event:
            
            if datapoint.attrib['key']=='concept:name':
                eventType = datapoint.attrib['value']

            if datapoint.attrib['key']=='lifecycle:transition':
                eventValue = datapoint.attrib['value']

        datapointName = eventType + '-' + eventValue    
        print(datapointName)
        datapointList.append(datapointName)
        print(datapointList)
            
        #### event names checking
        if('W_Completeren aanvraag-COMPLETE' in datapointList):
            W_Completeren_aanvraag_COMPLETE = W_Completeren_aanvraag_COMPLETE + 1

        if('W_Completeren aanvraag-START' in datapointList):
            W_Completeren_aanvraag_START = W_Completeren_aanvraag_START + 1

        if('W_Nabellen offertes-COMPLETE' in datapointList):
            W_Nabellen_offertes_COMPLETE = W_Nabellen_offertes_COMPLETE + 1
    
        if('W_Nabellen offertes-START' in datapointList):
           W_Nabellen_offertes_START = W_Nabellen_offertes_START + 1 

        if('A_SUBMITTED-COMPLETE' in datapointList):
            A_SUBMITTED_COMPLETE = A_SUBMITTED_COMPLETE + 1

        if('A_PARTLYSUBMITTED-COMPLETE' in datapointList):
            A_PARTLYSUBMITTED_COMPLETE = A_PARTLYSUBMITTED_COMPLETE + 1

        if('W_Nabellen incomplete dossiers-COMPLETE' in datapointList):
            W_Nabellen_incomplete_dossiers_COMPLETE = W_Nabellen_incomplete_dossiers_COMPLETE + 1

        if('W_Nabellen incomplete dossiers-START' in datapointList):
            W_Nabellen_incomplete_dossiers_START = W_Nabellen_incomplete_dossiers_START + 1

        if('W_Valideren aanvraag-COMPLETE' in datapointList):
            W_Valideren_aanvraag_COMPLETE = W_Valideren_aanvraag_COMPLETE + 1

        if('W_Valideren aanvraag-START' in datapointList):
            W_Valideren_aanvraag_START = W_Valideren_aanvraag_START + 1

        if('A_DECLINED-COMPLETE' in datapointList):
            A_DECLINED_COMPLETE = A_DECLINED_COMPLETE + 1

        if('W_Completeren aanvraag-SCHEDULE' in datapointList):
            W_Completeren_aanvraag_SCHEDULE = W_Completeren_aanvraag_SCHEDULE + 1

        if('A_PREACCEPTED-COMPLETE' in datapointList):
            A_PREACCEPTED_COMPLETE = A_PREACCEPTED_COMPLETE + 1

        if('O_SELECTED-COMPLETE' in datapointList):
            O_SELECTED_COMPLETE = O_SELECTED_COMPLETE + 1

        if('O_CREATED-COMPLETE' in datapointList):
            O_CREATED_COMPLETE = O_CREATED_COMPLETE + 1

        if('O_SENT-COMPLETE' in datapointList):
            O_SENT_COMPLETE = O_SENT_COMPLETE + 1

        if('W_Nabellen offertes-SCHEDULE' in datapointList):
            W_Nabellen_offertes_SCHEDULE = W_Nabellen_offertes_SCHEDULE + 1

        if('W_Afhandelen leads-COMPLETE' in datapointList):
            W_Afhandelen_leads_COMPLETE = W_Afhandelen_leads_COMPLETE + 1

        if('W_Afhandelen leads-START' in datapointList):
            W_Afhandelen_leads_START = W_Afhandelen_leads_START + 1

        if('A_ACCEPTED-COMPLETE' in datapointList):
            A_ACCEPTED_COMPLETE = A_ACCEPTED_COMPLETE + 1

        if('W_Valideren aanvraag-SCHEDULE' in datapointList):
            W_Valideren_aanvraag_SCHEDULE = W_Valideren_aanvraag_SCHEDULE + 1

        if('A_FINALIZED-COMPLETE' in datapointList):
            A_FINALIZED_COMPLETE = A_FINALIZED_COMPLETE + 1

        if('W_Afhandelen leads-SCHEDULE' in datapointList):
            W_Afhandelen_leads_SCHEDULE = W_Afhandelen_leads_SCHEDULE + 1

        if('O_CANCELLED-COMPLETE' in datapointList):
            O_CANCELLED_COMPLETE = O_CANCELLED_COMPLETE + 1

        if('O_SENT_BACK-COMPLETE' in datapointList):
            O_SENT_BACK_COMPLETE = O_SENT_BACK_COMPLETE+1

        if('A_CANCELLED-COMPLETE' in datapointList):
            A_CANCELLED_COMPLETE = A_CANCELLED_COMPLETE+1

        if('W_Nabellen incomplete dossiers-SCHEDULE' in datapointList):
            W_Nabellen_incomplete_dossiers_SCHEDULE = W_Nabellen_incomplete_dossiers_SCHEDULE+1

        if('A_REGISTERED-COMPLETE' in datapointList):
            A_REGISTERED_COMPLETE = A_REGISTERED_COMPLETE+1

        if('A_APPROVED-COMPLETE' in datapointList):
            A_APPROVED_COMPLETE = A_APPROVED_COMPLETE+1

        if('A_ACTIVATED-COMPLETE' in datapointList):
            A_ACTIVATED_COMPLETE = A_ACTIVATED_COMPLETE+1

        if('O_ACCEPTED-COMPLETE' in datapointList):
            O_ACCEPTED_COMPLETE = O_ACCEPTED_COMPLETE+1

        if('O_DECLINED-COMPLETE' in datapointList):
            O_DECLINED_COMPLETE = O_DECLINED_COMPLETE+1

        if('W_Beoordelen fraude-START' in datapointList):
            W_Beoordelen_fraude_START = W_Beoordelen_fraude_START+1

        if('W_Beoordelen fraude-COMPLETE' in datapointList):
            W_Beoordelen_fraude_COMPLETE = W_Beoordelen_fraude_COMPLETE+1

        if('W_Beoordelen fraude-SCHEDULE' in datapointList):
            W_Beoordelen_fraude_SCHEDULE = W_Beoordelen_fraude_SCHEDULE+1

        if('W_Wijzigen contractgegevens-SCHEDULE' in datapointList):
            W_Wijzigen_contractgegevens_SCHEDULE = W_Wijzigen_contractgegevens_SCHEDULE+1


        currentEventTimestamp = 0
        for datapoint in event:
            if datapoint.attrib['key']=='time:timestamp':
                currentEventTimestamp = datapoint.attrib['value']
                currentEventTimestamp = currentEventTimestamp[:-6]
                
        

        traceData.append(calcTimeBetweenTwoEvents(currentEventTimestamp, lastEventTimestamp))
        traceData.append(loanAmount)
        traceData.append(firstEventTimestamp)
        traceData.append(lastEventTimestamp)
        traceData.append(totalPrefixLength)
        traceData.append(currentPrefixLength)
        
        traceData.append(W_Completeren_aanvraag_COMPLETE)
        traceData.append(W_Completeren_aanvraag_START)
        traceData.append(W_Nabellen_offertes_COMPLETE)
        traceData.append(W_Nabellen_offertes_START)
        traceData.append(A_SUBMITTED_COMPLETE)
        traceData.append(A_PARTLYSUBMITTED_COMPLETE)
        traceData.append(W_Nabellen_incomplete_dossiers_COMPLETE)
        traceData.append(W_Nabellen_incomplete_dossiers_START)
        traceData.append(W_Valideren_aanvraag_COMPLETE)
        traceData.append(W_Valideren_aanvraag_START)
        traceData.append(A_DECLINED_COMPLETE)
        traceData.append(W_Completeren_aanvraag_SCHEDULE)
        traceData.append(A_PREACCEPTED_COMPLETE)
        traceData.append(O_SELECTED_COMPLETE)
        traceData.append(O_CREATED_COMPLETE)
        traceData.append(O_SENT_COMPLETE)
        traceData.append(W_Nabellen_offertes_SCHEDULE)
        traceData.append(W_Afhandelen_leads_COMPLETE)
        traceData.append(W_Afhandelen_leads_START)
        traceData.append(A_ACCEPTED_COMPLETE)
        traceData.append(W_Valideren_aanvraag_SCHEDULE)
        traceData.append(A_FINALIZED_COMPLETE)
        traceData.append(W_Afhandelen_leads_SCHEDULE)
        traceData.append(O_CANCELLED_COMPLETE)
        traceData.append(O_SENT_BACK_COMPLETE)
        traceData.append(A_CANCELLED_COMPLETE)
        traceData.append(W_Nabellen_incomplete_dossiers_SCHEDULE)
        traceData.append(A_REGISTERED_COMPLETE)
        traceData.append(A_APPROVED_COMPLETE)
        traceData.append(A_ACTIVATED_COMPLETE)
        traceData.append(O_ACCEPTED_COMPLETE)
        traceData.append(O_DECLINED_COMPLETE)
        traceData.append(W_Beoordelen_fraude_START)
        traceData.append(W_Beoordelen_fraude_COMPLETE)
        traceData.append(W_Beoordelen_fraude_SCHEDULE)
        traceData.append(W_Wijzigen_contractgegevens_SCHEDULE)
        
        print(count)
        count = count + 1
        print('.')
        print(traceData)
        csvwriter.writerow(traceData)
        traceData = []             # wipe for next trace
    
    

        

csvdata.close()

# read the csv file
# dataframe = pd.read_csv('data.csv')
# print(dataframe.shape)


############# add last event timestamp and first event timestamp to each datapoint