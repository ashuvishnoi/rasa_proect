import pandas as pd
from ast import literal_eval
from cdqa.pipeline.cdqa_sklearn import QAPipeline
from rasa_sdk import Action

# read the csv file
df = pd.read_csv('/Users/ashutoshvishnoi/Data_Science/intern_2/products/BankCurrupcy/qa_system/sample_data2/'
                 'answs.csv', converters={'paragraphs': literal_eval})

# Load the bert qa model
cdqa_pipeline = QAPipeline(reader='/Users/ashutoshvishnoi/Data_Science/intern_2/products/BankCurrupcy/'
                                  'qa_system/models/bert_qa.joblib')

ques_dict = []


cdqa_pipeline.fit_retriever(df)
print('-----Model loaded successfully and fit successfully----')


class ActionGetNewst(Action):

    def name(self):
        return 'action_get_bertAns'

    def run(self, dispatcher, tracker, domain):
        query = tracker.latest_message['text']
        prediction = cdqa_pipeline.predict(query, n_predictions=3)

        # dispatcher.utter_message('query: {}\n'.format(query))
        # dispatcher.utter_message('answer: {}\n'.format(prediction[0]))
        # dispatcher.utter_message('title: {}\n'.format(prediction[1]))

        dispatcher.utter_message('answer:   {}\n'.format(prediction[0][2]))
        # dispatcher.utter_message('answer2:   {}\n'.format(prediction[1][2]))
        # dispatcher.utter_message('answer3:   {}\n'.format(prediction[2][2]))

        '''
        dispatcher.utter_message('Please enter your rating regarding the above answer in the scale of 1 to 5')
        rating = int(tracker.['text'])
        print(f'rating given by user for query: {query} = {rating}')

        if rating > 3:
            ques_dict.append({'query:': query, 'answer': prediction[2]})
        else:
            a = 'Put it for model improvement'
            
        '''
        return[]



'''
from rasa_sdk import Action


class ActionGetNewst(Action):

    def name(self):
        return 'action_get_dogs_list'

    def run(self, dispatcher, tracker, domain):

        dispatcher.utter_message(f'dogs lists are {["puppy", "don", "adom"]}')
        return[]


# rasa run actions
'''
