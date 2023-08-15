import os, sys
from langchain.document_loaders import PyPDFLoader

from langchain.llms import OpenAI
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
import pandas as pd
import re
import replicate
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain import LLMChain

import json



#Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



#Function to extract data from text
def extracted_data(pages_data):

    template = """Extract all the following values : invoice no., Description, Quantity, date, 
        Unit price , Amount, Total, email, phone number and address from this data: {pages}

        Expected output: remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00','Amount': '2200.00','Total': '2200.00','Email': 'Santoshvarma0988@gmail.com','Phone number': '9999999999','Address': 'Mumbai, India'}}
        """
    
    #Creating the final PROMPT
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)

    #The below code will be used when we want to use OpenAI model
    '''
    llm = OpenAI(temperature=.7)
    full_response=llm(prompt_template.format(pages=pages_data))
    '''
    

    #The below code will be used when we want to use LLAMA 2 model,  we will use Replicate for hosting our model...
    '''
    response = replicate.run('replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1', 
                           input={"prompt":prompt_template.format(pages=pages_data) ,
                                  "temperature":0.1, "top_p":0.9, "max_length":512, "repetition_penalty":1})
    '''
    

    #C Transformers offers support for various open-source models, 
    #among them popular ones like Llama, GPT4All-J, MPT, and Falcon.


    #C Transformers is the Python library that provides bindings for transformer models implemented in C/C++ using the GGML library

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.1,
                                'top_p':0.9, 
                                'repetition_penalty':1})
    
    
    #Generating the response using LLM
    # response = llm(prompt_template.format(pages=pages_data))
    LLM_Chain=LLMChain(prompt=prompt_template, llm=llm)

    response = LLM_Chain.run("Extract all the following values : invoice no., Description, Quantity, date, Unit price , Amount, Total, email, phone number and address from this data")
    # response = llm.predict(prompt_template.format(pages=pages_data), max_length=512)
    print('-'*80)
    print(f'RESPONSE \n {response}')

    full_response = ''
    for item in response:
        full_response += item
    

    #print(full_response)
    return full_response


# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list):
# def create_docs():

    
    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                   'Description': pd.Series(dtype='str'),
                   'Quantity': pd.Series(dtype='str'),
                   'Date': pd.Series(dtype='str'),
	                'Unit price': pd.Series(dtype='str'),
                   'Amount': pd.Series(dtype='int'),
                   'Total': pd.Series(dtype='str'),
                   'Email': pd.Series(dtype='str'),
	                'Phone number': pd.Series(dtype='str'),
                   'Address': pd.Series(dtype='str')
                    })

    # document=[]
    # for file in os.listdir("Invoice/"):
    #     if file.endswith(".pdf"):
    #         pdf_path="./Invoice/"+file
    #         document.append(pdf_path)
            
       

    # user_pdf_list = document       
    for filename in user_pdf_list:
        
        print(filename)
        raw_data = get_pdf_text(filename)
        print('-'*80)
        print(f' EXTRACTED RAW DATA')
        print(raw_data)
        
        

        llm_extracted_data = extracted_data(raw_data)
        print('-'*80)
        print(f'LLM EXTRACTED DATA \n {llm_extracted_data}')
        #Adding items to our list - Adding data & its metadata

        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            # Converting the extracted text to a dictionary
            # data_dict = eval('{' + extracted_text + '}')
            # Remove leading and trailing spaces
            extracted_text = extracted_text.strip()
            data_str = extracted_text.strip('"')

            # Replace single quotes with double quotes for valid JSON format
            data_str = data_str.replace("'", "\"")

            # Parse the string as JSON to get a dictionary
            data_dict = json.loads("{" + data_str + "}")

            print(data_dict)
        else:
            print("No match found.")

        
        df=df.append([data_dict], ignore_index=True)
        print("********************DONE***************")
        #df=df.append(save_to_dataframe(llm_extracted_data), ignore_index=True)

    df.head()
    return df

# if __name__ == '__main__':
#     create_docs()