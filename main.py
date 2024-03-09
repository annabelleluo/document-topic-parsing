import json

from langchain_openai import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler

from utils import loader, to_json, template, _args

if __name__ == '__main__':
    args=_args()
    
    # Load and split the document into multiple chunkss
    original_doc, split_docs = loader(f"{args.document}.txt") 

    # LLM model 
    llm = ChatOpenAI(
        model_name = 'gpt-3.5-turbo',
        openai_api_key=args.openai_key,
        temperature = 0,
        streaming=True,
        callbacks=[StdOutCallbackHandler()]

    )
    # import the various prompts and pydantic parser
    parser, init_prompt, verification_prompt = template()

    # define a chain
    map_chain = init_prompt | llm | parser | (lambda x:{"text":split_docs, "extracted_companies":x.related_companies, "topic":x.topic}) | verification_prompt | llm | parser | to_json

    # invoke the chain on the chunk
    result = map_chain.invoke({"text":split_docs})
    
    # dump the output
    json_result = json.loads(result)
    with open(f'{args.document}_result.json', 'w') as f:
        json.dump(json_result, f)
    print(result)

