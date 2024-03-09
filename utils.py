import json, argparse

from typing import Any, Tuple, List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from schema import DocuExtract

def _args()-> argparse.Namespace:
    """
    Parses command line arguments and returns an argparse Namespace object.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openai-key",
        help="Openai token",
        type=str,
        required=True
    )

    parser.add_argument(
        "--document",
        help="document name without extension",
        type=str,
        required=True
    )
    args, _= parser.parse_known_args()
    return args

def loader(text: str) -> Tuple[str, List[str]]:
    """
    Loads a text document and splits it into smaller chunks.

    Args:
        text (str): The input text.

    Returns:
        Tuple[str, List[str]]: A tuple containing the entire document and a list of split documents.
    """
    loader = TextLoader(text, encoding= 'UTF-8')
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=". ")
    docs = text_splitter.split_documents(document)  
    return document, docs

def to_json(result: Any) -> str:
    """
    Converts a result object to a JSON string.

    Args:
        result (Any): The result object to be converted.

    Returns:
        str: A JSON string representation of the result.
    """
    return json.dumps(result.dict(), indent = 4)

def template() -> PromptTemplate:
    """
    Generates a prompt template for text extraction tasks.

    Returns:
        PromptTemplate: A template for text extraction prompts.
    """
    parser = PydanticOutputParser(pydantic_object=DocuExtract)

    prompt  = PromptTemplate(
    template="""
    You are an AI with advanced text extraction capabilities.
    Your task is to identify and list all the companies mentioned in the text along with their web domains. Also, identify the main topic of the article. 
    Here are some guidelines:
    - The text is enclosed within triple ticks.
    - For each company, provide the company name and its associated web domain. 
    - Remove mistake individuals, including CEOs or employees.
    - If the company domain is not available, make an educated guess based on the company name.
    
    {format_instructions}
    
    input: '''{text}'''

    """,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    ver_prompt = PromptTemplate(
    template="""
    You are an AI with advanced text analysis capabilities.
    Your task is to evaluate the relevance of each company in the json output obtained after parsing a text and remove wrong entries.
    
    Here are some guidelines:
    - Remove individuals, including CEOs or employees.
    - If a company changed its name, only keep the entry with the new name.
    - If a company has multiple entries, only keep one.
    - {topic} remain the same.
    
    Here is the original text:
    '''{text}'''
    And here are the extracted companies:
    {extracted_companies}

    {format_instructions}
    
    """,
    input_variables=["text", "extracted_companies", "topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return parser, prompt, ver_prompt