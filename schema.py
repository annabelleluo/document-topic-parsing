from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

class Company(BaseModel):
    company_name: str = Field(...)
    company_domain: Optional[str] = None
 
class DocuExtract(BaseModel):
    related_companies: List[Company] = Field(...)
    topic: str = Field(...)

