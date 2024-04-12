import datetime as dt
import pydantic as pd


class BaseSchema(pd.BaseModel):
    name: str
    last_name: str
    phone: str
    email: str


class GetContact(BaseSchema):
    id: int
    # datetime: dt.datetime


class CreateContact(BaseSchema):
    datetime: str
    pass
