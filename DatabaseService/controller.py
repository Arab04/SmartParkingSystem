from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from DatabaseService import services, schemas
from DatabaseService.database import SessionLocal

app = FastAPI()


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# FastAPI endpoints
@app.post("/contacts/", response_model=schemas.GetContact)
async def create_new_contact(contact: schemas.CreateContact, db: Session = Depends(get_db)):
    db_contact = services.create_contact(db, contact)
    return db_contact


@app.get("/contacts/{contact_id}", response_model=schemas.GetContact)
async def get_contact_by_id(contact_id: int, db: Session = Depends(get_db)):
    db_contact = services.get_contact(db, contact_id)
    if db_contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return db_contact
