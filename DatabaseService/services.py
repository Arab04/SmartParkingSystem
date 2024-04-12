from datetime import datetime

from sqlalchemy.orm import Session
from DatabaseService.models import Contact
from DatabaseService.schemas import GetContact, CreateContact
from DatabaseService import database as _database
from DatabaseService.database import SessionLocal


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_tables():
    return _database.Base.metadata.create_all(bind=_database.engine)


def get_contact(db: Session, contact_id: int):
    return db.query(Contact).filter(Contact.id == contact_id).first()


def create_contact(db: Session, contact: CreateContact):
    db_contact = Contact(**contact.dict())
    db.add(db_contact)
    db.commit()
    db.refresh(db_contact)
    return db_contact


def save_model(contact: CreateContact):
    db, = get_db()

